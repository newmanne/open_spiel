# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DQN agent implemented in PyTorch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_spiel.python import rl_agent
from open_spiel.python.examples.ubc_utils import single_action_result, turn_based_size, handcrafted_size, sor_profit_index
import time
from absl import logging
from cachetools import cached, LRUCache, TTLCache
from cachetools.keys import hashkey

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state is_final_step legal_actions_mask iteration upper_bound")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9

MIN_MAPPED_UTILITY = -1
MAX_MAPPED_UTILITY = 1

class ReplayBuffer(object):
  """ReplayBuffer of fixed size with a FIFO replacement policy.

  Stored transitions can be sampled uniformly.

  The underlying datastructure is a ring buffer, allowing 0(1) adding and
  sampling.
  """

  def __init__(self, replay_buffer_capacity):
    self._replay_buffer_capacity = replay_buffer_capacity
    self._data = []
    self._next_entry_index = 0

  def add(self, element):
    """Adds `element` to the buffer.

    If the buffer is full, the oldest element will be replaced.

    Args:
      element: data to be added to the buffer.
    """
    if len(self._data) < self._replay_buffer_capacity:
      self._data.append(element)
    else:
      self._data[self._next_entry_index] = element
      self._next_entry_index += 1
      self._next_entry_index %= self._replay_buffer_capacity

  def sample(self, num_samples):
    """Returns `num_samples` uniformly sampled from the buffer.

    Args:
      num_samples: `int`, number of samples to draw.

    Returns:
      An iterable over `num_samples` random elements of the buffer.

    Raises:
      ValueError: If there are less than `num_samples` elements in the buffer
    """
    if len(self._data) < num_samples:
      raise ValueError("{} elements could not be sampled from size {}".format(
          num_samples, len(self._data)))
    # return random.sample(self._data, num_samples)
    # Modified to use np sampling and not python random
    sampled_indices = np.random.choice(len(self._data), num_samples, replace=False)
    samples = []
    for s in sampled_indices:
      samples.append(self._data[s])
    return samples

  def __len__(self):
    return len(self._data)

  def __iter__(self):
    return iter(self._data)

  def clear(self):
    self._data = []
    self._next_entry_index = 0

class SonnetLinear(nn.Module):
  """A Sonnet linear module.

  Always includes biases and only supports ReLU activations.
  """

  def __init__(self, in_size, out_size, activate_relu=True):
    """Creates a Sonnet linear layer.

    Args:
      in_size: (int) number of inputs
      out_size: (int) number of outputs
      activate_relu: (bool) whether to include a ReLU activation layer
    """
    super(SonnetLinear, self).__init__()
    self._activate_relu = activate_relu
    stddev = 1.0 / math.sqrt(in_size)
    mean = 0
    lower = (-2 * stddev - mean) / stddev
    upper = (2 * stddev - mean) / stddev
    # Weight initialization inspired by Sonnet's Linear layer,
    # which cites https://arxiv.org/abs/1502.03167v3
    # pytorch default: initialized from
    # uniform(-sqrt(1/in_features), sqrt(1/in_features))
    self._weight = nn.Parameter(
        torch.Tensor(
            stats.truncnorm.rvs(
                lower, upper, loc=mean, scale=stddev, size=[out_size,
                                                            in_size])))
    self._bias = nn.Parameter(torch.zeros([out_size]))

  def forward(self, tensor):
    y = F.linear(tensor, self._weight, self._bias)
    return F.relu(y) if self._activate_relu else y


class MLP(nn.Module):
  """A simple network built from nn.linear layers."""

  def __init__(self,
               input_size,
               hidden_sizes,
               output_size,
               num_players, 
               num_products,
               activate_final=False):
    """Create the MLP.

    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      output_size: (int) number of outputs
      activate_final: (bool) should final layer should include a ReLU
    """

    super(MLP, self).__init__()

    self.num_players = num_players
    self.num_products = num_products
    self.num_actions = output_size
    self.lb = turn_based_size(self.num_players)
    self.ub = self.lb + handcrafted_size(self.num_actions, self.num_products)


    self._layers = []
    # Hidden layers
    for size in hidden_sizes:
      self._layers.append(SonnetLinear(in_size=input_size, out_size=size))
      input_size = size
    # Output layer

    self._layers.append(
        SonnetLinear(
            in_size=input_size,
            out_size=output_size,
            activate_relu=activate_final))

    self.model = nn.ModuleList(self._layers)

  def reshape_infostate(self, infostate_tensor):
    # MLP doesn't need to reshape infostates: just use flat tensor
    return torch.tensor(infostate_tensor[self.lb: self.ub])

  def prep_batch(self, infostate_list):        
    """
    Prepare a list of infostate tensors to be used as an input to the network.

    Args:
    - infostate_list: a list of infostate tensors, each with shape (num_features)
    
    Returns: (num_examples, num_features) tensor of features
    """
    return torch.vstack(infostate_list)

  def forward(self, x):
    for layer in self.model:
      x = layer(x)
    return x


class DQN(rl_agent.AbstractAgent):
  """DQN Agent implementation in PyTorch.

  See open_spiel/python/examples/breakthrough_dqn.py for an usage example.
  """

  def __init__(self,
               player_id,
               num_actions,
               num_players,
               q_network_model=MLP,
               q_network_args={},
               replay_buffer_capacity=10000,
               batch_size=128,
               replay_buffer_class=ReplayBuffer,
               learning_rate=0.01,
               update_target_network_every=1000,
               learn_every=10,
               discount_factor=1.0,
               min_buffer_size_to_learn=1000,
               epsilon_start=1.0,
               epsilon_end=0.1,
               epsilon_decay_duration=int(1e6),
               optimizer_str="sgd",
               loss_str="mse",
               upper_bound_utility=None,
               lower_bound_utility=None,
               double_dqn=True,
               device='cpu',
               ):
    """Initialize the DQN agent."""

    # This call to locals() is used to store every argument used to initialize
    # the class instance, so it can be copied with no hyperparameter change.
    self._kwargs = locals()

    self.upper_bound_utility = upper_bound_utility
    self.lower_bound_utility = lower_bound_utility
    self._num_players = num_players
    self._double_dqn = double_dqn
    if self._double_dqn:
      logging.info(f"Double DQN activated for player {player_id}")

    self.player_id = player_id
    self._num_actions = num_actions
    self._batch_size = batch_size
    self._update_target_network_every = update_target_network_every
    self._last_network_copy = -1
    self._learn_every = learn_every
    self._min_buffer_size_to_learn = min_buffer_size_to_learn
    self._discount_factor = discount_factor

    self._epsilon_start = epsilon_start
    self._epsilon_end = epsilon_end
    self._epsilon_decay_duration = epsilon_decay_duration
    self._train_time = 0 # Poor man's profiling

    if not isinstance(replay_buffer_capacity, int):
      raise ValueError("Replay buffer capacity not an integer.")
    self._replay_buffer = replay_buffer_class(replay_buffer_capacity)
    self._prev_timestep = None
    self._prev_action = None
    self._prev_action_greedy = False

    # Step counter to keep track of learning, eps decay and target network.
    self._step_counter = 0
    self._iteration = 0
    self._cache = LRUCache(maxsize=5000)

    # Keep track of the last training loss achieved in an update step.
    self._last_loss_value = torch.tensor([0])

    # Create the Q-network instances
    self._device = device
    logging.info(f"Creating DQN using device: {self._device} for player {player_id}")
    self._q_network = q_network_model(**q_network_args).to(self._device)
    self._target_q_network = q_network_model(**q_network_args).to(self._device)

    if loss_str == "mse":
      self.loss_class = F.mse_loss
    elif loss_str == "huber":
      self.loss_class = F.smooth_l1_loss
    else:
      raise ValueError("Not implemented, choose from 'mse', 'huber'.")

    if optimizer_str == "adam":
      self._optimizer = torch.optim.Adam(
          self._q_network.parameters(), lr=learning_rate)
    elif optimizer_str == "sgd":
      self._optimizer = torch.optim.SGD(
          self._q_network.parameters(), lr=learning_rate)
    else:
      raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")


  def step(self, time_step, is_evaluation=False, add_transition_record=True):
    """Returns the action to be taken and updates the Q-network if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.
      add_transition_record: Whether to add to the replay buffer on this step.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """

    # Act step: don't act at terminal info states or if its not our turn.
    if (not time_step.last()) and (time_step.is_simultaneous_move() or self.player_id == time_step.current_player()):
      legal_actions = time_step.observations["legal_actions"][self.player_id]
      # Note that the TakeSingleActionDecorator is problematic here, because the transition would not added. So we do this instead, to avoid running the network
      if len(legal_actions) == 1: # Don't run the network for a single choice
        action, probs = single_action_result(legal_actions, self._num_actions)
      else:
        info_state_flat = time_step.observations["info_state"][self.player_id]
        epsilon = self._get_epsilon(is_evaluation)
        action, probs = self._epsilon_greedy(info_state_flat, legal_actions, epsilon)
    else:
      action = None
      probs = []

    # Don't mess up with the state during evaluation.
    if not is_evaluation:
      self._step_counter += 1

      if self._iteration % self._learn_every == 0:
        self._last_loss_value = self.learn()

      if self._iteration % self._update_target_network_every == 0 and self._last_network_copy < self._iteration:
        # logging.info(f"Copying target Q network for player {self.player_id} after {self._iteration} iterations")
        # state_dict method returns a dictionary containing a whole state of the module.
        self._target_q_network.load_state_dict(self._q_network.state_dict())
        self._last_network_copy = self._iteration

      if self._prev_timestep and add_transition_record:
        # We may omit record adding here if it's done elsewhere.
        self.add_transition(self._prev_timestep, self._prev_action, time_step)

      if time_step.last():  # prepare for the next episode.
        self._prev_timestep = None
        self._prev_action = None
        self._iteration += 1
        return
      else:
        self._prev_timestep = time_step
        self._prev_action = action

    return rl_agent.StepOutput(action=action, probs=probs)


  def add_transition(self, prev_time_step, prev_action, time_step):
    """Adds the new transition using `time_step` to the replay buffer.

    Adds the transition from `self._prev_timestep` to `time_step` by
    `self._prev_action`.

    Args:
      prev_time_step: prev ts, an instance of rl_environment.TimeStep.
      prev_action: int, action taken at `prev_time_step`.
      time_step: current ts, an instance of rl_environment.TimeStep.
    """
    assert prev_time_step is not None
    reward = time_step.rewards[self.player_id]

    legal_actions = (time_step.observations["legal_actions"][self.player_id])
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0

    info_state_flat = prev_time_step.observations["info_state"][self.player_id][:]

    ### FOR DYNAMIC BOUNDING. ONLY WORKS WHEN BIDDER UTILITY CAN BE COMPUTED INDEPENDENTLY
    if time_step.last():
      max_profit_still_possible = reward
    else:
      idx = sor_profit_index(self._num_players)
      sor_profits = np.array(info_state_flat[idx : idx + self._num_actions])
      max_profit_still_possible = max(sor_profits[legal_actions])
    ### END

    info_state = self._q_network.reshape_infostate(info_state_flat).to(self._device)

    next_info_state_flat = time_step.observations["info_state"][self.player_id][:]
    next_info_state = self._q_network.reshape_infostate(next_info_state_flat).to(self._device)


    transition = Transition(
        info_state=info_state,
        action=prev_action,
        reward=reward,
        next_info_state=next_info_state,
        is_final_step=float(time_step.last()),
        legal_actions_mask=legal_actions_mask,
        iteration=self._iteration,
        upper_bound=max_profit_still_possible
        )
    self._replay_buffer.add(transition)

  def _epsilon_greedy(self, info_state_flat, legal_actions, epsilon):
    """Returns a valid epsilon-greedy action and valid action probs.

    Action probabilities are given by a softmax over legal q-values.

    Args:
      info_state: hashable representation of the information state.
      legal_actions: list of legal actions at `info_state`.
      epsilon: float, probability of taking an exploratory action.

    Returns:
      A valid epsilon-greedy action and valid action probabilities.
    """
    probs = np.zeros(self._num_actions)
    if np.random.rand() < epsilon:
      action = np.random.choice(legal_actions)
      probs[legal_actions] = 1.0 / len(legal_actions)
      self._prev_action_greedy = True
    else:
      self._prev_action_greedy = False
      key = hashkey(tuple(info_state_flat))
      val = self._cache.get(key)
      if val is not None:
        return val
      else:
        info_state = self._q_network.reshape_infostate(info_state_flat)
        info_state = self._q_network.prep_batch([info_state]).to(self._device)
        with torch.no_grad():
          q_values = self._q_network(info_state).cpu().detach()[0]
        legal_q_values = q_values[legal_actions]
        action = legal_actions[torch.argmax(legal_q_values)]
        probs[action] = 1.0
        self._cache[key] = (action, probs)
    return action, probs

  def _get_epsilon(self, is_evaluation, power=1.0):
    """Returns the evaluation or decayed epsilon value."""
    if is_evaluation:
      return 0.0
    decay_steps = min(self._iteration, self._epsilon_decay_duration)
    decayed_epsilon = (
        self._epsilon_end + (self._epsilon_start - self._epsilon_end) *
        (1 - decay_steps / self._epsilon_decay_duration)**power)
    return decayed_epsilon

  def learn(self):
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """
    self._clear_cache()
    start = time.time()

    if (len(self._replay_buffer) < self._batch_size or
        len(self._replay_buffer) < self._min_buffer_size_to_learn):
      # return None
      return torch.tensor([0])

    transitions = self._replay_buffer.sample(self._batch_size)

    info_states = self._q_network.prep_batch([t.info_state for t in transitions]).to(self._device)
    next_info_states = self._q_network.prep_batch([t.next_info_state for t in transitions]).to(self._device)

    actions = torch.LongTensor([t.action for t in transitions])
    rewards = torch.Tensor([t.reward for t in transitions])
    iters = torch.LongTensor([t.iteration for t in transitions])
    upper_bounds = torch.Tensor(np.repeat([t.upper_bound for t in transitions], self._num_actions)).reshape(-1, self._num_actions)

    are_final_steps = torch.Tensor([t.is_final_step for t in transitions])
    legal_actions_mask = torch.Tensor(np.array([t.legal_actions_mask for t in transitions]))

    self._q_values = self._q_network(info_states).cpu()
    if self._double_dqn:
      next_q_values = self._q_network(next_info_states).cpu().detach()
    
    self._target_q_values = self._target_q_network(next_info_states).cpu().detach()

    # Clamp targets - no point in learning values higher than we know they really are
    if self.lower_bound_utility is not None and self.upper_bound_utility is not None:
      lbs = torch.tensor([self.lower_bound_utility] * len(self._target_q_values) * self._num_actions).reshape(-1, self._num_actions)
      self._target_q_values = torch.clamp(self._target_q_values, min=lbs, max=upper_bounds)
      if self._double_dqn:
        next_q_values = torch.clamp(next_q_values, min=lbs, max=upper_bounds)

    illegal_actions = 1 - legal_actions_mask
    illegal_logits = illegal_actions * ILLEGAL_ACTION_LOGITS_PENALTY  
    if self._double_dqn:
      # pick the action that has the highest Q value according to the normal Q network, and
      max_indices = torch.argmax(next_q_values + illegal_logits, dim=1)
      # grab the Q value of that action from the target Q network
      max_next_q = torch.gather(self._target_q_values, 1, max_indices.view(-1, 1)).squeeze()
    else:
      max_next_q = torch.max(self._target_q_values + illegal_logits, dim=1)[0]
    target = (rewards + (1 - are_final_steps) * self._discount_factor * max_next_q)
    action_indices = torch.stack([
        torch.arange(self._q_values.shape[0], dtype=torch.long), actions
    ], dim=0)
    predictions = self._q_values[list(action_indices)]

    loss = self.loss_class(predictions, target)

    self._optimizer.zero_grad()
    loss.backward()
    self._optimizer.step()

    duration = time.time() - start
    self._train_time += duration

    return loss.detach()

  def _clear_cache(self):
    self._cache.clear()

  @property
  def q_values(self):
    return self._q_values

  @property
  def replay_buffer(self):
    return self._replay_buffer

  @property
  def loss(self):
    return self._last_loss_value

  @property
  def prev_timestep(self):
    return self._prev_timestep

  @property
  def prev_action(self):
    return self._prev_action

  @property
  def prev_action_greedy(self):
    return self._prev_action_greedy

  @property
  def step_counter(self):
    return self._step_counter

  def get_weights(self):
    variables = [m.weight for m in self._q_network.model]
    variables.append([m.weight for m in self._target_q_network.model])
    return variables

  def copy_with_noise(self, sigma=0.0, copy_weights=True):
    """Copies the object and perturbates it with noise.

    Args:
      sigma: gaussian dropout variance term : Multiplicative noise following
        (1+sigma*epsilon), epsilon standard gaussian variable, multiplies each
        model weight. sigma=0 means no perturbation.
      copy_weights: Boolean determining whether to copy model weights (True) or
        just model hyperparameters.

    Returns:
      Perturbated copy of the model.
    """
    _ = self._kwargs.pop("self", None)
    copied_object = DQN(**self._kwargs)

    q_network = getattr(copied_object, "_q_network")
    target_q_network = getattr(copied_object, "_target_q_network")

    if copy_weights:
      with torch.no_grad():
        for q_model in q_network.model:
          q_model.weight *= (1 + sigma * torch.randn(q_model.weight.shape))
        for tq_model in target_q_network.model:
          tq_model.weight *= (1 + sigma * torch.randn(tq_model.weight.shape))
    return copied_object

  def clear_buffer(self):
    self._replay_buffer.clear()