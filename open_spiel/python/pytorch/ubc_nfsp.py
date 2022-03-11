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

"""Neural Fictitious Self-Play (NFSP) agent implemented in PyTorch.

See the paper https://arxiv.org/abs/1603.01121 for more details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import enum
import os
from typing import Optional
from absl import logging
import numpy as np
import time

import torch
import torch.nn.functional as F

from open_spiel.python import rl_agent
from open_spiel.python.pytorch import ubc_dqn
from open_spiel.python.examples.ubc_utils import single_action_result, fast_choice
from cachetools import cached, LRUCache, TTLCache
from cachetools.keys import hashkey


Transition = collections.namedtuple(
    "Transition", "info_state action_probs legal_actions_mask iteration")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9

def brier_score(
    input: torch.Tensor,
    target: torch.Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
):
  # input: (examples, actions)
  # target: (examples)
  # make target_onehot: (examples, action)
  num_examples, num_actions = input.shape
  target_onehot = F.one_hot(target, num_classes = num_actions).to(torch.float)

  # call MSE
  return F.mse_loss(
    input,
    target_onehot,
    size_average,
    reduce,
    reduction,
  )


class NFSP(rl_agent.AbstractAgent):
  """NFSP Agent implementation in PyTorch.

  See open_spiel/python/examples/kuhn_nfsp.py for an usage example.
  """

  def __init__(self,
               player_id,
               num_actions,
               num_players,
               sl_model,
               sl_model_args,
               rl_model,
               rl_model_args,
               reservoir_buffer_capacity,
               anticipatory_param,
               batch_size=128,
               rl_learning_rate=0.01,
               sl_learning_rate=0.01,
               min_buffer_size_to_learn=1000,
               learn_every=64,
               sl_loss_str="cross_entropy",
               optimizer_str="sgd",
               add_explore_transitions=True, # Should I add transitions to the resevoir buffer if they are caused due to exploration?
               device='cpu',
               **kwargs):
    """Initialize the `NFSP` agent."""
    self.player_id = player_id
    self._num_actions = num_actions
    self._batch_size = batch_size
    self._learn_every = learn_every
    self._anticipatory_param = anticipatory_param
    self._min_buffer_size_to_learn = min_buffer_size_to_learn
    self._add_explore_transitions = add_explore_transitions

    self._reservoir_buffer = ReservoirBuffer(reservoir_buffer_capacity)
    self._prev_timestep = None
    self._prev_action = None

    # Step counter to keep track of learning.
    self._step_counter = 0
    self._iteration = 0
    self._global_iteration = 0
    self._last_learn_iteration = -1
    self._cache = LRUCache(maxsize=5000)

    self._device = device
    logging.info(f"Creating NFSP using device: {device} for player {player_id}")

    # Inner RL agent
    kwargs.update({
        "batch_size": batch_size,
        "learning_rate": rl_learning_rate,
        "learn_every": learn_every,
        "min_buffer_size_to_learn": min_buffer_size_to_learn,
        "optimizer_str": optimizer_str,
        "device": device,
    })
    self._rl_agent = ubc_dqn.DQN(
      player_id,
      num_actions, 
      num_players,
      q_network_model=rl_model,
      q_network_args=rl_model_args,
      **kwargs
    )

    # Keep track of the last training loss achieved in an update step.
    self._last_sl_loss_value = None

    # Average policy network.
    self._avg_network = sl_model(**sl_model_args).to(self._device)

    self._savers = [
        ("q_network", self._rl_agent._q_network),
        ("avg_network", self._avg_network)
    ]

    if sl_loss_str == "cross_entropy":
      self.loss_class = F.cross_entropy
    elif sl_loss_str == "brier_score":
      self.loss_class = brier_score
    else:
      raise ValueError("Not implemented, choose from 'cross_entropy', 'brier'.")

    if optimizer_str == "adam":
      self.optimizer = torch.optim.Adam(self._avg_network.parameters(), lr=sl_learning_rate)
    elif optimizer_str == "sgd":
      self.optimizer = torch.optim.SGD(self._avg_network.parameters(), lr=sl_learning_rate)
    else:
      raise ValueError("Not implemented. Choose from ['adam', 'sgd'].")

    self._sample_episode_policy()

  @contextlib.contextmanager
  def temp_mode_as(self, best_response_mode):
    """Context manager to temporarily overwrite the mode."""
    previous_mode = self._best_response_mode
    self._best_response_mode = best_response_mode
    yield
    self._best_response_mode = previous_mode

  def _sample_episode_policy(self):
    if np.random.rand() < self._anticipatory_param:
      self._best_response_mode = True
    else:
      self._best_response_mode = False

  def _act(self, info_state, legal_actions):
    if len(legal_actions) == 1: # Let's not run the NN if you are faced with a single action (imagine a case where one player drops out and remaining players duel onwards)
      return single_action_result(legal_actions, self._num_actions)
    else:
      key = hashkey(tuple(info_state))
      val = self._cache.get(key)
      if val is not None:
        probs = np.array(val)
      else:
        probs = np.zeros(self._num_actions)
        info_state_reshaped = self._avg_network.reshape_infostate(info_state)
        info_state_batch = self._avg_network.prep_batch([info_state_reshaped]).to(self._device)
        with torch.no_grad():
          action_values = self._avg_network(info_state_batch).cpu()
        self._last_action_values = action_values[0]
        legal_values = action_values[0][legal_actions]
        probs[legal_actions] = F.softmax(legal_values, dim=0).detach().numpy()
        probs /= sum(probs)
        self._cache[key] = probs
      action = fast_choice(range(len(probs)), probs)
    return action, probs

  @property
  def loss(self):
    return (self._last_sl_loss_value, self._last_rl_loss_value().detach())

  def _last_rl_loss_value(self):
    return self._rl_agent.loss

  def step(self, time_step, is_evaluation=False):
    """Returns the action to be taken and updates the Q-networks if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    # I don't want to be in BR mode during evals
    if is_evaluation:
      self._best_response_mode = False

    if self._best_response_mode:
      agent_output = self._rl_agent.step(time_step, is_evaluation)
      if not is_evaluation and not time_step.last():
        if self._rl_agent.prev_action_greedy or self._add_explore_transitions:
          self._add_transition(time_step, agent_output)
    else:
      # Act step: don't act at terminal info states.
      if not time_step.last():
        info_state = time_step.observations["info_state"][self.player_id]
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        action, probs = self._act(info_state, legal_actions)
        agent_output = rl_agent.StepOutput(action=action, probs=probs)

      if self._prev_timestep and not is_evaluation:
        self._rl_agent.add_transition(self._prev_timestep, self._prev_action, time_step)

    if not is_evaluation:
      self._step_counter += 1

      if self._iteration % self._learn_every == 0 and self._last_learn_iteration < self._iteration:
        self._last_sl_loss_value = self._learn()
        # If learn step not triggered by rl policy, learn.
        if not self._best_response_mode:
          self._rl_agent.learn()

      # Prepare for the next episode.
      if time_step.last():
        self._sample_episode_policy()
        self._prev_timestep = None
        self._prev_action = None
        self._iteration += 1
        return
      else:
        self._prev_timestep = time_step
        self._prev_action = agent_output.action
    else:
      if time_step.last():
        return rl_agent.StepOutput(action=None, probs=[])

    return agent_output

  def _add_transition(self, time_step, agent_output):
    """Adds the new transition using `time_step` to the reservoir buffer.

    Transitions are in the form (time_step, agent_output.probs, legal_mask).

    Args:
      time_step: an instance of rl_environment.TimeStep.
      agent_output: an instance of rl_agent.StepOutput.
    """
    legal_actions = time_step.observations["legal_actions"][self.player_id]
    legal_actions_mask = np.zeros(self._num_actions)
    legal_actions_mask[legal_actions] = 1.0

    info_state_flat = time_step.observations["info_state"][self.player_id][:]
    info_state = self._avg_network.reshape_infostate(info_state_flat).to(self._device)

    transition = Transition(
        info_state=info_state,
        action_probs=agent_output.probs,
        legal_actions_mask=legal_actions_mask,
        iteration=self._iteration
        )
    self._reservoir_buffer.add(transition)

  def _learn(self):
    """Compute the loss on sampled transitions and perform a avg-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """
    self._last_learn_iteration = self._iteration
    self._clear_cache()
    if (len(self._reservoir_buffer) < self._batch_size or len(self._reservoir_buffer) < self._min_buffer_size_to_learn):
      return None

    transitions = self._reservoir_buffer.sample(self._batch_size)

    info_state_list = [t.info_state for t in transitions]
    info_states = self._avg_network.prep_batch(info_state_list).to(self._device)

    action_probs = torch.Tensor(np.array([t.action_probs for t in transitions]))

    self.optimizer.zero_grad()
    loss = self.loss_class(self._avg_network(info_states).cpu(),
                           torch.max(action_probs, dim=1)[1])
    loss.backward()
    self.optimizer.step()

    loss_value = loss.detach()
    if np.isnan(loss_value):
      raise ValueError("NaN loss - is your SL learning rate too high?")

    return loss_value

  def _clear_cache(self):
    self._cache.clear()

  def _full_checkpoint_name(self, checkpoint_dir, name):
    checkpoint_filename = "_".join([name, "pid" + str(self.player_id)])
    return os.path.join(checkpoint_dir, checkpoint_filename)

  def _latest_checkpoint_filename(self, name):
    checkpoint_filename = "_".join([name, "pid" + str(self.player_id)])
    return checkpoint_filename + "_latest"

  def set_global_iteration(self, global_iteration):
    self._global_iteration = global_iteration
    self._rl_agent.set_global_iteration(global_iteration)

  def save(self):
    """Saves the average policy network and the inner RL agent's q-network.

    Note that this does not save the experience replay buffers and should
    only be used to restore the agent's policy, not resume training.

    Args:
      checkpoint_dir: directory where checkpoints will be saved.
    """
    restore_dict = dict()
    for name, model in self._savers:
      restore_dict[name] = model.state_dict()
    return restore_dict

  def has_checkpoint(self, checkpoint_dir):
    for name, _ in self._savers:
      path = self._full_checkpoint_name(checkpoint_dir, name)
      if os.path.exists(path):
        return True
    return False

  def restore(self, restore_dict):
    """Restores the average policy network and the inner RL agent's q-network.

    Note that this does not restore the experience replay buffers and should
    only be used to restore the agent's policy, not resume training.

    Args:
      checkpoint_dir: directory from which checkpoints will be restored.
    """
    for name, model in self._savers:
      model.load_state_dict(restore_dict[name])

  def clear_buffer(self):
    self._reservoir_buffer.clear()
    self._rl_agent.clear_buffer()

class ReservoirBuffer(object):
  """Allows uniform sampling over a stream of data.

  This class supports the storage of arbitrary elements, such as observation
  tensors, integer actions, etc.

  See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
  """

  def __init__(self, reservoir_buffer_capacity):
    self._reservoir_buffer_capacity = reservoir_buffer_capacity
    self._data = []
    self._add_calls = 0

  def add(self, element):
    """Potentially adds `element` to the reservoir buffer.

    Args:
      element: data to be added to the reservoir buffer.
    """
    if len(self._data) < self._reservoir_buffer_capacity:
      self._data.append(element)
    else:
      idx = np.random.randint(0, self._add_calls + 1)
      if idx < self._reservoir_buffer_capacity:
        self._data[idx] = element
    self._add_calls += 1

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
      raise ValueError("{} elements could not be sampled from size {}".format(num_samples, len(self._data)))

    # return random.sample(self._data, num_samples)
    # Modified to use np sampling and not python random
    sampled_indices = np.random.choice(len(self._data), num_samples, replace=False)
    samples = []
    for s in sampled_indices:
      samples.append(self._data[s])
    return samples

  def clear(self):
    self._data = []
    self._add_calls = 0

  def __len__(self):
    return len(self._data)

  def __iter__(self):
    return iter(self._data)
