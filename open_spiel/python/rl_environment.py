# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reinforcement Learning (RL) Environment for Open Spiel.

This module wraps Open Spiel Python interface providing an RL-friendly API. It
covers both turn-based and simultaneous move games. Interactions between agents
and the underlying game occur mostly through the `reset` and `step` methods,
which return a `TimeStep` structure (see its docstrings for more info).

The following example illustrates the interaction dynamics. Consider a 2-player
Kuhn Poker (turn-based game). Agents have access to the `observations` (a dict)
field from `TimeSpec`, containing the following members:
 * `info_state`: list containing the game information state for each player. The
   size of the list always correspond to the number of players. E.g.:
   [[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]].
 * `legal_actions`: list containing legal action ID lists (one for each player).
   E.g.: [[0, 1], [0]], which corresponds to actions 0 and 1 being valid for
   player 0 (the 1st player) and action 0 being valid for player 1 (2nd player).
 * `current_player`: zero-based integer representing the player to make a move.

At each `step` call, the environment expects a singleton list with the action
(as it's a turn-based game), e.g.: [1]. This (zero-based) action must correspond
to the player specified at `current_player`. The game (which is at decision
node) will process the action and take as many steps necessary to cover chance
nodes, halting at a new decision or final node. Finally, a new `TimeStep`is
returned to the agent.

Simultaneous-move games follow analogous dynamics. The only differences is the
environment expects a list of actions, one per player. Note the `current_player`
field is "irrelevant" here, admitting a constant value defined in spiel.h, which
defaults to -2 (module level constant `SIMULTANEOUS_PLAYER_ID`).

See open_spiel/python/examples/rl_example.py for example usages.
"""

import collections

import enum
from os import stat
from absl import logging
import numpy as np

import pyspiel

SIMULTANEOUS_PLAYER_ID = pyspiel.PlayerId.SIMULTANEOUS


class TimeStep(
    collections.namedtuple(
        "TimeStep", ["observations", "rewards", "discounts", "step_type"])):
  """Returned with every call to `step` and `reset`.

  A `TimeStep` contains the data emitted by a game at each step of interaction.
  A `TimeStep` holds an `observation` (list of dicts, one per player),
  associated lists of `rewards`, `discounts` and a `step_type`.

  The first `TimeStep` in a sequence will have `StepType.FIRST`. The final
  `TimeStep` will have `StepType.LAST`. All other `TimeStep`s in a sequence will
  have `StepType.MID.

  Attributes:
    observations: a list of dicts containing observations per player.
    rewards: A list of scalars (one per player), or `None` if `step_type` is
      `StepType.FIRST`, i.e. at the start of a sequence.
    discounts: A list of discount values in the range `[0, 1]` (one per player),
      or `None` if `step_type` is `StepType.FIRST`.
    step_type: A `StepType` enum value.
  """
  __slots__ = ()

  def first(self):
    return self.step_type == StepType.FIRST

  def mid(self):
    return self.step_type == StepType.MID

  def last(self):
    return self.step_type == StepType.LAST

  def is_simultaneous_move(self):
    return self.observations["current_player"] == SIMULTANEOUS_PLAYER_ID

  def current_player(self):
    return self.observations["current_player"]


class StepType(enum.Enum):
  """Defines the status of a `TimeStep` within a sequence."""

  FIRST = 0  # Denotes the first `TimeStep` in a sequence.
  MID = 1  # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
  LAST = 2  # Denotes the last `TimeStep` in a sequence.

  def first(self):
    return self is StepType.FIRST

  def mid(self):
    return self is StepType.MID

  def last(self):
    return self is StepType.LAST


# Global pyspiel members
def registered_games():
  return pyspiel.registered_games()


class ChanceEventSampler(object):
  """Default sampler for external chance events."""

  def __init__(self, seed=None):
    self.seed(seed)

  def seed(self, seed=None):
    self._rng = np.random.RandomState(seed)

  def __call__(self, state):
    """Sample a chance event in the given state."""
    actions, probs = zip(*state.chance_outcomes())
    return self._rng.choice(actions, p=probs)


class ObservationType(enum.Enum):
  """Defines what kind of observation to use."""
  OBSERVATION = 0  # Use observation_tensor
  INFORMATION_STATE = 1  # Use information_state_tensor


class Environment(object):
  """Open Spiel reinforcement learning environment class."""

  def __init__(self,
               game,
               discount=1.0,
               chance_event_sampler=None,
               observation_type=None,
               include_full_state=False,
               mfg_distribution=None,
               mfg_population=None,
               enable_legality_check=False,
               all_simultaneous=False,
               terminal_rewards=False,
               **kwargs):
    """Constructor.

    Args:
      game: [string, pyspiel.Game] Open Spiel game name or game instance.
      discount: float, discount used in non-initial steps. Defaults to 1.0.
      chance_event_sampler: optional object with `sample_external_events` method
        to sample chance events.
      observation_type: what kind of observation to use. If not specified, will
        default to INFORMATION_STATE unless the game doesn't provide it.
      include_full_state: whether or not to include the full serialized
        OpenSpiel state in the observations (sometimes useful for debugging).
      mfg_distribution: the distribution over states if the game is a mean field
        game.
      mfg_population: The Mean Field Game population to consider.
      enable_legality_check: Check the legality of the move before stepping.
      **kwargs: dict, additional settings passed to the Open Spiel game.
    """
    self._chance_event_sampler = chance_event_sampler or ChanceEventSampler()
    self._include_full_state = include_full_state
    self._mfg_distribution = mfg_distribution
    self._mfg_population = mfg_population
    self._enable_legality_check = enable_legality_check
    self._all_simultaneous = all_simultaneous
    self._terminal_rewards = terminal_rewards


    if isinstance(game, str):
      if kwargs:
        game_settings = {key: val for (key, val) in kwargs.items()}
        logging.info("Using game settings: %s", game_settings)
        self._game = pyspiel.load_game(game, game_settings)
      else:
        logging.info("Using game string: %s", game)
        self._game = pyspiel.load_game(game)
    else:  # pyspiel.Game or API-compatible object.
      logging.info("Using game instance: %s", game.get_type().short_name)
      self._game = game

    self._num_players = self._game.num_players()
    self._is_turn_based = ((self._game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL) or (self._game.get_type().dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD))

    self._state = None
    self._should_reset = True

    # Discount returned at non-initial steps.
    self._discounts = [discount] * self._num_players

    # Determine what observation type to use.
    if observation_type is None:
      if self._game.get_type().provides_information_state_tensor:
        observation_type = ObservationType.INFORMATION_STATE
      else:
        observation_type = ObservationType.OBSERVATION

    # Check the requested observation type is supported.
    if observation_type == ObservationType.OBSERVATION:
      if not self._game.get_type().provides_observation_tensor:
        raise ValueError(f"observation_tensor not supported by {game}")
    elif observation_type == ObservationType.INFORMATION_STATE:
      if not self._game.get_type().provides_information_state_tensor:
        raise ValueError(f"information_state_tensor not supported by {game}")
    self._use_observation = (observation_type == ObservationType.OBSERVATION)

    if self._game.get_type().dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD:
      assert mfg_distribution is not None
      assert mfg_population is not None
      assert 0 <= mfg_population < self._num_players

  def seed(self, seed=None):
    self._chance_event_sampler.seed(seed)

  def get_time_step(self):
    """Returns a `TimeStep` without updating the environment.

    Returns:
      A `TimeStep` namedtuple containing:
        observation: list of dicts containing one observations per player, each
          corresponding to `observation_spec()`.
        reward: list of rewards at this timestep, or None if step_type is
          `StepType.FIRST`.
        discount: list of discounts in the range [0, 1], or None if step_type is
          `StepType.FIRST`.
        step_type: A `StepType` value.
    """
    current_player = self._state.current_player()
    observations = {
        "info_state": [],
        "legal_actions": [],
        "current_player": current_player,
        "serialized_state": []
    }
    rewards = []
    step_type = StepType.LAST if self._state.is_terminal() else StepType.MID
    self._should_reset = step_type == StepType.LAST

    if self._all_simultaneous and self._terminal_rewards and step_type != StepType.LAST:
      for player_id in range(self.num_players):
        rewards.append(0) # Rewards are terminal - let's not bother
        if player_id == current_player:
          info_state = self._state.observation_tensor(player_id) if self._use_observation else self._state.information_state_tensor(player_id)
          observations['info_state'].append(info_state)
          observations["legal_actions"].append(self._state.legal_actions(player_id))
        else:
          observations['info_state'].append(None) # Let's not bother computing this one - we won't ever look at it
          observations["legal_actions"].append(None)
    else:
      cur_rewards = self._state.rewards()
      for player_id in range(self.num_players):
        rewards.append(cur_rewards[player_id])
        observations["info_state"].append(self._state.observation_tensor(player_id) if self._use_observation else self._state.information_state_tensor(player_id))
        observations["legal_actions"].append(self._state.legal_actions(player_id))
    discounts = self._discounts
    if step_type == StepType.LAST:
      # When the game is in a terminal state set the discount to 0.
      discounts = [0. for _ in discounts]

    if self._include_full_state:
      observations["serialized_state"] = pyspiel.serialize_game_and_state(self._game, self._state)

    return TimeStep(
        observations=observations,
        rewards=rewards,
        discounts=discounts,
        step_type=step_type)

  def _check_legality(self, actions):
    if self.is_turn_based:
      legal_actions = self._state.legal_actions()
      if actions[0] not in legal_actions:
        raise RuntimeError(f"step() called on illegal action {actions[0]}")
    else:
      for p in range(len(actions)):
        legal_actions = self._state.legal_actions(p)
        if legal_actions and actions[p] not in legal_actions:
          raise RuntimeError(f"step() by player {p} called on illegal " +
                             f"action: {actions[p]}")

  def step(self, actions):
    """Updates the environment according to `actions` and returns a `TimeStep`.

    If the environment returned a `TimeStep` with `StepType.LAST` at the
    previous step, this call to `step` will start a new sequence and `actions`
    will be ignored.

    This method will also start a new sequence if called after the environment
    has been constructed and `reset` has not been called. Again, in this case
    `actions` will be ignored.

    Args:
      actions: a list containing one action per player, following specifications
        defined in `action_spec()`.

    Returns:
      A `TimeStep` namedtuple containing:
        observation: list of dicts containing one observations per player, each
          corresponding to `observation_spec()`.
        reward: list of rewards at this timestep, or None if step_type is
          `StepType.FIRST`.
        discount: list of discounts in the range [0, 1], or None if step_type is
          `StepType.FIRST`.
        step_type: A `StepType` value.
    """
    assert len(actions) == self.num_actions_per_step, ("Invalid number of actions! Expected {}".format(self.num_actions_per_step))
    if self._should_reset:
      return self.reset()

    if self._enable_legality_check:
      self._check_legality(actions)

    if self.is_turn_based:
      self._state.apply_action(actions[0])
    else:
      self._state.apply_actions(actions)
    self._sample_external_events()

    return self.get_time_step()

  def reset(self, epsilon=None):
    """Starts a new sequence and returns the first `TimeStep` of this sequence.

    Returns:
      A `TimeStep` namedtuple containing:
        observations: list of dicts containing one observations per player, each
          corresponding to `observation_spec()`.
        rewards: list of rewards at this timestep, or None if step_type is
          `StepType.FIRST`.
        discounts: list of discounts in the range [0, 1], or None if step_type
          is `StepType.FIRST`.
        step_type: A `StepType` value.
    """
    self._should_reset = False
    # if self._game.get_type().dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD and self._num_players > 1:
    #   self._state = self._game.new_initial_state_for_population(self._mfg_population)
    # else:
    if epsilon is not None and np.random.binomial(1, epsilon):
      self._state = self._random_state()
    else:
      self._state = self._game.new_initial_state()

    self._sample_external_events(reset=True)

    observations = {
        "info_state": [],
        "legal_actions": [],
        "current_player": [],
        "serialized_state": []
    }
    for player_id in range(self.num_players):
      observations["info_state"].append(self._state.observation_tensor(player_id) if self._use_observation else self._state.information_state_tensor(player_id))
      observations["legal_actions"].append(self._state.legal_actions(player_id))
    observations["current_player"] = self._state.current_player()

    if self._include_full_state:
      observations["serialized_state"] = pyspiel.serialize_game_and_state(self._game, self._state)

    return TimeStep(
        observations=observations,
        rewards=None,
        discounts=None,
        step_type=StepType.FIRST)

  def _sample_external_events(self, reset=False):
    """Sample chance events until we get to a decision node."""
    # Modified function for speedup since we aren't interested in mean field games
    while self._state.is_chance_node(): 
      outcome = self._chance_event_sampler(self._state, reset=reset)
      self._state.apply_action(outcome)

  def observation_spec(self):
    """Defines the observation per player provided by the environment.

    Each dict member will contain its expected structure and shape. E.g.: for
    Kuhn Poker {"info_state": (6,), "legal_actions": (2,), "current_player": (),
                "serialized_state": ()}

    Returns:
      A specification dict describing the observation fields and shapes.
    """
    return dict(
        info_state=tuple([
            self._game.observation_tensor_size() if self._use_observation else
            self._game.information_state_tensor_size()
        ]),
        legal_actions=(self._game.num_distinct_actions(),),
        current_player=(),
        serialized_state=(),
    )

  def action_spec(self):
    """Defines per player action specifications.

    Specifications include action boundaries and their data type.
    E.g.: for Kuhn Poker {"num_actions": 2, "min": 0, "max":1, "dtype": int}

    Returns:
      A specification dict containing per player action properties.
    """
    return dict(
        num_actions=self._game.num_distinct_actions(),
        min=0,
        max=self._game.num_distinct_actions() - 1,
        dtype=int,
    )

  # Environment properties
  @property
  def use_observation(self):
    """Returns whether the environment is using the game's observation.

    If false, it is using the game's information state.
    """
    return self._use_observation

  # Game properties
  @property
  def name(self):
    return self._game.get_type().short_name

  @property
  def num_players(self):
    return self._game.num_players()

  @property
  def num_actions_per_step(self):
    return 1 if self.is_turn_based else self.num_players

  # New RL calls for more advanced use cases (e.g. search + RL).
  @property
  def is_turn_based(self):
    return self._is_turn_based

  @property
  def max_game_length(self):
    return self._game.max_game_length()

  @property
  def is_chance_node(self):
    return self._state.is_chance_node()

  @property
  def game(self):
    return self._game

  def set_state(self, new_state):
    """Updates the game state."""
    assert new_state.get_game() == self.game, ("State must have been created by the same game.")
    self._state = new_state

  @property
  def get_state(self):
    return self._state

  @property
  def mfg_distribution(self):
    return self._mfg_distribution

  def update_mfg_distribution(self, mfg_distribution):
    """Updates the distribution over the states of the mean field game."""
    assert (
        self._game.get_type().dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD)
    self._mfg_distribution = mfg_distribution
    
  def _random_state(self):
    # TODO: You might do better accumulating states in a big buffer first, rather than doing it like this, both in perforamcne and in ammortizing the "warming up" of the chain

    # TODO: This would be better using fast choice
    n_chance = self._game.max_chance_outcomes()
    n_actions = self._game.num_distinct_actions()
    k_max = max(n_chance, n_actions) + 1 # +1 for parent

    n_steps = 0
    N_STEPS_MAX = 10_000 # TODO: Should be a parameter
    state = self._game.new_initial_state()
    action_hist = []
    state_hist = []

    randomness = np.random.rand(N_STEPS_MAX * 10) # TODO: Better choice here, or just cycle through this
    rand_index = 0

    def flip_coin(p, rand_index):
        return randomness[rand_index] < p, rand_index + 1

    failure_count = 0
    while n_steps < N_STEPS_MAX and not state.is_chance_node():
        stay = True
        node_degree = 1 # Parent
        if state.is_chance_node():
            legal_actions = state.chance_outcomes()
        else:
            legal_actions = state.legal_actions()
        node_degree += len(legal_actions)

        # Should I stay or should I go?
        # Stay at node 𝑛 with probability 1−𝑘𝑛/𝑘max
        stay_prob = 1 - (node_degree / k_max)
        while stay:
            stay, rand_index = flip_coin(stay_prob, rand_index)
            if stay:
                n_steps += 1 
                continue
        else:
            if state.is_initial_state():
                transition_to_parent = False
            else:
                transition_to_parent, rand_index = flip_coin(1 / node_degree, rand_index)
            if transition_to_parent:
                state = state_hist.pop()
                action = action_hist.pop()
                n_steps += 1
            else:
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    action_list, prob_list = zip(*outcomes)
                    action = np.random.choice(action_list, p=prob_list)
                else:
                    action = np.random.choice(state.legal_actions(state.current_player())) # TODO: You surely want to use fast_choice but I can't depend on ubc_utils here
                # Only go through with this if it's not terminal
                clone = state.clone()
                clone.apply_action(action)
                if not clone.is_terminal(): # TODO: Could be smarter about resmampling here by taking away the same choices
                    state_hist.append(state.clone())
                    state = clone
                    action_hist.append(action)
                else:
                    failure_count += 1
    return state
