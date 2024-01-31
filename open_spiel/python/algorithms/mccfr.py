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

"""Python base module for the implementations of Monte Carlo Counterfactual Regret Minimization."""

import numpy as np
from open_spiel.python import policy
from cachetools import LRUCache

INFOSET_CACHE_SIZE = 500_000

REGRET_INDEX = 0
AVG_POLICY_INDEX = 1
VISIT_COUNT_INDEX = 2
AVG_REWARD_INDEX = 3

class AveragePolicy(policy.Policy):
  """A policy object representing the average policy for MCCFR algorithms."""

  def __init__(self, game, player_ids, infostates):
    # Do not create a copy of the dictionary
    # but work on the same object

    # NOTE (Neil): Adding the dict b/c I don't want to mess with the LRU cache when accessing things in the policy

    super().__init__(game, player_ids)
    self._infostates = dict(infostates)

  def action_probabilities(self, state, player_id=None):
    """Returns the MCCFR average policy for a player in a state.

    If the policy is not defined for the provided state, a uniform
    random policy is returned.

    Args:
      state: A `pyspiel.State` object.
      player_id: Optional, the player id for which we want an action. Optional
        unless this is a simultaneous state at which multiple players can act.

    Returns:
      A `dict` of `{action: probability}` for the specified player in the
      supplied state. If the policy is defined for the state, this
      will contain the average MCCFR strategy defined for that state.
      Otherwise, it will contain all legal actions, each with the same
      probability, equal to 1 / num_legal_actions.
    """
    if player_id is None:
      player_id = state.current_player()
    legal_actions = state.legal_actions()
    info_state_key = state.information_state_string(player_id)
    retrieved_infostate = self._infostates.get(info_state_key, None)
    if retrieved_infostate is None:
      return {a: 1 / len(legal_actions) for a in legal_actions}
    avstrat = (
        retrieved_infostate[AVG_POLICY_INDEX] /
        retrieved_infostate[AVG_POLICY_INDEX].sum())
    return {legal_actions[i]: avstrat[i] for i in range(len(legal_actions))}


class MCCFRSolverBase(object):
  """A base class for both outcome MCCFR and external MCCFR."""

  def __init__(self, game, regret_matching_plus=False, linear_averaging=False, regret_init='uniform', regret_init_strength = 1., avg_reward_decay=1.0):
    self._game = game
    self._infostates = LRUCache(INFOSET_CACHE_SIZE)  # infostate keys -> [regrets, avg strat, visit count]
    self._num_players = game.num_players()
    self.regret_matching_plus = regret_matching_plus
    if self.regret_matching_plus:
      self.touched = set()
    self.linear_averaging = linear_averaging
    self._iteration = 0 # For linear averaging
    self.regret_init = regret_init
    self.regret_init_strength = regret_init_strength
    self.avg_reward_decay = avg_reward_decay

  def _lookup_infostate_info(self, info_state_key, num_legal_actions, state):
    """Looks up an information set table for the given key.

    Args:
      info_state_key: information state key (string identifier).
      num_legal_actions: number of legal actions at this information state.

    Returns:
      A list of:
        - the average regrets as a numpy array of shape [num_legal_actions]
        - the average strategy as a numpy array of shape
        [num_legal_actions].
          The average is weighted using `my_reach`
    """
    retrieved_infostate = self._infostates.get(info_state_key, None)
    if retrieved_infostate is not None:
      return retrieved_infostate

    # Start with a small amount of regret and total accumulation, to give a
    # uniform policy: this will get erased fast.

    initial_regrets = np.ones(num_legal_actions, dtype=np.float64) / 1e6
    if self.regret_init != 'uniform': # TODO: If you run on non-clock auction, hasattr
      initial_regrets += state.regret_init(self.regret_init) * self.regret_init_strength

    self._infostates[info_state_key] = [
        initial_regrets, # regret list
        np.ones(num_legal_actions, dtype=np.float64) / 1e6, # avg policy
        0, # visit count
        0, # avg reward
    ]
    return self._infostates[info_state_key]

  def _add_regret(self, info_state_key, action_idx, amount):
    self._infostates[info_state_key][REGRET_INDEX][action_idx] += amount if not self.linear_averaging else amount * self._iteration

  def _add_avstrat(self, info_state_key, action_idx, amount):
    self._infostates[info_state_key][AVG_POLICY_INDEX][action_idx] += amount

  def _add_visit(self, info_state_key):
    self._infostates[info_state_key][VISIT_COUNT_INDEX] += 1

  def _add_reward(self, info_state_key, reward):
    self._infostates[info_state_key][AVG_REWARD_INDEX] = self.avg_reward_decay * (self._infostates[info_state_key][AVG_REWARD_INDEX]) + (1 - self.avg_reward_decay) * reward

  def average_policy(self):
    """Computes the average policy, containing the policy for all players.

    Returns:
      An average policy instance that should only be used during
      the lifetime of solver object.
    """
    return AveragePolicy(self._game, list(range(self._num_players)),
                         self._infostates)

  def _regret_matching(self, regrets, num_legal_actions):
    """Applies regret matching to get a policy.

    Args:
      regrets: numpy array of regrets for each action.
      num_legal_actions: number of legal actions at this state.

    Returns:
      numpy array of the policy indexed by the index of legal action in the
      list.
    """
    positive_regrets = np.maximum(regrets,
                                  np.zeros(num_legal_actions, dtype=np.float64))
    sum_pos_regret = positive_regrets.sum()
    if sum_pos_regret <= 0:
      return np.ones(num_legal_actions, dtype=np.float64) / num_legal_actions
    else:
      return positive_regrets / sum_pos_regret

  def get_solver_stats(self):
    # TODO: other stats
    return {'num_infostates': len(self._infostates)}
