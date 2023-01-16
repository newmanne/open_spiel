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

# Lint as python3
import enum
 
import numpy as np
import pandas as pd
import scipy.stats
import sys
import pyspiel
import yaml
import collections
from functools import lru_cache
import pickle
from tqdm import tqdm
import os

def pad_array(A, size):
    t = size - len(A)
    return A + [0] * t

_DEFAULT_PARAMS = {
    "config_file": 'config.yml',
}

class Action(enum.IntEnum):
  EXIT = 0
  BID = 1


CONFIG_ROOT = os.environ.get("SEQUENTIAL_AUCTION_CONFIG_DIR", "/TODO")
def config_path_from_config_name(config_name):
    return f'{CONFIG_ROOT}/{config_name}.yml'

_NUM_PLAYERS = 10
_GAME_TYPE = pyspiel.GameType(
    short_name="seqauc",
    long_name="Sequential Auctions",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.REWARDS,
    max_num_players=_NUM_PLAYERS,
    min_num_players=1,
    parameter_specification=_DEFAULT_PARAMS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=False,
    provides_observation_tensor=False,
    provides_factored_observation_string=False)

class SequentialAuctionGame(pyspiel.Game):
  """A Python version of Sequential Auctions"""

  def __init__(self, params=None):
    with open(config_path_from_config_name(params['config_file']), 'r') as f:
      self.config = yaml.safe_load(f)
    
    game_info = pyspiel.GameInfo(
        num_distinct_actions=self.num_distinct_actions(),
        max_chance_outcomes=99999,
        min_utility=-1e9,
        num_players=_NUM_PLAYERS,
        max_utility=1e9,
        max_game_length=100
    )


    super().__init__(_GAME_TYPE, game_info, params if params else {})

  def num_distinct_actions(self):
    # You can remain in the auction, or you can leave
    return 2

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return SequentialAuctionState(self, self.config)

  def information_state_tensor_size(self):
    return self.config['n_auctions'] * 2 + 1 

class SequentialAuctionState(pyspiel.State):
  """A python version of the SequentialAuction state."""

  def __init__(self, game, config):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    config = copy.deepcopy(game.config)

    products = self.config['products'] # List with quantities of each product
    opening_prices = self.config['opening_prices']
    n_bidders = len(self.config['bidders'])

    self._game_over = False
    self._is_chance = False
    self.current_auction = 0
    self.final_prices = []
    self._rewards = np.zeros(n_bidders)
    self._returns = np.zeros(n_bidders)

    self.bundles = np.zeros((self.n_players, len(products))) # What does each player currently own?

    self._next_player = pyspiel.PlayerId.SIMULTANEOUS

  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    elif self._is_chance:
      return pyspiel.PlayerId.CHANCE
    else:
      return pyspiel.PlayerId.SIMULTANEOUS

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0
    # Remain, or leave
    return [Action.EXIT, Action.BID]

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    # Tiebreaking - given the winners and the amount available, who to exclude?
    pass # TODO: Compute the number of ways to divvy


    if self.product_index == self.n_products - 1:
      self._game_over = True
    else:
      self.product_index += 1

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    # This is not called at simultaneous-move states.
    assert self._is_chance and not self._game_over

  def _apply_actions(self, actions):
    """Applies the specified actions (per player) to the state."""
    assert not self._is_chance and not self._game_over

    # Are people still in?
    active_bidders = [a == Action.BID for a in actions]
    n_active = sum(active_bidders)
    if n_active > self.product_quantity:
      # Another increment
      self.cur_price *= self.increment
      self.active_bidders = active_bidders
    else:
      # Sold, possibly with a tiebreaker
      self.final_prices.append(self.cur_price)
      self.winners.append(np.nonzero(active_bidders)[0].tolist())
      self._next_player = pyspiel.PlayerId.CHANCE

    # TODO: Where does this go?
    for bidder_id in range(self.n_bidders):
      pass
      # self._rewards[bidder_id] = 0 if bidder_id not  else 10
    self._returns += self._rewards

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return "CHANCE"
    else:
      return Action(action).name

  def information_state_tensor(self, player_id):
    return []

  def information_state_string(self, player_id):
    return str(self.information_state_tensor())

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over

  def rewards(self):
    """Reward at the previous step."""
    return self._rewards

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    return self._returns

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    s = f'Past bids: \n'
    return s

# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, SequentialAuctionGame)