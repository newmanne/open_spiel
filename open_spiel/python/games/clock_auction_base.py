"""
Data structures and constants used throughout the clock auction implementation.
"""

import enum
import itertools
from typing import List, Dict
from dataclasses import dataclass, field
from functools import cached_property
import numpy as np

DEFAULT_MAX_ROUNDS = 100
DEFAULT_AGENT_MEMORY = 1
MAX_CACHE_SIZE = 100_000

class ActivityPolicy(enum.IntEnum):
  ON = 0
  OFF = 1

class UndersellPolicy(enum.IntEnum):
  UNDERSELL = 0
  UNDERSELL_ALLOWED = 1

class InformationPolicy(enum.IntEnum):
  SHOW_DEMAND = 0
  HIDE_DEMAND = 1

class InformationPolicyConstants:
  OVER_DEMAND = 1
  AT_SUPPLY = 0
  UNDER_DEMAND = 3

class ValueFormat(enum.IntEnum):
  LINEAR = 0
  FULL = 1
  MARGINAL = 2

class TiebreakingPolicy(enum.IntEnum):
  DROP_BY_PLAYER = 0
  DROP_BY_LICENSE = 1

@dataclass
class LotteryState:
  activity: List[int] = field(default_factory=lambda : [])
  processed_demand: List[List[int]] = field(default_factory=lambda : [])
  submitted_demand: List[List[int]] = field(default_factory=lambda : [])

@dataclass
class AuctionParams:
  opening_prices: List[float] = field(default_factory=lambda : [])
  licenses: List[int] = field(default_factory=lambda : [])
  license_names: List[str] = field(default_factory=lambda : [])
  activity: List[int] = field(default_factory=lambda : [])
  num_products: int = 0
  increment: float = 0.1
  reveal_type_round: int = None
  skip_single_chance_nodes: bool = True # If true, skip chance nodes that only have 1 outcome. Results in smaller game tree. Only relevant if the algorithm actually does a full traversal.

  max_round: int = DEFAULT_MAX_ROUNDS
  player_types: Dict = None

  all_bids: List[List[int]] = None
  bid_to_index: Dict = None
  all_bids_activity: List[int] = None

  activity_policy: ActivityPolicy = ActivityPolicy.ON
  grace_rounds: int = 1

  undersell_policy: UndersellPolicy = UndersellPolicy.UNDERSELL
  information_policy: InformationPolicy = InformationPolicy.SHOW_DEMAND
  tiebreaking_policy: TiebreakingPolicy = TiebreakingPolicy.DROP_BY_PLAYER

  agent_memory: int = DEFAULT_AGENT_MEMORY
  heuristic_deviations: int = None
  reward_shaping: str = None

  sor_bid_bonus_rho: float = 1. # Units of bonus points up for grabs for bidding truthfully. Still kinda sucks at breaking indifference when profits are similar because the bonuses will be correspondingly similar.

  base_game_name: str = ''
  rule: str = ''

  @cached_property
  def max_activity(self):
    return np.array(self.activity) @ np.array(self.licenses)

  def max_budget_for_player(self, player_id):
    return max([t['bidder'].get_budget() for t in self.player_types[player_id]])

  @cached_property
  def max_budget(self):
    return max([self.max_budget_for_player(p) for p in range(len(self.player_types))])

  @cached_property
  def max_total_spend(self):
    return sum([self.max_budget_for_player(p) for p in range(len(self.player_types))])

  def max_opponent_spend(self, player_id):
    return sum([self.max_budget_for_player(p) for p in range(len(self.player_types)) if p != player_id])

  @cached_property
  def max_opponent_spends(self):
    return np.array([self.max_opponent_spend(p) for p in range(len(self.player_types))])
    

@dataclass
class BidderState:
  processed_demand: List[List[int]] = field(default_factory=lambda : [])
  submitted_demand: List[List[int]] = field(default_factory=lambda : [])
  activity: List[int] = field(default_factory=lambda : [])
  bidder: object = None # clock_auction_bidders.Bidder (type is clock_auction_bidders.Bidder but not worth the import chaos)
  type_index: int = None
  max_possible_activity: int = 0
  grace_rounds: int = 1

  def get_max_activity(self):
    if len(self.activity) == 0:
      return self.max_possible_activity
    else:
      return max(self.activity[-self.grace_rounds:])

def action_to_bundles(licenses):
    bids = []
    for n in licenses:
        b = []
        for i in range(n + 1):
            b.append(i)
        bids.append(b)
    actions = np.array(list(itertools.product(*bids)))
    return actions
