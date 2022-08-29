from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing.sharedctypes import Value
from tkinter.messagebox import NO
import numpy as np
import pandas as pd
import pyspiel
import json
import os
import logging
import enum
import itertools
from typing import List, Dict, Tuple, Optional, Any, Union, Iterable
import math
from open_spiel.python.games import clock_auction_bidders
from functools import cached_property

DEFAULT_MAX_ROUNDS = 25
DEFAULT_AGENT_MEMORY = 3

class ActivityPolicy(enum.IntEnum):
  ON = 0
  OFF = 1

class UndersellPolicy(enum.IntEnum):
  UNDERSELL = 0
  UNDERSELL_ALLOWED = 1

class InformationPolicy(enum.IntEnum):
  SHOW_DEMAND = 0
  HIDE_DEMAND = 1

class ValueFormat(enum.IntEnum):
  LINEAR = 0
  FULL = 1
  MARGINAL = 2

@dataclass
class AuctionParams:
  opening_prices: List[float] = field(default_factory=lambda : [])
  licenses: List[int] = field(default_factory=lambda : [])
  activity: List[int] = field(default_factory=lambda : [])
  num_products: int = 0
  increment: float = 0.1

  max_round: int = DEFAULT_MAX_ROUNDS
  player_types: Dict = None

  all_bids: List[List[int]] = None
  all_bids_activity: List[int] = None

  activity_policy: ActivityPolicy = ActivityPolicy.ON
  undersell_policy: UndersellPolicy = UndersellPolicy.UNDERSELL
  information_policy: InformationPolicy = InformationPolicy.SHOW_DEMAND
  default_player_order: List[List[int]] = None

  tiebreaks: bool = True
  agent_memory: int = DEFAULT_AGENT_MEMORY

  @cached_property
  def max_activity(self):
    return np.array(self.activity) @ np.array(self.licenses)

  @cached_property
  def max_budget(self):
    budgets = []
    for types in self.player_types.values():
        for t in types:
          budget = t['bidder'].get_budget()
          budgets.append(budget)
    return max(budgets)

  def max_opponent_spend(self, player_id):
    max_opps_budgets = []
    for p, types in self.player_types.items():
      if p == player_id:
        continue
      budgets = []
      for t in types:
        budget = t['bidder'].get_budget()
        budgets.append(budget)
      max_opps_budgets.append(max(budgets))
    return np.array(max_opps_budgets).sum()

@dataclass
class TieBreakState:
  tie_breaks_needed: List = field(default_factory=lambda : [])
  selected_order: List = field(default_factory=lambda : [])

  def reset(self):
    self.tie_breaks_needed = []
    self.selected_order = []

@dataclass
class BidderState:
  processed_demand: List[List[int]] = field(default_factory=lambda : [])
  submitted_demand: List[List[int]] = field(default_factory=lambda : [])
  activity: int = None
  bidder: clock_auction_bidders.Bidder = None
  type_index: int = None

def action_to_bundles(licenses):
    bids = []
    for n in licenses:
        b = []
        for i in range(n + 1):
            b.append(i)
        bids.append(b)
    actions = np.array(list(itertools.product(*bids)))
    return actions

_GAME_TYPE = pyspiel.GameType(
    short_name="python_clock_auction",
    long_name="Python Clock Auction",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=10,
    min_num_players=2,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True,
    parameter_specification={
      "filename": 'parameters.json'
      }
    )

def parse_auction_params(file_name):
  if file_name.startswith('/'):
    full_path = file_name
  else:
    logging.info("Reading from env variable CLOCK_AUCTION_CONFIG_DIR. If it is not set, there will be trouble.")
    config_dir = os.environ.get('CLOCK_AUCTION_CONFIG_DIR')
    if config_dir is None:
      raise ValueError("CLOCK_AUCTION_CONFIG_DIR env variable is not set.")
    logging.info(f"CLOCK_AUCTION_CONFIG_DIR={config_dir}")
    full_path = f'{config_dir}/{file_name}';

  logging.info(f"Parsing configuration from {full_path}")
  with open(full_path, 'r') as f:
    game_params = json.load(f)

    players = game_params['players']

    opening_prices = game_params['opening_price']
    licenses = np.array(game_params['licenses'])
    num_products = len(licenses)

    if len(opening_prices) != num_products:
      raise ValueError("Number of opening prices must match number of products.")
    
    activity = game_params['activity']
    if len(activity) != num_products:
      raise ValueError("Number of activity must match number of products.")

    activity_policy = game_params.get('activity_policy', ActivityPolicy.ON)
    if isinstance(activity_policy, bool):
      activity_policy = ActivityPolicy.ON if activity_policy else ActivityPolicy.OFF

    information_policy = game_params.get('information_policy', InformationPolicy.SHOW_DEMAND)  
    if isinstance(information_policy, str):
      information_policy = InformationPolicy[information_policy.upper()]

    undersell_policy = game_params.get('undersell_rule', UndersellPolicy.UNDERSELL)
    if isinstance(undersell_policy, str):
      undersell_policy = UndersellPolicy[undersell_policy.upper()]

    all_bids = action_to_bundles(licenses)
    all_bids_activity = np.array([activity @ bid for bid in all_bids])

    types = defaultdict(list)

    for player_id, player in enumerate(players):
      player_types = player['type']
      for player_type in player_types:
        value_format = player_type.get('value_format', ValueFormat.LINEAR)
        if isinstance(value_format, str):
          value_format = ValueFormat[value_format.upper()]
        budget = player_type['budget']
        prob = player_type['prob']
        pricing_bonus = player_type.get('pricing_bonus', 0)

        values = player_type['value']
        if value_format == ValueFormat.LINEAR:
          if len(values) != num_products:
            raise ValueError("Number of values must match number of products.")
          bidder = clock_auction_bidders.LinearBidder(values, budget, pricing_bonus, all_bids)  
        elif value_format == ValueFormat.FULL:
          if len(values) != len(all_bids):
            raise ValueError("Number of values must match number of bids.")
          bidder = clock_auction_bidders.EnumeratedValueBidder(values, budget, pricing_bonus, all_bids)
        elif value_format == ValueFormat.MARGINAL:
          bidder = clock_auction_bidders.MarginalValueBidder(values, budget, pricing_bonus, all_bids)
        else:
          raise ValueError("Unknown value format")
        
        types[player_id].append(dict(prob=prob, bidder=bidder))

  logging.info("Done config parsing")
  return AuctionParams(
      opening_prices=opening_prices,
      licenses=licenses,
      num_products=num_products,
      activity=activity,
      increment=game_params.get('increment', 0.1),
      max_round=game_params.get('max_rounds', DEFAULT_MAX_ROUNDS),
      player_types=types,
      all_bids=all_bids,
      all_bids_activity=all_bids_activity,
      activity_policy=activity_policy,
      undersell_policy=undersell_policy,
      information_policy=information_policy,
      tiebreaks=game_params.get('tiebreaks', True),
      agent_memory=game_params.get('agent_memory', DEFAULT_MAX_ROUNDS),
      default_player_order=[[p for p in range(len(players))] for _ in range(num_products)]
    )

class ClockAuctionGame(pyspiel.Game):

  def __init__(self, params=None):
    file_name = params.get('filename', 'parameters.json')
    self.auction_params = parse_auction_params(file_name)
    num_players = len(self.auction_params.player_types)

    # Max of # of type draws and tie-breaking
    max_chance_outcomes = max(max([len(v) for v in self.auction_params.player_types.values()]), math.factorial(num_players))

    # You can bid for [0...M_j] for any of the j products
    num_actions = (1 + np.array(self.auction_params.licenses)).prod()

    # MAX AND MIN UTILITY
    self.upper_bounds = []
    self.lower_bounds = []
    open_prices_per_bundles = np.array([self.auction_params.opening_prices @ bid for bid in self.auction_params.all_bids])
    for player_id, types in self.auction_params.player_types.items():
      player_upper_bounds = []
      player_lower_bounds = []
      for t in types:
        bidder = t['bidder']
        # What if you won your favorite package at opening prices?
        bound = bidder.get_profits(open_prices_per_bundles).max()
        player_upper_bounds.append(bound)
        # What if you spent your entire budget and got nothing? (A tighter not implemented bound: if you got the single worst item for you, since you must be paying for something)
        player_lower_bounds.append(-bidder.budget)
      self.upper_bounds.append(max(player_upper_bounds))
      self.lower_bounds.append(min(player_lower_bounds))

    game_info = pyspiel.GameInfo(
        num_distinct_actions=num_actions,
        max_chance_outcomes=max_chance_outcomes, 
        num_players=num_players,
        min_utility=-99999,
        max_utility=max(self.upper_bounds),
        utility_sum=-99999,
        max_game_length=9999)

    super().__init__(_GAME_TYPE, game_info, params)
      
  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return ClockAuctionState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if params is None:
      params = dict()
    
    params['auction_params'] = self.auction_params

    return ClockAuctionObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)

class ClockAuctionState(pyspiel.State):
  """A python version of the Atari Game state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.auction_params = game.auction_params
    self._game_over = False
    self._auction_finished = False
    self._is_chance = True
    self.tie_break_state = TieBreakState()
    self.bidders = []
    self.posted_prices = [np.array(self.auction_params.opening_prices)]
    self.sor_prices = [np.array(self.auction_params.opening_prices)]
    self.clock_prices = [(1 + self.auction_params.increment) * self.sor_prices[0]]
    self.aggregate_demand = []
    self.round = 1
    self._cur_player = 0
    self._final_payments = None
    self.price_increments = np.zeros(self.auction_params.num_products)

  def current_player(self) -> pyspiel.PlayerId:
    """Returns the current player.

    If the game is over, TERMINAL is returned. If the game is at a chance
    node then CHANCE is returned. Otherwise SIMULTANEOUS is returned.
    """
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    elif self._is_chance:
      return pyspiel.PlayerId.CHANCE
    else:
      return self._cur_player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    assert player >= 0 
    legal_actions = []
    bidder = self.bidders[player]

    if len(bidder.submitted_demand) > 0 and sum(bidder.submitted_demand[-1]) == 0:
      # Don't need to recalculate - can only bid 0
      return [0]

    price = np.array(self.clock_prices[-1])
    budget = bidder.bidder.budget
    hard_budget_on = True # TODO: Make param
    positive_profit_on = False # TODO: Make param

    prices = np.array([price @ bid for bid in self.auction_params.all_bids])
    profits = bidder.bidder.get_profits(prices)

    # Note we assume all drops go through. A more sophisticated bidder might think differently (e.g., try to fulfill budget in expectation)
    # Consider e.g. if you drop a product you might get stuck! So you can wind up over your budget if your drop fails
    # Also consider that if you drop a product and get stuck, you only pay SoR on that product

    legal_actions = np.ones(len(self.auction_params.all_bids), dtype=bool)

    if self.auction_params.activity_policy == ActivityPolicy.ON:
      legal_actions[np.where(bidder.activity < self.auction_params.all_bids_activity)[0]] = 0

    if hard_budget_on:
      legal_actions[np.where(prices > budget)[0]] = 0

    if positive_profit_on:
      legal_actions[np.where(profits < 0)[0]] = 0

    # TODO: Shouldn't I be using LegalProfits here? (i.e., accounting for the fact that it needs to be a legal bundle?)
    if not (profits > 0).any():
      # If you have no way to make a profit ever going forwards, just drop out. Helps minimize game size
      return [0]
    else:
      # At least one bid leads to positive profit. Dropping out is never the right thing to do in this case. It will always be action 0
      legal_actions[0] = 0

    if sum(legal_actions) == 0:
      raise ValueError("No legal actions!")
    
    return legal_actions.nonzero()[0]

  def _apply_action(self, action):
    """Applies the specified action to the state.
    """
    if not self.is_chance_node():
      # Colelct the bid in submitted demand
      assert not self._game_over
      bidder = self.bidders[self._cur_player]
      assert len(bidder.processed_demand) == len(bidder.submitted_demand)
      assert self.round - 1 == len(bidder.submitted_demand)
      bid = self.auction_params.all_bids[action]

      if self.auction_params.activity_policy == ActivityPolicy.ON:
        bid_activity_cost = self.auction_params.all_bids_activity[action]
        if bidder.activity < bid_activity_cost:
          raise ValueError(f"Bidder {self._cur_player} is not active enough ({bidder.activity}) to bid on {bid} with cost of {bid_activity_cost}")

      bidder.submitted_demand.append(np.array(bid))

      # Collect bids until we've got from all players: only then can we process the demand
      if self._cur_player == self.num_players() - 1:
        self._handle_bids()
      else:
        self._cur_player += 1 # Advance player and collect more bids

    else: # CHANCE NODE
      if self.round == 1:
        if len(self.bidders) < self.num_players(): # Chance node assigns a value and budget to a player
          self.bidders.append(
            BidderState(
              bidder=self.auction_params.player_types[len(self.bidders)][action]['bidder'],
              activity=self.auction_params.max_activity,
              type_index=action,
            )
          )
        if len(self.bidders) == self.num_players(): # All of the assignments have been made
          self._is_chance = False
      else:
        # Tie breaking 
        ts = self.tie_break_state

        # Assign ordering for this product by indexing into the list of permutations with action
        j = len(ts.selected_order)
        order = list(list(itertools.permutations(ts.tie_breaks_needed[j]))[action])

        # Pad with players that aren't in the list so we have a complete ordering (recall that tie_breaks_needed has only the players in conflict)
        for player_id in range(self.num_players()):
          if player_id not in order:
            order.append(player_id)
        ts.selected_order.append(order)
        j += 1

        # Skip ahead to the next point we need a tie
        while j < self.auction_params.num_products:
          if len(ts.tie_breaks_needed[j]) <= 1:
            ts.selected_order.append(list(self.auction_params.default_player_order[j]))
            j += 1
          else:
            break # We need to actually do this one and it will require another chance node
        
        if j == self.auction_params.num_products: # Done with chance nodes
          self._process_bids(ts.selected_order)

  def _handle_bids(self):
    # Demand Processing
    if self.round == 1 or self.auction_params.undersell_policy == UndersellPolicy.UNDERSELL_ALLOWED:
      # Just copy it straight over
      for bidder in self.bidders:
        bid = bidder.submitted_demand[-1]
        bidder.processed_demand.append(np.array(bid))
      self._post_process()
    elif self.auction_params.undersell_policy == UndersellPolicy.UNDERSELL:
      tiebreaks_not_needed = self._determine_tiebreaks()
      if tiebreaks_not_needed: # No chance node required. Just get on with the game 
        self._process_bids(self.auction_params.default_player_order)
      else:
        self._is_chance = True
    else:
      raise ValueError("Unknown undersell policy")

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      if len(self.bidders) < self.num_players():
        return f'Assign player {len(self.bidders)} type {self.auction_params.player_types[len(self.bidders)][action]["bidder"]}'
      else:
        return f'Tie-break action {action}'
    else:
      bid = self.auction_params.all_bids[action]
      activity = self.auction_params.all_bids_activity[action]
      price = bid @ self.clock_prices[-1] 
      return f'Bid for {bid} licenses @ ${price:.2f} with activity {activity}'

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    assert self._game_over
    assert self._auction_finished

    self._final_payments = np.zeros(self.num_players())
    returns = np.zeros_like(self._final_payments)
    final_prices = self.posted_prices[-1]

    for player_id, bidder in enumerate(self.bidders):
      final_bid = bidder.processed_demand[-1]
      payment = final_bid @ final_prices
      self._final_payments[player_id] = payment
      returns[player_id] = bidder.bidder.value_for_package(final_bid, -1) - payment

    return returns

  def __str__(self):
    with np.printoptions(precision=3):

      """String for debug purposes. No particular semantics are required."""
      result = f'Round: {self.round}\n'

      # Player types
      for player_id, bidder in enumerate(self.bidders):
        result += f'Player {player_id}: {bidder.bidder}\n'

      if self.round > 1:
        result += f'Price: {self.posted_prices[-1]}\n'

        result += 'Processed Demand:\n'
        for player_id, player in enumerate(self.bidders):
          if len(player.processed_demand) > 0:
            result += f'{player.processed_demand[-1]}\n'

        if len(self.aggregate_demand) > 0:
          result += f'Aggregate demand: {self.aggregate_demand[-1]}\n'

      if self._auction_finished:
        for player_id, player in enumerate(self.bidders):
          result += f'Final bids player {player_id}: {player.processed_demand[-1]}\n'

    return result

  def chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self._is_chance
    if len(self.bidders) < self.num_players(): # Chance node is assigning a type
      player_index = len(self.bidders)
      player_types = self.auction_params.player_types[player_index]
      probs = [t['prob'] for t in player_types]
    else: # Chance node is breaking a tie
      ts = self.tie_break_state
      chance_outcomes_required = math.factorial(len(ts.tie_breaks_needed[len(ts.selected_order)]))
      probs = [1 / chance_outcomes_required] * chance_outcomes_required
    
    return list(enumerate(probs))

  def _post_process(self):
    # Calculate aggregate demand
    aggregate_demand = np.zeros(self.auction_params.num_products, dtype=int)
    for bidder in self.bidders:
      bid = bidder.processed_demand[-1]
      
      # Lower activity based on processed demand (TODO: May want to revisit this for grace period)
      bidder.activity = bid @ self.auction_params.activity
      aggregate_demand += bid

    # Calculate excess demand
    excess_demand = aggregate_demand > self.auction_params.licenses
    
    self.aggregate_demand.append(aggregate_demand)

    if excess_demand.any():
      # Normal case: Increment price for overdemanded items, leave other items alone
      next_price = np.zeros(self.auction_params.num_products)
      next_clock = np.zeros_like(next_price)
      for j in range(self.auction_params.num_products):
        if excess_demand[j]:
          self.price_increments[j] += 1
        next_price[j] = self.clock_prices[-1][j] if excess_demand[j] else self.sor_prices[-1][j]
        next_clock[j] = next_price[j] * (1 + self.auction_params.increment)
      self.posted_prices.append(next_price)
      self.sor_prices.append(next_price)
      self.clock_prices.append(next_clock)
    else:
      # Demand <= supply for each item. We are finished
      self._auction_finished = True
      self._game_over = True
      self.posted_prices.append(np.array(self.posted_prices[-1]))

    if not self._auction_finished:
      self.round += 1
      if self.round > self.auction_params.max_round:
        # An alternative: set game_over = True (auction_finished will still be false) and simply track this. Maybe give large negative rewards. But right now this seems more obvious as a way of triggering
        raise ValueError("Auction went on too long")

    self._cur_player = 0
    self._is_chance = False

  def _process_bids(self, player_order):
    assert len(player_order) == self.auction_params.num_products
    for j in range(self.auction_params.num_products):
      assert len(player_order[j]) == self.num_players()
    
    current_agg = self.aggregate_demand[-1]

    # Copy over the current aggregate demand
    # TODO: For now points is a zero vector, but possible grace period implementations would change that

    bids = []
    requested_changes = []
    points = [bidder.activity for bidder in self.bidders]
    for player_id, bidder in enumerate(self.bidders):
      last_round_holdings = bidder.processed_demand[-1]
      bids.append(last_round_holdings)

      rq = np.zeros(self.auction_params.num_products)
      for j in range(self.auction_params.num_products):
        delta = bidder.submitted_demand[-1][j] - last_round_holdings[j]
        rq[j] = delta
        points[player_id] -= last_round_holdings[j] * self.auction_params.activity[j]
      requested_changes.append(rq)

    changed = True

    while changed:
      changed = False
      for j in range(self.auction_params.num_products):
        for player_id in player_order[j]:
          bid = bids[player_id]
          changes = requested_changes[player_id]

          # Process drops
          if changes[j] < 0:
            drop_room = current_agg[j] - self.auction_params.licenses[j]
            if drop_room > 0:
              amount = min(drop_room, -changes[j])
              bid[j] -= amount
              assert bid[j] >= 0
              changed = True
              points[player_id] += amount * self.auction_params.activity[j]
              current_agg[j] -= amount
              changes[j] += amount
        
          # Process pickups
          while changes[j] > 0 and (self.auction_params.activity_policy == ActivityPolicy.OFF or points[player_id] >= self.auction_params.activity[j]):
            bid[j] += 1
            assert bid[j] <= self.auction_params.licenses[j]
            current_agg[j] += 1
            changed = True
            points[player_id] -= self.auction_params.activity[j]
            changes[j] -= 1

    # Finally, copy over submitted -> processed
    for player_id, bidder in enumerate(self.bidders):
      for product_id in range(self.auction_params.num_products):
        assert bids[player_id][product_id] >= 0
        assert bids[player_id][product_id] <= self.auction_params.licenses[product_id]
        assert bids[player_id][product_id] <= max(bidder.submitted_demand[-1][product_id], bidder.processed_demand[-1][product_id]) # Either what you asked for, or what you used to have

      bidder.processed_demand.append(np.array(bids[player_id]))
      assert len(bidder.processed_demand) == len(bidder.submitted_demand)
        
    self._post_process()

  def _determine_tiebreaks(self):
    # Step 1: Figure out for each product whether we may be in a tie-breaking situation. Note that we can have false positives, they "just" make the game bigger. 
    #         One necessary condition: At least two people want to drop the same product AND the combined dropping will take the product below supply
    #         Note that this doesn't consider demand that might be added to the product - this could resolve the issue, but if that pick-up can only be processed conditional on another drop, it gets more complicated... We ignore this for now.
    #

    # TODO: A much simpler impl would just select one of factorial(num_players) arrangements for each product in advance and use that, tie breaks required or not (i.e., always put a chance node before bids are processed that determines processing order). So why do it this way? Because that's much worse for tabular algorithms, and they can help debug

    ts = self.tie_break_state
    ts.reset()

    if not self.auction_params.tiebreaks or self.round == 1:
      return False

    drops_per_product = np.zeros((self.num_players(), self.auction_params.num_products))
    for player_id, bidder in enumerate(self.bidders):
      drops_per_product[player_id] = np.abs(bidder.processed_demand[-1] - bidder.submitted_demand[-1])

    total_drops = np.sum(drops_per_product, axis=0)

    tie_breaking_not_needed = True    
    current_agg = self.aggregate_demand[-1]
    for j in range(self.auction_params.num_products):
      tiebreaks = []
      if current_agg[j] - total_drops[j] < self.auction_params.licenses[j]:
        for player_id in range(self.num_players()):
          if drops_per_product[player_id][j] > 0:
            tiebreaks.append(player_id)
      if len(tiebreaks) > 1: 
        tie_breaking_not_needed = False
      else:
        ts.selected_order.append(list(self.auction_params.default_player_order[j]))
      ts.tie_breaks_needed.append(tiebreaks)

    return tie_breaking_not_needed


  def get_final_payments(self):
    assert self._game_over
    return self._final_payments

  def get_allocation(self):
    assert self._game_over
    return [bidder.processed_demand[-1] for bidder in self.bidders]


class ClockAuctionObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""
  
  def __init__(self, iig_obs_type, params):
    auction_params = params.get('auction_params')
    self.normalize = params.get('normalize', True) # Raw features? Or normalized ones
    if not isinstance(auction_params, AuctionParams):
      raise ValueError("params must be an AuctionParams object")
    self.auction_params = auction_params

    num_players = len(auction_params.player_types)
    num_products = auction_params.num_products

    # self.round_buffer = 100 if iig_obs_type.perfect_recall else auction_params.agent_memory
    self.round_buffer = auction_params.agent_memory # TODO: We are abusing the API here a bit. Only offers binary perfect recall or not, but we want to interpolate.
    length = self.round_buffer * num_products
    shape = (self.round_buffer, num_products)

    """Initializes an empty observation tensor."""
    # Determine which observation pieces we want to include.
    # NOTE: It should be possible to use the params to exclude some of these if we want a smaller input to the NN (or to have the NN reassamble the tensor from specific pieces).

    pieces = [("player", num_players, (num_players,))] 
    if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      # 1-hot type encoding
      max_num_types = max([len(p) for p in auction_params.player_types.values()])
      pieces.append(("bidder_type", max_num_types, (max_num_types,)))
      pieces.append(("activity", 1, (1,)))
      pieces.append(("sor_exposure", 1, (1,)))

      pieces.append(("submitted_demand_history", length, shape))
      pieces.append(("processed_demand_history", length, shape))
      num_bundles = len(auction_params.all_bids)
      pieces.append(("sor_profits", num_bundles, (num_bundles,)))
      pieces.append(("clock_profits", num_bundles, (num_bundles,)))

    if iig_obs_type.public_info:
      # 1-hot round encoding
      pieces.append(("round", self.round_buffer, (self.round_buffer,)))
      pieces.append(("agg_demand_history", length, shape))
      pieces.append(("posted_price_history", (self.round_buffer + 1) * num_products, (self.round_buffer + 1, num_products))) # slightly different shape b/c there is a real initial value
      pieces.append(("clock_prices", num_products, (num_products,)))
      pieces.append(("price_increments", num_products, (num_products,)))

    # Build the single flat tensor.
    total_size = sum(size for name, size, shape in pieces)
    self.tensor = np.zeros(total_size, np.float32)

    # Build the named & reshaped views of the bits of the flat tensor.
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    self.tensor.fill(0)
    length = min(state.round - 1, self.round_buffer)
    start_ind = max(0, state.round - self.round_buffer)
    end_ind = start_ind + length

    if "player" in self.dict:
      self.dict["player"][player] = 1
    if "round" in self.dict:
      self.dict["round"][state.round - 1] = 1

    # These require the bidder to be initialized
    if len(state.bidders) > player:
      if "bidder_type" in self.dict:
        self.dict["bidder_type"][state.bidders[player].type_index] = 1
      if "activity" in self.dict:
        activity = state.bidders[player].activity
        if self.normalize:
          activity /= self.auction_params.max_activity
        self.dict["activity"][0] = activity
      if "sor_profits" in self.dict:
        price = np.array(state.sor_prices[-1])
        prices = np.array([price @ bid for bid in self.auction_params.all_bids])
        profits = state.bidders[player].bidder.get_profits(prices)
        if self.normalize:
          profits = profits / self.auction_params.max_budget
        self.dict["sor_profits"] = profits
      if "clock_profits" in self.dict:
        price = np.array(state.clock_prices[-1])
        prices = np.array([price @ bid for bid in self.auction_params.all_bids])
        profits = state.bidders[player].bidder.get_profits(prices)
        if self.normalize:
          profits = profits / self.auction_params.max_budget
        self.dict["clock_profits"] = profits

    if state.round > 1:
      if "agg_demand_history" in self.dict:
        agg_demand_history = np.array(state.aggregate_demand)[start_ind:end_ind]
        if self.normalize:
          agg_demand_history = agg_demand_history / self.auction_params.licenses
        self.dict["agg_demand_history"][:length] = agg_demand_history
      if "submitted_demand_history" in self.dict:
        submitted_demand_history = np.array(state.bidders[player].submitted_demand)[start_ind:end_ind]
        if self.normalize:
          submitted_demand_history = submitted_demand_history / self.auction_params.licenses
        self.dict["submitted_demand_history"][:length] = submitted_demand_history
      if "processed_demand_history" in self.dict:
        processed_demand_history = np.array(state.bidders[player].processed_demand)[start_ind:end_ind]
        if self.normalize:
          processed_demand_history = processed_demand_history / self.auction_params.licenses
        self.dict["processed_demand_history"][:length] = processed_demand_history
      if "sor_exposure" in self.dict:
        sor_exposure = state.bidders[player].processed_demand[-1] @ state.sor_prices[-1]
        if self.normalize:
          sor_exposure = sor_exposure / self.auction_params.max_budget # TODO: Why not use a player specific bound here?
        self.dict["sor_exposure"] = sor_exposure

    if "posted_price_history" in self.dict:
      posted_prices = np.array(state.posted_prices)[start_ind:end_ind + 1]
      if self.normalize:
        posted_prices = posted_prices / self.auction_params.max_budget
      self.dict["posted_price_history"][:length + 1] = posted_prices

    if "clock_prices" in self.dict:
      clock_prices = np.array(state.clock_prices[-1])
      if self.normalize:
        clock_prices = clock_prices / self.auction_params.max_budget
      self.dict['clock_prices'] = clock_prices

    if "price_increments" in self.dict:
      price_increments = np.array(state.price_increments)
      if self.normalize:
        price_increments = price_increments / self.auction_params.max_round
      self.dict['price_increments'] = price_increments

    if np.isnan(self.tensor).any():
      raise ValueError(f"NaN in observation {self.dict}")

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []
    if "player" in self.dict:
      pieces.append(f"p{player}")
    if "bidder_type" in self.dict:
      pieces.append(f"t{state.bidders[player].type_index}")
    if "activity" in self.dict:
      pieces.append(f"a{state.bidders[player].activity}")
    if "round" in self.dict:
      pieces.append(f"r{state.round}")
    if "agg_demand_history" in self.dict:
      pieces.append(f"agg{state.aggregate_demand}")
    if "submitted_demand_history" in self.dict:
      pieces.append(f"sub{state.bidders[player].submitted_demand}")
    if "processed_demand_history" in self.dict:
      pieces.append(f"proc{state.bidders[player].processed_demand}")
    if "posted_price_history" in self.dict:
      pieces.append(f"posted{state.posted_prices}")
    if "clock_prices" in self.dict:
      pieces.append(f"clock{state.clock_prices[-1]}")
    if "sor_exposure" in self.dict and state.round > 1:
      pieces.append(f"sor_exposure{state.bidders[player].processed_demand[-1] @ state.sor_prices[-1]}")
    if "price_increments" in self.dict and state.round > 1:
      pieces.append(f"increments{state.price_increments}")

    return " ".join(str(p) for p in pieces)

# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, ClockAuctionGame)
