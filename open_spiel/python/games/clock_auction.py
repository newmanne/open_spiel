from collections import defaultdict
import numpy as np
from open_spiel.python.examples.ubc_utils import players_not_me, pulp_solve, random_string, permute_array
import pyspiel
import logging
import math
from open_spiel.python.games.clock_auction_parser import parse_auction_params
from open_spiel.python.games.clock_auction_observer import ClockAuctionObserver

from cachetools import LRUCache
from open_spiel.python.games.clock_auction_base import AuctionParams, LotteryState, ActivityPolicy, UndersellPolicy, InformationPolicy, TiebreakingPolicy, InformationPolicyConstants, ValueFormat, BidderState, MAX_CACHE_SIZE
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, LpBinary, lpSum, lpDot, LpMaximize, LpInteger, value
import pandas as pd
from functools import lru_cache, cached_property
import inspect

ACTION_TIEBREAK_EPS = 0

def make_key(bidders):
  key = (tuple([(b.get_max_activity(), tuple(b.processed_demand[-1]), tuple(b.submitted_demand[-1])) for b in bidders]))
  return key

def _apply_bid(auction_params, change_request, bid, points, current_agg):
  (player_id, product_id, delta) = change_request
  changed = False

  # Process drop
  if delta < 0:
    drop_room = current_agg[product_id] - auction_params.licenses[product_id]
    if drop_room > 0:
      amount = min(drop_room, -delta)
      bid[product_id] -= amount
      delta += amount
      assert bid[product_id] >= 0
      changed = True
      points[player_id] += amount * auction_params.activity[product_id]
      current_agg[product_id] -= amount

  # Process pickup
  while delta > 0 and (auction_params.activity_policy == ActivityPolicy.OFF or points[player_id] >= auction_params.activity[product_id]):
    bid[product_id] += 1
    assert bid[product_id] <= auction_params.licenses[product_id]
    current_agg[product_id] += 1
    changed = True
    points[player_id] -= auction_params.activity[product_id]
    delta -= 1

  return changed, delta

def stateless_process_bids(auction_params, bidders, current_agg, processing_queue):
  # Compute the output without modifying any state and return the solution (everyone's processed demand)
  points = [bidder.get_max_activity() for bidder in bidders]
  bids = []
  for player_id, bidder in enumerate(bidders):
    last_round_holdings = bidder.processed_demand[-1]
    for j in range(auction_params.num_products):
      points[player_id] -= last_round_holdings[j] * auction_params.activity[j]
    bids.append(last_round_holdings.copy())

  """
  for (player_id, product_id, delta) in self.processing_queue:
    - try to process it
    - if it fails, add it to the unfinished queue

    - try:
      - process each bid in the unfinished queue; restart from beginning of unfinished queue if even partially successful
    - until nothing changes
  """
  starting_agg_demand = current_agg.copy()
  unfinished_queue = []

  for change_request in processing_queue:
    (player_id, product_id, delta) = change_request
    _, new_delta = _apply_bid(auction_params, change_request, bids[player_id], points, current_agg)

    if new_delta != 0: # The bid is not fully finished
      unfinished_queue.append((player_id, product_id, new_delta))

    unfinished_processing = True

    # Try to process the unfinished queue
    while unfinished_processing:
      unfinished_processing = False
      for idx, unfinished_change_request in enumerate(list(unfinished_queue)):
        (player_id, product_id, delta) = unfinished_change_request
        unfinished_change, new_delta = _apply_bid(auction_params, unfinished_change_request, bids[player_id], points, current_agg)
        if new_delta == 0:
          del unfinished_queue[idx]
        else:
          unfinished_queue[idx] = (player_id, product_id, new_delta)
        if unfinished_change:
          unfinished_processing = True
          break

  # Sanity checks
  for player_id, bidder in enumerate(bidders):
    for product_id in range(auction_params.num_products):
      assert 0 <= bids[player_id][product_id] <= min(auction_params.licenses[product_id], max(bidder.submitted_demand[-1][product_id], bidder.processed_demand[-1][product_id])) # Can't be bigger than supply, and can't be bigger than what you asked for, or what you used to have
      
  for j in range(auction_params.num_products):
    if (current_agg[j] < auction_params.licenses[j]) and (current_agg[j] < starting_agg_demand[j]):
      raise ValueError("Aggregate demand fell for underdemanded product {}".format(j))
  
  return bids

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
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=False,
    parameter_specification={
      "filename": 'parameters.json'
      }
    )

class ClockAuctionGame(pyspiel.Game):

  def __init__(self, params=None):
    file_name = params.get('filename', 'parameters.json')
    self.auction_params = parse_auction_params(file_name)

    if self.auction_params.fold_randomness:
      # Maps from a state to a set of outcomes, with probs 
      # How to determine a unique entry? If I face the same vector of (activity, processed demands, submitted demand), that implies same lottery mapping to new processed demands regardless of all else!
      # Note that you could imagine pickling this
      logging.info("Folding randomness...")
      self.lottery_cache = LRUCache(maxsize=MAX_CACHE_SIZE)
    
    self.state_cache = LRUCache(maxsize=MAX_CACHE_SIZE)

    num_players = len(self.auction_params.player_types)

    # Max of # of type draws and tie-breaking
    max_chance_outcomes = max(max([len(v) for v in self.auction_params.player_types.values()]), math.factorial(num_players))

    # You can bid for [0...M_j] for any of the j products
    num_actions = (1 + np.array(self.auction_params.licenses)).prod()

    # MAX AND MIN UTILITY
    self.upper_bounds = []
    self.lower_bounds = []
    for types in self.auction_params.player_types.values():
      player_upper_bounds = []
      player_lower_bounds = []
      for t in types:
        bidder = t['bidder']
        # What if you won your favorite package at opening prices?
        bound = bidder.get_profits(self.auction_params.opening_prices).max()
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

  def observation_tensor_shape(self):
    """Returns the shape of the observation tensor."""
    observer = self.make_py_observer()
    return observer.observation_shape

  def clear_cache(self):
    if hasattr(self, 'lottery_cache'):
      logging.info("Clearing lottery cache")
      self.lottery_cache.clear()
    if hasattr(self, 'state_cache'):
      logging.info("Clearing state cache")
      self.state_cache.clear()

class ClockAuctionState(pyspiel.State):
  """A python version of the Atari Game state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.auction_params = game.auction_params
    self._game_over = False
    self._auction_finished = False
    self._is_chance = True
    self.bidders = []
    self.posted_prices = [np.array(self.auction_params.opening_prices)]
    self.sor_prices = [np.array(self.auction_params.opening_prices)]
    self.clock_prices = [np.array(self.auction_params.opening_prices) * (1 + self.auction_params.increment)]
    self.aggregate_demand = [np.zeros(self.auction_params.num_products, dtype=int)]
    self.round = 1
    self._cur_player = 0
    self._final_payments = None
    self.price_increments = np.zeros(self.auction_params.num_products, dtype=int)
    self.legal_action_mask = np.ones(len(self.auction_params.all_bids), dtype=bool)
    self.processing_queue = None
    self.folded_chance_outcomes = None

    # TODO: This should really be a decorator or at the very least be read from auction params but I'm tired
    self.tiebreak_actions = True
    self._action_rewards = defaultdict(float)

  # An LRU cache here is a bad idea - it will get too big. Just use the game cache
  def child(self, action):
    key = tuple(self.history() + [action])
    kid = self.get_game().state_cache.get(key) 
    if kid is None:
      kid = super(ClockAuctionState, self).child(action)

      # don't clone AuctionParams and bidders
      kid.auction_params = self.auction_params
      for i in range(len(self.bidders)):
        kid.bidders[i].bidder = self.bidders[i].bidder

      self.get_game().state_cache[key] = kid
    return kid
  
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

  @cached_property
  def _cached_legal_actions(self):
    """Returns a list of legal actions, sorted in ascending order."""
    player = self._cur_player
    assert player >= 0 
    legal_actions = []
    bidder = self.bidders[player]

    if (self.round >= self.auction_params.max_round) or (len(bidder.submitted_demand) > 1 and bidder.submitted_demand[-1].sum() == 0):
      # Don't need to recalculate - can only bid 0
      # (If you're over max rounds, the government randomly tie-breaks)
      return [0]

    sor_price = np.asarray(self.sor_prices[-1])
    # budget = bidder.bidder.budget
    # hard_budget_on = True # TODO: Make param.
    # positive_profit_on = False # TODO: Make param

    # sor_bundle_prices = self.auction_params.all_bids @ sor_price
    profits = bidder.bidder.get_profits(sor_price)

    # Note we assume all drops go through. A more sophisticated bidder might think differently (e.g., try to fulfill budget in expectation)
    # Consider e.g. if you drop a product you might get stuck! So you can wind up over your budget if your drop fails
    # Also consider that if you drop a product and get stuck, you only pay SoR on that product

    legal_actions = self.legal_action_mask.copy()
    legal_actions[np.where(bidder.bidder.get_values() < 0)[0]] = 0 # Shorthand to make actions illegal, does not really mean "negative value"

    if self.auction_params.activity_policy == ActivityPolicy.ON:
      legal_actions[np.where(bidder.get_max_activity() < self.auction_params.all_bids_activity)[0]] = 0


    # if hard_budget_on:
    #   legal_actions[prices > budget] = 0

    # if positive_profit_on:
    #   legal_actions[profits < 0] = 0

    if not (profits[legal_actions] > 0).any():
      # If you have no way to make a profit ever going forwards, just drop out. This is done at SoR prices. If you stay in longer, things get (weakly) worse! Helps minimize game size
      return [0]
    # elif bidder.bidder.drop_out_heuristic:
    #   # At least one bid leads to positive profit. Dropping out is never the right thing to do in this case. It will always be action 0
    #   legal_actions[0] = 0

    if bidder.bidder.straightforward:
      clock_price = np.asarray(self.clock_prices[-1])
      clock_profits = bidder.bidder.get_profits(clock_price) # Select your favorite bundle at the clock prices is our naive straightforward bidder. This is problematic though because you can get stuck etc.
      clock_profits[np.where(legal_actions == 0)[0]] = -1
      return [np.argmax(clock_profits)]

    if legal_actions.sum() == 0:
      raise ValueError("No legal actions!\n{self}")
    
    return legal_actions.nonzero()[0]

  def _legal_actions(self, player):
    return self._cached_legal_actions

  # Override
  def apply_action(self, action):
    raise ValueError("You are circumventing caching!!! Use state.child()")

  def clear_cached_properties(self):
    # Need to do this because clone() should force a recompute
    self.__dict__.pop('_as_string', None) 
    self.__dict__.pop('_chance_outcomes', None) 
    self.__dict__.pop('_returns', None) 
    self.__dict__.pop('_my_hash', None) 
    self.__dict__.pop('_cached_legal_actions', None) 

  def _apply_action(self, action):
    """Applies the specified action to the state.
    """
    # IF YOU APPLY AN ACTION, YOU MODIFY THE STATE SO YOU MUST CLEAR ALL CACHED PROPERTIES
    self.clear_cached_properties()

    if not self.is_chance_node():
      # Colelct the bid in submitted demand
      assert not self._game_over
      bidder = self.bidders[self._cur_player]
      assert len(bidder.processed_demand) == len(bidder.submitted_demand)
      assert self.round == len(bidder.submitted_demand)
      bid = self.auction_params.all_bids[action]

      ### Tiebreak to prefer larger actions
      if self.tiebreak_actions:
        self._action_rewards[self._cur_player] += action * ACTION_TIEBREAK_EPS
      ###

      if self.auction_params.activity_policy == ActivityPolicy.ON:
        bid_activity_cost = self.auction_params.all_bids_activity[action]
        if bidder.get_max_activity() < bid_activity_cost:
          raise ValueError(f"Bidder {self._cur_player} is not active enough ({bidder.get_max_activity()}) to bid on {bid} with cost of {bid_activity_cost}")

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
              submitted_demand=[np.zeros(self.auction_params.num_products, dtype=int)], # Start with this dummy entry so we can always index by round
              processed_demand=[np.zeros(self.auction_params.num_products, dtype=int)],
              bidder=self.auction_params.player_types[len(self.bidders)][action]['bidder'],
              activity=[self.auction_params.max_activity],
              type_index=action,
              grace_rounds=self.auction_params.grace_rounds,
            )
          )
        if len(self.bidders) == self.num_players(): # All of the assignments have been made
          self._is_chance = False
      else:
        self._handle_chance_tiebreak(action)


  def _handle_chance_tiebreak(self, action):
    # Tie breaking 
    if self.auction_params.fold_randomness:
      # Let's just index into the solution that we must have already computed
      processed = self.folded_chance_outcomes['results'][action]
      # key = make_key(self.bidders)
      # processed = self.get_game().lottery_cache.get(key)['results'][action] # TODO: Error when undersell policy is off - you will need to comptue explicitly
    else:
      self.processing_queue = permute_array(self.processing_queue, action)
      processed = stateless_process_bids(self.auction_params, self.bidders, self.aggregate_demand[-1].copy(), self.processing_queue)

    # Actually copy
    for player_id, bidder in enumerate(self.bidders):
      bidder.processed_demand.append(np.asarray(processed[player_id]))
      assert len(bidder.processed_demand) == len(bidder.submitted_demand)    
    
    self._post_process()


  def _generate_processing_queue(self):
    # Generate a processing queue, in a fixed order, where each element is a bid of the form (player, product, delta)
    self.processing_queue = []
    for player_id, bidder in enumerate(self.bidders):
      for product_id in range(self.auction_params.num_products):
        current = bidder.processed_demand[-1][product_id]
        requested = bidder.submitted_demand[-1][product_id]

        if requested > current:
          # always process demand increases all at once
          self.processing_queue.append((player_id, product_id, requested - current))
        elif requested < current:
          # in DROP_BY_PLAYER, also process demand _decreases_ all at once
          if self.auction_params.tiebreaking_policy == TiebreakingPolicy.DROP_BY_PLAYER:
            self.processing_queue.append((player_id, product_id, requested - current))
          # in DROP_BY_LICENSE, process demand decreases one at a time
          elif self.auction_params.tiebreaking_policy == TiebreakingPolicy.DROP_BY_LICENSE:
            for _ in range(current - requested):
              self.processing_queue.append((player_id, product_id, -1))
          else:
            raise ValueError(f"Unknown tiebreaking policy: {self.auction_params.tiebreaking_policy}")
        else: # requested == current
          pass

  def _handle_bids(self):
    # Demand Processing
    if (self.round == 1) or (self.auction_params.undersell_policy == UndersellPolicy.UNDERSELL_ALLOWED):
      # Just copy it straight over
      for bidder in self.bidders:
        bid = bidder.submitted_demand[-1]
        bidder.processed_demand.append(np.asarray(bid))
      self._post_process()
    elif self.auction_params.undersell_policy == UndersellPolicy.UNDERSELL:
      self._generate_processing_queue()

      if self.auction_params.fold_randomness:
        # 1) Is this in the cache?
        key = make_key(self.bidders)
        res = self.get_game().lottery_cache.get(key)
        if res is None:
          seen = []
          # 1) Find the number of permtuations
          n_permutations = math.factorial(len(self.processing_queue))
          # 2) Try ALL of them. 
          # TODO: Surely you could be much smarter about this (e.g, don't I need >= 2 drop bids for this ordering to matter? And even then, given that I always e.g. put drop bids at the front, couldn't I achieve every outcome still while having many fewer permutations?)
          # If it becomes the bottleneck, then use https://more-itertools.readthedocs.io/en/stable/api.html#more_itertools.distinct_permutations
          for i in range(n_permutations):
            queue = permute_array(self.processing_queue, i)
            processed = stateless_process_bids(self.auction_params, self.bidders, self.aggregate_demand[-1].copy(), queue)
            seen.append(tuple(tuple(p) for p in processed))

          # 3) Only keep the unique ones
          # OUTCOMES
          p = pd.Series(seen).value_counts(normalize=True).to_dict()
          res = {
            'lottery': list(p.values()),
            'results': list(p.keys())
          }
          # print(f"New state not in cache, processed... {n_permutations} outcomes into {len(p)} outcomes")
          # 4) Cache it
          self.get_game().lottery_cache[key] = res

        self.folded_chance_outcomes = res

        ### Was there only a single outcome? If so, let's remove a chance node here so the game tree is smaller (good for cache size)
        if self.auction_params.skip_single_chance_nodes and len(self.folded_chance_outcomes['lottery']) == 1:
          self._handle_chance_tiebreak(0)
        else:
          self._is_chance = True

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

  @cached_property
  def _returns(self):
    """Total reward for each player over the course of the game so far."""
    assert self._game_over
    assert self._auction_finished

    self._final_payments = np.zeros(self.num_players())
    returns = np.zeros_like(self._final_payments)
    final_prices = self.posted_prices[-1]

    # Calculate final payments (need to do this first so we can calculate spite bonuses)
    for player_id, bidder in enumerate(self.bidders):
      final_bid = bidder.processed_demand[-1]
      payment = final_bid @ final_prices
      self._final_payments[player_id] = payment

    # Calculate utilities
    for player_id, bidder in enumerate(self.bidders):
      final_bid = bidder.processed_demand[-1]
      value = bidder.bidder.value_for_package(final_bid)

      other_payments = sum(self._final_payments[i] for i in range(self.num_players()) if i != player_id)
      pricing_bonus = bidder.bidder.pricing_bonus * other_payments

      player_return = value - self._final_payments[player_id] + pricing_bonus
      if self.tiebreak_actions:
        player_return += self._action_rewards[player_id]

      returns[player_id] = bidder.bidder.get_utility(player_return)


    return returns

  def returns(self):
    return self._returns

  @cached_property
  def _as_string(self):
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

        result += f'Aggregate demand: {self.aggregate_demand[-1]}\n'

      if self._auction_finished:
        for player_id, player in enumerate(self.bidders):
          result += f'Final bids player {player_id}: {player.processed_demand[-1]}\n'

    return result
  
  def __str__(self):
    return self._as_string

  @cached_property
  def _chance_outcomes(self):
    """Returns the possible chance outcomes and their probabilities."""
    assert self._is_chance
    if len(self.bidders) < self.num_players(): # Chance node is assigning a type
      player_index = len(self.bidders)
      player_types = self.auction_params.player_types[player_index]
      probs = [t['prob'] for t in player_types]
    else: # Chance node is breaking a tie
      if self.auction_params.fold_randomness:
        probs = self.folded_chance_outcomes['lottery']
        # self.folded_chance_outcomes = None # TODO: I want to do this, but it actually makes it unsafe to call chance_outcomes multiple times in a row, so what can you do.... Definitely opens up a risk though
      else:
        chance_outcomes_required = math.factorial(len(self.processing_queue))
        if chance_outcomes_required > 1_000:
          return dict(upper=chance_outcomes_required - 1) # This breaks the openspiel API, but otherwise generating the list gets too big and blows up memory. See UBC Chance Event Sampler that knows how to decode this. Also, it's WAYYYY faster than actually making the lists
        probs = [1 / chance_outcomes_required] * chance_outcomes_required
    return list(enumerate(probs))

  def chance_outcomes(self):
    return self._chance_outcomes

  def _post_process(self):
    # Calculate aggregate demand
    aggregate_demand = np.zeros(self.auction_params.num_products, dtype=int)
    for bidder in self.bidders:
      bid = bidder.processed_demand[-1].copy()
      
      # Update activity history
      bidder.activity.append(bid @ self.auction_params.activity)
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
        raise ValueError(f"Auction went on too long {self.history()}")

    self._cur_player = 0
    self._is_chance = False

  def get_final_payments(self):
    assert self._game_over
    return self._final_payments

  @property
  def revenue(self):
    assert self._game_over
    return sum(self.get_final_payments())

  # METRICS
  @property
  def revenue_potential(self):
    if self.round == 1:
      return 0

    # TODO: Assumes undersell is turned on, or this is completely wrong
    guaranteed_to_sell = np.minimum(self.auction_params.licenses, self.aggregate_demand[-1])
    return guaranteed_to_sell @ self.posted_prices[-1]
    
  @property
  def revenue_potential_normalized(self):
    return self.revenue_potential / self.auction_params.max_total_spend

  @property
  def pricing_potential(self):
      if self.round == 1:
        return np.zeros(self.num_players())

      # Define pricing as opponent payments - oppnonent cost of bundle at starting prices
      increased_cost = np.zeros(self.num_players())
      allocation = [bidder.processed_demand[-1] for bidder in self.bidders]
      for player_id, bidder in enumerate(self.bidders):
        increased_cost[player_id] = (self.posted_prices[-1] - self.auction_params.opening_prices) @ allocation[player_id]
      pricing = np.zeros_like(increased_cost)
      for player_id in range(self.num_players()):
        for other_player_id in players_not_me(player_id,  self.num_players()):
            pricing[player_id] += increased_cost[other_player_id]
      return pricing 

  @property
  def pricing_potential_normalized(self):
    return self.pricing_potential / self.auction_params.max_opponent_spends

  @property
  def auction_length_potential(self):
    return self.round

  @property
  def auction_length_potential_normalized(self):
    return self.round / self.auction_params.max_round

  @property
  def welfare_potential(self):
    # Calculate on-demand welfare of current solution
    return self.get_welfare()

  @property
  def welfare_potential_normalized(self):
    # TODO: This is a real bound but too expensive to compute. No reason we couldn't use something much looser.
    # For example, you could find the type of each player that values the full bundle the most and then sum over the best-types for each player
    return self.get_welfare() / self.efficent_allocation()[0] 

  def get_allocation(self):
    assert self._game_over
    return [bidder.processed_demand[-1] for bidder in self.bidders]

  def get_welfare(self):
    welfare = 0
    allocation = [bidder.processed_demand[-1] for bidder in self.bidders]
    for bidder, package in zip(self.bidders, allocation):
      welfare += bidder.bidder.value_for_package(package)
    return float(welfare)

  def get_welfare_sparse_normalized(self):
    if self._game_over:
      max_welfare, alloc = self.efficent_allocation()
      return self.get_welfare() / max_welfare
    else:
      return 0

  def efficient_allocation(self):
    num_actions = len(self.auction_params.all_bids)
    num_players = self.num_players()
    n_vars = num_players * num_actions
    var_id_to_player_bundle = dict() # VarId -> (player, bundle)

    values = []
    q = 0
    for player, bidder in enumerate(self.bidders):
        v = bidder.bidder.get_values()
        values += v
        for val, bundle in zip(v, self.auction_params.all_bids):
          var_id_to_player_bundle[q] = (player, bundle)
          q += 1

    problem = LpProblem(f"EfficientAllocation", LpMaximize)
    bundle_variables = LpVariable.dicts("X", np.arange(n_vars), cat=LpBinary)

    # OBJECTIVE
    problem += lpDot(values, bundle_variables.values())

    # Constraint: Only 1 bundle per bidder
    for i in range(num_players):
        problem += lpSum(list(bundle_variables.values())[i * num_actions: (i+1) * num_actions]) == 1, f"1-per-bidder-{i}"
        
    # Constraint: Can't overallocate any items
    supply = self.auction_params.licenses
    for i in range(self.auction_params.num_products):
        product_amounts = [bundle[i] for (player, bundle) in var_id_to_player_bundle.values()]
        problem += lpDot(bundle_variables.values(), product_amounts) <= supply[i], f"supply-{i}"

    allocation = []
    try: 
        problem.writeLP(f'efficient_allocation_{random_string(10)}.lp')
        obj = pulp_solve(problem, save_if_failed=True)
        for var_id in range(n_vars):
            # print(var_id, bundle_variables[var_id], value(bundle_variables[var_id]), var_id_to_player_bundle[var_id])
            if value(bundle_variables[var_id]) > .99: # Rounding stupidness
                allocation.append(var_id_to_player_bundle[var_id][1])
    except ValueError as e:
        # if MIP is infeasible, drop out - TODO: Should this ever happen?
        feasible_result = False
        logging.warning(f'Failed to solve MIP; dropping out')

    return obj, allocation

  @cached_property
  def _my_hash(self):
    return hash(tuple(self.history())) # If you cache this, you need to remember to clear it in a clone

  def __hash__(self): # Two states that have the same history are the same
    return self._my_hash

  def regret_init(self, regret_init: str):
    """
    Return the initial regret for each action, based on the current state of the game.
    For CFR.
    """
    if regret_init == 'straightforward_clock' or regret_init == 'straightforward_sor':  
      player = self._cur_player
      prices = self.clock_prices if regret_init == 'straightforward_clock' else self.sor_prices
      profits = self.bidders[player].bidder.get_profits(prices[-1])
      legal_profits = [profits[i] for i in self.legal_actions()]

      # map max legal profit to 1 and min to 0
      # except if max = min: then return 0 for all
      if np.max(legal_profits) == np.min(legal_profits):
        rescaled_profits = np.zeros_like(legal_profits)
      else:
        rescaled_profits = (legal_profits - np.min(legal_profits)) / (np.max(legal_profits) - np.min(legal_profits))
      return rescaled_profits

    else:
      raise ValueError(f"Unknown regret initializer {regret_init}")


# Register the game with the OpenSpiel library
pyspiel.register_game(_GAME_TYPE, ClockAuctionGame)