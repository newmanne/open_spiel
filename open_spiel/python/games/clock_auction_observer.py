from open_spiel.python.games.clock_auction_base import AuctionParams, InformationPolicyConstants, InformationPolicy
import numpy as np
from functools import lru_cache
from cachetools import LRUCache
from cachetools.keys import hashkey

class ClockAuctionObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""
  
  def __init__(self, iig_obs_type, params):
    auction_params = params.get('auction_params')
    self.normalize = params.get('normalize', True) # Raw features? Or normalized ones
    if not isinstance(auction_params, AuctionParams):
      raise ValueError("params must be an AuctionParams object")
    self.auction_params = auction_params
    self.tensor_cache = LRUCache(maxsize=50_000)

    num_players = len(auction_params.player_types)
    num_products = auction_params.num_products
    num_bundles = len(auction_params.all_bids)

    # self.round_buffer = 100 if iig_obs_type.perfect_recall else auction_params.agent_memory
    self.round_buffer = auction_params.agent_memory # TODO: We are abusing the API here a bit. Only offers binary perfect recall or not, but we want to interpolate.

    """Initializes an empty observation tensor."""
    # Determine which observation pieces we want to include.
    # NOTE: It should be possible to use the params to exclude some of these if we want a smaller input to the NN (or to have the NN reassamble the tensor from specific pieces).

    self.pieces = [] # list of (feature name, shape, lambda state, player_id: feature_value, normalizer) 

    # Constants (but vary each round)
    # TODO: If you reinitialize this, the shape is "wrong" 
    # self.pieces.append(("round", (1,), lambda state, **kwargs: state.round, self.auction_params.max_round)) # TODO: This is controversial for ML based methods because they can just condition on it and it might hurt generalization
    self.pieces.append(("my_activity", (num_bundles,), lambda bidder, **kwargs: bidder.activity if bidder is not None else self.auction_params.max_activity, self.auction_params.max_activity))
    self.pieces.append(("sor_exposure", (num_bundles,), lambda state, bidder, **kwargs: bidder.processed_demand[-1] @ state.sor_prices[-1] if bidder is not None else 0, self.auction_params.max_budget))

    ######## 1 per bundle quantities #######
    ### Constant throughout the auction
    self.pieces.append(("amount", (num_products, num_bundles), self.auction_params.all_bids.T, self.auction_params.licenses[:, None]))
    self.pieces.append(("activity", (num_bundles,), self.auction_params.all_bids_activity, self.auction_params.max_activity))

    # This is constant (but only a per-player basis)
    self.pieces.append(("values", (num_bundles,), lambda bidder, **kwargs: bidder.bidder.get_values() if bidder is not None else 0, self.auction_params.max_budget))

    # These are so easily derivable - do we really need them? I'm commenting out for now
    # self.pieces.append(("sor_profits", (num_bundles,), lambda state, bidder, **kwargs: bidder.bidder.get_profits(state.sor_prices[-1]) if bidder is not None else 0, self.auction_params.max_budget))
    # self.pieces.append(("clock_profits", (num_bundles,), lambda state, bidder, **kwargs: bidder.bidder.get_profits(state.clock_prices[-1]) if bidder is not None else 0, self.auction_params.max_budget))

    # This is easily derviable from the SoR prices last cell - just mulitiply by increment
    # self.pieces.append(("clock_bundle_prices", (num_bundles,), lambda state, **kwargs: self.auction_params.all_bids @ state.clock_prices[-1], self.auction_params.max_budget))

    # TODO: Are these slow? You could easily store the raw in the state directly to stop the index searching

    def prev(prev_getter):
      def prev_getter_wrapper(state, bidder, **kwargs):
        retval = np.zeros(num_bundles)
        if bidder is not None and state.round > 1:
          retval[state.auction_params.bid_to_index[prev_getter(state, bidder)]] = 1
        return retval
      return prev_getter_wrapper

    self.pieces.append(("prev_processed", (num_bundles,), prev(lambda state, bidder: tuple(bidder.processed_demand[-1])), 1)) # 1-hot
    self.pieces.append(("prev_demanded", (num_bundles,), prev(lambda state, bidder: tuple(bidder.submitted_demand[-1])), 1)) # 1-hot

    # memory x n_products
    def get_history_for_variable(history_getter):
      def get_history(state, bidder, **kwargs):
        retval = np.zeros((self.auction_params.agent_memory, num_bundles))
        if state.round > 1:
          history = np.array(history_getter(bidder))[-self.auction_params.agent_memory:] # (memory, n_products)
          for i, bid in enumerate(reversed(history)):
            retval[i, state.auction_params.bid_to_index[tuple(bid)]] = 1
          # Note, if you are going to change this, remember that the indexing is complicated. You have to make sure that a column has the constant meaning of nth round ago, so values have to "slide" in time, with the newest entry in the spot that previous newest was before. Especially tricky w/ fewer elements than length
        return retval
      return get_history
    
    def get_agg_demand_history(state, **kwargs):
      retval = np.zeros((num_bundles, self.auction_params.agent_memory, num_products))
      agg_demand_history = np.array(state.aggregate_demand)[-self.auction_params.agent_memory:] # (memory, n_products)
      if self.auction_params.information_policy == InformationPolicy.SHOW_DEMAND:
        pass
      elif self.auction_params.information_policy == InformationPolicy.HIDE_DEMAND:
        over_demanded = agg_demand_history > self.auction_params.licenses
        at_demand = agg_demand_history == self.auction_params.licenses
        under_demanded = agg_demand_history < self.auction_params.licenses
        agg_demand_history[over_demanded] = InformationPolicyConstants.OVER_DEMAND
        agg_demand_history[at_demand] = InformationPolicyConstants.AT_SUPPLY
        agg_demand_history[under_demanded] = InformationPolicyConstants.UNDER_DEMAND
      else:
        raise ValueError("Unknown information policy")
      retval[:, -len(agg_demand_history):] = agg_demand_history
      return retval.transpose(1, 2, 0)

    def get_sor_bundle_prices_history(state, **kwargs):
      # Note, if you are going to change this, remember that the indexing is complicated. You have to make sure that a column has the constant meaning of nth round ago, so values have to "slide" in time, with the newest entry in the spot that previous newest was before. Especially tricky w/ fewer elements than length
      retval = np.zeros((num_bundles, self.auction_params.agent_memory))
      sor_prices = np.asarray(state.sor_prices)[-self.auction_params.agent_memory:] # (memory, num_products)
      retval[:, -len(sor_prices):] = (sor_prices @ self.auction_params.all_bids.T).T # (memory, num_bundles)
      return retval.T

    # 1-hot encoding x memory
    self.pieces.append((f"submitted_demand_history", (self.auction_params.agent_memory, num_bundles), get_history_for_variable(lambda bidder: bidder.submitted_demand[1:]), 1)) # 1 b/c there's a placeholder 0 demand in the first round for indexing
    # 1-hot encoding x memory
    self.pieces.append((f"processed_demand_history", (self.auction_params.agent_memory, num_bundles), get_history_for_variable(lambda bidder: bidder.processed_demand[1:]), 1))

    self.pieces.append((f"agg_demand_history", (self.auction_params.agent_memory, num_products, num_bundles), get_agg_demand_history, self.auction_params.licenses[None, :, None] * num_players)) 
    self.pieces.append((f"sor_bundle_prices_history", (self.auction_params.agent_memory, num_bundles), get_sor_bundle_prices_history, self.auction_params.max_budget))

    if self.auction_params.reveal_type_round != -1:
      for i in range(num_players):
        # Could be smaller if I excluded my own types, but this is simpler 
        self.pieces.append((f"revealed_values_{i}", (num_bundles), lambda state, **kwargs: state.bidders[i].bidder.get_values() if state.round >= self.auction_params.reveal_type_round else 0, self.auction_params.max_budget))

    # Build the single flat tensor.
    total_size = sum(np.prod(shape) for name, shape, getter, normalizer in self.pieces)

    num_features = total_size // num_bundles
    if total_size % num_features != 0:
      raise ValueError(f"Miscounted features! Total size: {total_size}, num_bundles {num_bundles}, features {num_features}, actual {total_size / num_bundles}")
    self.observation_shape = (num_features, num_bundles)
    self.tensor = np.zeros(total_size, np.float32)

    # Build the named & reshaped views of the bits of the flat tensor.
    self.dict = {}
    index = 0
    for name, shape, _, _ in self.pieces:
      size = np.prod(shape)
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size


  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""

    # BE VERY VERY CAREFUL NOT TO OVERRIDE THE DICT ENTRIES FROM POINTING INTO THE TENSOR
    # Very subtle e.g., self.dict["sor_profits"][:] = profits vs self.dict["sor_profits"] = profits
    my_hash_key = hashkey(state, player, self.normalize)
    res = self.tensor_cache.get(my_hash_key)
    if res:
      self.tensor[:] = res
      return

    self.tensor.fill(0)

    getter_kwargs = {
      "player": player,
      "state": state,
      "bidder": state.bidders[player] if len(state.bidders) > player else None, 
    }

    if self.normalize:
      for key, shape, func_or_value, normalizer in self.pieces:
        if callable(func_or_value):
          value = func_or_value(**getter_kwargs)
          # print(key, value, normalizer)
          self.dict[key][:] = value / normalizer
        else:
          # print(key, func_or_value, normalizer)
          self.dict[key][:] = func_or_value / normalizer
    else:
      for key, shape, func_or_value, normalizer in self.pieces:
        if callable(func_or_value):
          value = func_or_value(**getter_kwargs)
          self.dict[key][:] = value
        else:
          self.dict[key][:] = func_or_value

    if np.isnan(self.tensor).any():
      raise ValueError(f"NaN in observation {self.dict}")

    if self.normalize and (np.abs(self.tensor) > 10).any():
      raise ValueError(f"Observation {self.dict} has values > 1")
    
    self.tensor_cache[my_hash_key] = tuple(self.tensor)

  @lru_cache(maxsize=None)
  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []
    pieces.append(f"p{player}t{state.bidders[player].type_index}")
    pieces.append(f"r{state.round}")
    pieces.append(f"posted{np.round(np.array(state.posted_prices[-self.auction_params.agent_memory:]), 2).tolist()}")
    # Dumb temporary hack to be backward compatible with old policies. I'm so sorry for this.
    if self.auction_params.skip_single_chance_nodes:
      pieces.append(f"sub{np.array(state.bidders[player].submitted_demand[-self.auction_params.agent_memory:]).tolist()}") # Need this for perfect recall
    if "activity" in self.dict:
      pieces.append(f"a{state.bidders[player].activity}")
    if "agg_demand_history" in self.dict:
        agg = np.array(state.aggregate_demand[-self.auction_params.agent_memory:])
        if self.auction_params.information_policy == InformationPolicy.HIDE_DEMAND:
          over_demanded = agg > self.auction_params.licenses
          at_demand = agg == self.auction_params.licenses
          under_demanded = agg < self.auction_params.licenses
          agg[over_demanded] = InformationPolicyConstants.OVER_DEMAND
          agg[at_demand] = InformationPolicyConstants.AT_SUPPLY
          agg[under_demanded] = InformationPolicyConstants.UNDER_DEMAND
          pieces.append(f"agg{agg.tolist()}")
        else:
          pieces.append(f"agg{agg.tolist()}")
    # if "submitted_demand_history" in self.dict: # TODO: If you want TRUE perfect recall, we have to uncomment this
    if "processed_demand_history" in self.dict:
      pieces.append(f"proc{np.array(state.bidders[player].processed_demand[-self.auction_params.agent_memory:]).tolist()}")
    # if "clock_prices" in self.dict: # Nice for debugging, but adds no value (dervied from posted_prices)
      # pieces.append(f"clock{state.clock_prices[-1]}")
    # if "sor_exposure" in self.dict and state.round > 1: # Nice for debugging, but adds no new info (dervied from clock prices)
      # pieces.append(f"sor_exposure{state.bidders[player].processed_demand[-1] @ state.sor_prices[-1]}")
    # if "price_increments" in self.dict and state.round > 1: # Nice for debugging, but adds no new info (Derived from clcok prices)
      # pieces.append(f"increments{state.price_increments}")
    return " ".join(str(p) for p in pieces)
