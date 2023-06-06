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
    num_bundles = len(auction_params.all_bids)

    # self.round_buffer = 100 if iig_obs_type.perfect_recall else auction_params.agent_memory
    self.round_buffer = auction_params.agent_memory # TODO: We are abusing the API here a bit. Only offers binary perfect recall or not, but we want to interpolate.

    """Initializes an empty observation tensor."""
    # Determine which observation pieces we want to include.
    # NOTE: It should be possible to use the params to exclude some of these if we want a smaller input to the NN (or to have the NN reassamble the tensor from specific pieces).

    pieces = [] # list of (feature name, ...)

    # Constants
    pieces.append(("my_activity", (num_bundles,)))
    pieces.append(("sor_exposure", (num_bundles,)))

    # Bundle quantities
    for p in self.auction_params.license_names:
      pieces.append((f"amount_{p}", (num_bundles,)))

    # 1 per bundle
    pieces.append(("values", (num_bundles,)))
    pieces.append(("activity", (num_bundles,)))
    pieces.append(("sor_profits", (num_bundles,)))
    pieces.append(("clock_profits", (num_bundles,)))
    pieces.append(("clock_bundle_prices", (num_bundles,)))

    pieces.append(("prev_processed", (num_bundles,))) # 1-hot
    pieces.append(("prev_demanded", (num_bundles,))) # 1-hot


    # memory x n_products
    pieces.append((f"submitted_demand_history", (num_bundles, num_products * self.auction_params.agent_memory)))
    pieces.append((f"processed_demand_history", (num_bundles, num_products * self.auction_params.agent_memory)))
    pieces.append((f"agg_demand_history", (num_bundles, num_products * self.auction_params.agent_memory)))
    pieces.append((f"sor_bundle_prices_history", (num_bundles, num_products * self.auction_params.agent_memory)))

    if self.auction_params.reveal_type_round != -1:
      for i in range(num_players):
        # Could be smaller if I excluded my own types, but this is simpler 
        pieces.append((f"revealed_values_{i}", (num_bundles,)))

    # Build the single flat tensor.
    total_size = sum(np.prod(shape) for name, shape in pieces)
    self.tensor = np.zeros(total_size, np.float32)

    # Build the named & reshaped views of the bits of the flat tensor.
    self.dict = {}
    index = 0
    for name, shape in pieces:
      size = np.prod(shape)
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""

    # BE VERY VERY CAREFUL NOT TO OVERRIDE THE DICT ENTRIES FROM POINTING INTO THE TENSOR
    # Very subtle e.g., self.dict["sor_profits"][:] = profits vs self.dict["sor_profits"] = profits
    self.tensor.fill(0)

    length = min(state.round, self.round_buffer)
    start_ind = max(1, state.round - self.round_buffer)
    end_ind = start_ind + length

    for i, p in enumerate(self.auction_params.license_names):
      # Bundle quantities
      if f"amount_{p}" in self.dict:
        amounts = self.auction_params.all_bids[:, i]
        if self.normalize:
          amounts = amounts / self.auction_params.all_bids.max()
        self.dict[f"amount_{p}"][:] = amounts

    if "activity" in self.dict:
      activities = self.auction_params.activity
      if self.normalize:
        activities = activities / self.auction_params.max_activity
      self.dict["activity"][:] = activities

    if "clock_bundle_prices" in self.dict:
      price = np.asarray(self.clock_prices[-1])
      prices = self.auction_params.all_bids @ price
      if self.normalize:
        prices = prices / self.auction_params.max_budget
      self.dict["clock_bundle_prices"][:] = prices

    # These require the bidder to be initialized
    if len(state.bidders) > player:
      if "values" in self.dict:
        values = state.bidders[player].bidder.get_values()
        if self.normalize:
          values = values / self.auction_params.max_budget
        self.dict["values"][:] = values
      if "my_activity" in self.dict:
        activity = state.bidders[player].activity
        if self.normalize:
          activity /= self.auction_params.max_activity
        self.dict["activity"][:] = activity
      if "sor_profits" in self.dict:
        price = np.array(state.sor_prices[-1])
        profits = state.bidders[player].bidder.get_profits(price)
        if self.normalize:
          profits = profits / self.auction_params.max_budget
        self.dict["sor_profits"][:] = profits
      if "clock_profits" in self.dict:
        price = np.array(state.clock_prices[-1])
        profits = state.bidders[player].bidder.get_profits(price)
        if self.normalize:
          profits = profits / self.auction_params.max_budget
        self.dict["clock_profits"][:] = profits

    if state.round > 1:     
      if "prev_processed" in self.dict and state.round > 1:
        bundle_index = np.asarray(self.auction_params.all_bids == state.bidders[player].processed_demand[-1]).nonzero()[0]
        self.dict["prev_processed"][bundle_index] = 1

      if "prev_demanded" in self.dict:
        bundle_index = np.asarray(self.auction_params.all_bids == state.bidders[player].submitted_demand[-1]).nonzero()[0]
        self.dict["prev_demanded"][bundle_index] = 1

      if "agg_demand_history" in self.dict:
        if self.auction_params.information_policy == InformationPolicy.SHOW_DEMAND:
          agg_demand_history = np.array(state.aggregate_demand)[start_ind:end_ind]
        elif self.auction_params.information_policy == InformationPolicy.HIDE_DEMAND:
          agg_demand_history = np.array(state.aggregate_demand)[start_ind:end_ind]
          over_demanded = agg_demand_history > self.auction_params.licenses
          at_demand = agg_demand_history == self.auction_params.licenses
          under_demanded = agg_demand_history < self.auction_params.licenses
          agg_demand_history[over_demanded] = InformationPolicyConstants.OVER_DEMAND
          agg_demand_history[at_demand] = InformationPolicyConstants.AT_SUPPLY
          agg_demand_history[under_demanded] = InformationPolicyConstants.UNDER_DEMAND
        else:
          raise ValueError("Unknown information policy")
        if self.normalize:
          agg_demand_history = agg_demand_history / self.auction_params.licenses
        self.dict["agg_demand_history"][:min(length, len(agg_demand_history))] = agg_demand_history.T
      if "submitted_demand_history" in self.dict:
        submitted_demand_history = np.array(state.bidders[player].submitted_demand)[start_ind:end_ind]
        if self.normalize:
          submitted_demand_history = submitted_demand_history / self.auction_params.licenses
        self.dict["submitted_demand_history"][:min(length, len(submitted_demand_history))] = submitted_demand_history.T
      if "processed_demand_history" in self.dict:
        processed_demand_history = np.array(state.bidders[player].processed_demand)[start_ind:end_ind]
        if self.normalize:
          processed_demand_history = processed_demand_history / self.auction_params.licenses
        self.dict["processed_demand_history"][:min(length, len(processed_demand_history))] = processed_demand_history.T
      if "sor_exposure" in self.dict:
        sor_exposure = state.bidders[player].processed_demand[-1] @ state.sor_prices[-1]
        if self.normalize:
          sor_exposure = sor_exposure / self.auction_params.max_budget # TODO: Why not use a player specific bound here?
        self.dict["sor_exposure"][:] = sor_exposure

    if "sor_bundle_prices_history" in self.dict:
      # TODO: No shot the indexing is correct here
      posted_prices = np.array(state.posted_prices)[start_ind:end_ind]
      all_prices = np.asarray(state.posted_prices) @ self.auction_params.all_bids # TODO: Guess
      if self.normalize:
        all_prices = all_prices / self.auction_params.max_budget
      self.dict["sor_bundle_prices_history"][:min(length, len(posted_prices))] = all_prices.T

    for i in range(len(self.auction_params.player_types)):
      if f"revealed_types_{i}" in self.dict and state.round >= self.auction_params.reveal_type_round:
        self.dict[f'revealed_types_{i}'][:] = state.bidders[i].get_values()

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