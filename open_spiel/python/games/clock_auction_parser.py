import logging
import json
from open_spiel.python.examples.ubc_utils import num_to_letter
from open_spiel.python.games.clock_auction_base import AuctionParams, ActivityPolicy, UndersellPolicy, InformationPolicy, TiebreakingPolicy, ValueFormat, DEFAULT_AGENT_MEMORY, DEFAULT_MAX_ROUNDS, action_to_bundles
import numpy as np
from collections import defaultdict
import os
from open_spiel.python.games.clock_auction_bidders import LinearBidder, EnumeratedValueBidder, MarginalValueBidder

def parse_auction_params(file_name):
  if file_name.startswith('/'):
    full_path = file_name
  else:
    logging.info("Reading from env variable CLOCK_AUCTION_CONFIG_DIR. If it is not set, there will be trouble.")
    config_dir = os.environ.get('CLOCK_AUCTION_CONFIG_DIR')
    if config_dir is None:
      raise ValueError("CLOCK_AUCTION_CONFIG_DIR env variable is not set.")
    logging.info(f"CLOCK_AUCTION_CONFIG_DIR={config_dir}")
    full_path = f'{config_dir}/{file_name}'

  logging.info(f"Parsing configuration from {full_path}")
  with open(full_path, 'r') as f:
    game_params = json.load(f)

    players = game_params['players']

    opening_prices = game_params['opening_price']
    licenses = np.array(game_params['licenses'])
    if 'license_names' in game_params:
      license_names = game_params['license_names']
    else:
      license_names = [num_to_letter(i) for i in range(len(licenses))]

    num_products = len(licenses)

    if len(opening_prices) != num_products:
      raise ValueError("Number of opening prices must match number of products.")
    
    activity = game_params['activity']
    if len(activity) != num_products:
      raise ValueError("Number of activity must match number of products.")

    activity_policy = game_params.get('activity_policy', ActivityPolicy.ON)
    if isinstance(activity_policy, str):
      activity_policy = ActivityPolicy[activity_policy.upper()]

    information_policy = game_params.get('information_policy', InformationPolicy.SHOW_DEMAND)  
    if isinstance(information_policy, str):
      information_policy = InformationPolicy[information_policy.upper()]

    undersell_policy = game_params.get('undersell_policy', UndersellPolicy.UNDERSELL)
    if isinstance(undersell_policy, str):
      undersell_policy = UndersellPolicy[undersell_policy.upper()]

    tiebreaking_policy = game_params.get('tiebreaking_policy', TiebreakingPolicy.DROP_BY_PLAYER)
    if isinstance(tiebreaking_policy, str):
      tiebreaking_policy = TiebreakingPolicy[tiebreaking_policy.upper()]

    fold_randomness = game_params.get('fold_randomness', True)

    reveal_type_round = int(game_params.get('reveal_type_round', -1))
    grace_rounds = int(game_params.get('grace_rounds', 1))

    all_bids = action_to_bundles(licenses)
    bid_to_index = dict()
    for i, bid in enumerate(all_bids):
      bid_to_index[tuple(bid)] = i

    all_bids_activity = np.array([activity @ bid for bid in all_bids])

    types = defaultdict(list)

    for player_id, player in enumerate(players):
      player_types = player['type']
      for player_type in player_types:
        values = player_type['value']
        if np.array(values).ndim == 2:
          default_assumption = ValueFormat.MARGINAL
        else:
          default_assumption = ValueFormat.LINEAR
        value_format = player_type.get('value_format', default_assumption)
        if isinstance(value_format, str):
          value_format = ValueFormat[value_format.upper()]
        budget = player_type['budget']
        prob = player_type['prob']
        pricing_bonus = player_type.get('pricing_bonus', 0)
        drop_out_heuristic = player_type.get('drop_out_heuristic', True)
        name = player_type.get('name', None)
        utility_function_config = player_type.get('utility_function', {'name': 'quasilinear'})

        if value_format == ValueFormat.LINEAR:
          if len(values) != num_products:
            raise ValueError("Number of values must match number of products.")
          bidder = LinearBidder(values, budget, pricing_bonus, all_bids, drop_out_heuristic, utility_function_config)
        elif value_format == ValueFormat.FULL:
          if len(values) != len(all_bids):
            raise ValueError("Number of values must match number of bids.")
          bidder = EnumeratedValueBidder(values, budget, pricing_bonus, all_bids, drop_out_heuristic, utility_function_config, name, straightforward=player_type.get('straightforward', False))
        elif value_format == ValueFormat.MARGINAL:
          bidder = MarginalValueBidder(values, budget, pricing_bonus, all_bids, drop_out_heuristic, utility_function_config)
        else:
          raise ValueError("Unknown value format")
        
        types[player_id].append(dict(prob=prob, bidder=bidder))

  logging.info("Done config parsing")
  return AuctionParams(
      opening_prices=opening_prices,
      licenses=licenses,
      license_names=license_names,
      num_products=num_products,
      activity=activity,
      increment=game_params.get('increment', 0.1),
      max_round=game_params.get('max_rounds', DEFAULT_MAX_ROUNDS),
      player_types=types,
      all_bids=all_bids,
      bid_to_index=bid_to_index,
      all_bids_activity=all_bids_activity,
      activity_policy=activity_policy,
      grace_rounds=grace_rounds,
      undersell_policy=undersell_policy,
      information_policy=information_policy,
      tiebreaking_policy=tiebreaking_policy,
      reveal_type_round=reveal_type_round,
      fold_randomness=fold_randomness,
      agent_memory=game_params.get('agent_memory', DEFAULT_AGENT_MEMORY),
    )
