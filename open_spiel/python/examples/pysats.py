from dataclasses import dataclass
import numpy as np
import pandas as pd
import networkx as nx
import yaml
import argparse
import logging
import itertools
import copy
import json
from pathlib import Path

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class FloatSampler:
    def __init__(self, rng, lower=0, upper=1):
        self.lower = lower
        self.upper = upper
        self._rng = rng

    def sample(self):
        return self._rng.uniform(self.lower, self.upper)

@dataclass(frozen=True)
class Region:
    name: str
    population: float

BRITISH_COLUMBIA = Region(population=5e6, name='British Columbia')
ONTARIO = Region(population=14e6, name='Ontario')
QUEBEC = Region(population=8e6, name='Quebec')
SIGNAL = Region(population=10, name='Signal')

def map_one():
    G = nx.Graph()
    G.add_nodes_from([BRITISH_COLUMBIA, ONTARIO, QUEBEC])
    G.add_edges_from([
        (BRITISH_COLUMBIA, ONTARIO),
        (ONTARIO, QUEBEC),
    ])
    return G

def map_bc():
    G = nx.Graph()
    G.add_nodes_from([BRITISH_COLUMBIA])
    return G

def map_ontario_quebec():
    G = nx.Graph()
    G.add_nodes_from([ONTARIO, QUEBEC])
    G.add_edges_from([
        (ONTARIO, QUEBEC),
    ])
    return G

# map_name -> (graph generator, bid to quantity matrix)
# default bid to quantity matrix (None) is identity
map_generators = {
    '3Province': (map_one, None),
    'BC': (map_bc, None),
    # BC only, with a second encumbered license with 60% coverage
    'BCEncumbered': (map_bc, np.array([[1, 0.6]])),
    # ditto, but with 40% coverage
    'BCEncumbered40': (map_bc, np.array([[1, 0.4]])),
    
    'OntarioQuebec': (map_ontario_quebec, None),
    # Ontario+Quebec map, but Quebec also has an encumbered license with 70% coverage (product #3)
    'OntarioQuebecEncumbered': (map_ontario_quebec, np.array([[1, 0, 0], [0, 1, 0.7]])),
    # Ontario+Quebec map, plus a signalling-only product with no value (product #3)
    'OntarioQuebecSignal': (map_ontario_quebec, np.array([[1, 0, 0], [0, 1, 0]])),
}

def add_signal(G):
    G.add_node(SIGNAL)
    return G

def generate_bids(licenses): # Important this is the same order as clock_auction.py
    bids = []
    for n in licenses:
        b = []
        for i in range(n + 1):
            b.append(i)
        bids.append(b)
    return np.array(list(itertools.product(*bids)))

def synergy(n):
    ''' Returns a synergy value for n licenses'''
    return 1 if n <= 1 else 1.2

# Concrete instantiations of params
@dataclass 
class BidderRegionParams:
    z_lower: float
    z_upper: float
    max_subscriber_value: float
    market_share: float
    region: Region
    region_index: int
    max_capacity: float

def subscriber_value(x, max_capacity, z_lower, z_upper, population, market_share, max_value_per_subscriber):
    epsilon = 1e-6
    control_one = max(z_lower * population * market_share, epsilon * max_capacity)
    control_two = min(z_upper * population * market_share, (1 - epsilon) * max_capacity)

    # print(x, max_capacity, x / max_capacity)

    control_points = [
        (0, 0),
        (control_one, 0.27 * max_value_per_subscriber), 
        (control_two, 0.73 * max_value_per_subscriber),
        (max_capacity, max_value_per_subscriber)
    ]
    xp, fp = zip(*control_points)
    sv = np.interp(x, xp, fp, 0, max_value_per_subscriber)

    # print("Market share", market_share, "SV", sv, "Max value per sub", max_value_per_subscriber, "MS*MVS", market_share * max_value_per_subscriber)


    if sv < 0 or sv > max_value_per_subscriber:
        raise ValueError(f'Subscriber value {sv} outside of range [0, {max_value_per_subscriber}]')
    
    return sv

class Bidder:

    def __init__(self, market_share_sampler, value_per_subscriber_sampler, auction_map=None, licenses_in_region=None, scale=1, name=None, z_spread=0.1, allow_float_values=False) -> None:
        """
        Generic bidder class.

        Args:
        - market_share_sampler: float sampler for market share
        - value_per_subscriber_sampler: float sampler for value per subscriber
        - auction_map: map from one of the map generators
        - licenses_in_region: list of number of licenses in each region
        - scale: scale factor applied to all values
        - name: name of bidder
        - z_spread: set bidder's z_lower and z_upper to market_share +/- z_spread
        - allow_float_values: if False, truncate values to integers
        """
        self.region_to_params = dict()
        self.regions = list(auction_map.nodes)
        self.licenses_in_region = licenses_in_region
        self.auction_map = auction_map
        self.scale = scale
        self.name = name
        self.z_spread = z_spread
        self.allow_float_values = allow_float_values
        for region_index, region in enumerate(self.regions):
            market_share = market_share_sampler.sample()
            max_capacity = self.licenses_in_region[region_index] * synergy(self.licenses_in_region[region_index])
            region_params = BidderRegionParams(
                z_lower=(max(0, market_share - z_spread) / (market_share * region.population)) * max_capacity, 
                z_upper=(min(1, market_share + z_spread) / (market_share * region.population)) * max_capacity, 
                max_subscriber_value=value_per_subscriber_sampler.sample(),
                market_share=market_share,
                region=region,
                region_index=region_index,
                max_capacity=max_capacity
            )
            self.region_to_params[region] = region_params

    def independent_region_value(self, region_params, n_licenses):
        x = synergy(n_licenses) * n_licenses
        return subscriber_value(x, region_params.max_capacity, region_params.z_lower, region_params.z_upper, region_params.region.population, region_params.market_share, region_params.max_subscriber_value)

    def value(self, package):
        v = 0
        for i, p in enumerate(package):
            region_params = self.region_to_params[self.regions[i]]
            if region_params.region == SIGNAL:
                continue
            gamma = self.gamma(region_params, package)
            if gamma < 0 and sum(package) > 0:
                raise ValueError(f'Negative gamma: {gamma} for package {package} for bidder {self}')
            sv = self.independent_region_value(region_params, p)
            independent_region_value = region_params.market_share * region_params.region.population * sv
            v += independent_region_value * gamma

        v_scaled = v / self.scale
        if not self.allow_float_values:
            v_scaled = int(v_scaled)
        return max(0, v_scaled)

    def output_clock_auction(self, all_bids):
        # Return my values in clock auction format as list comprehension
        return [self.value(bid) for bid in all_bids]

    def budget(self, all_bids):
        return max(self.output_clock_auction(all_bids))

class LocalBidder(Bidder):

    def __init__(self, market_share_sampler, value_per_subscriber_sampler, local_regions=None, **kwargs) -> None:
        super().__init__(market_share_sampler, value_per_subscriber_sampler, **kwargs)
        self.local_regions = local_regions

    def gamma(self, region_params, package):
        return self.local_regions[region_params.region_index]

class RegionalBidder(Bidder):

    def __init__(self, market_share_sampler, value_per_subscriber_sampler, hq=None, gamma_factor=0.42, **kwargs) -> None:
        super().__init__(market_share_sampler, value_per_subscriber_sampler, **kwargs)
        self.hq = self.regions[hq]
        self.gamma_factor = gamma_factor

    def gamma(self, region_params, package):
        return self.gamma_factor**(len(nx.shortest_path(self.auction_map, self.hq, region_params.region)) - 1)

class NationalBidder(Bidder):
    
    def __init__(self, market_share_sampler, value_per_subscriber_sampler, k_max=4, b_sampler=None, **kwargs) -> None:
        super().__init__(market_share_sampler, value_per_subscriber_sampler, **kwargs)
        self.k_max = k_max
        # TODO: Should sanitize b_range such that gamma cannot go negative
        self.b = b_sampler.sample()

    def gamma(self, region_params, package):
        # TODO: Gamma should not apply to signal product
        k = sum(package == 0)
        return 1 - (min(k, self.k_max) * self.b)**2

    def __str__(self):
        return f'National bidder with b={self.b}'

class ExplicitBidder:
    """
    TODO: this is probably broken after allowing encumbered licenses.
    """

    def __init__(self, name=None, values=None, all_bids=None, budget=None, **kwargs) -> None:
        self.name = name
        if values is None:
            raise ValueError("Values cannot be None!")
        if len(values) != len(all_bids):
            raise ValueError(f"Expected size {len(all_bids)} for values and received size {len(values)}")
        self.values = values
        self.budget_amount = budget

    def __str__(self) -> str:
        return "#$*!@ (explicit)"
    
    def output_clock_auction(self):
        return self.values
    
    def budget(self):
        return max(self.values) if self.budget_amount is None else self.budget_amount
    

def run_sats(config, output_file, seed=1234):
    # Set seeds
    # TODO: Use your own random generator
    config = copy.deepcopy(config)

    rng = np.random.default_rng(seed=seed)
    auction_params = config['auction_params']

    # Generate map
    try:
        map_generator, bid_to_quantity_matrix = map_generators[config['map']]
        auction_map = map_generator()
    except KeyError:
        raise ValueError(f"Unknown map {config['map']} not found")
    
    signal_amount = config.get('signal', 0)
    if signal_amount:
        auction_map = add_signal(auction_map)
        auction_params['licenses'].append(signal_amount)
        auction_params['activity'].append(1)
        auction_params['opening_price'].append(10)

    scale = config['scale']
    all_bids = generate_bids(auction_params['licenses'])
    if bid_to_quantity_matrix is None:
        bid_to_quantity_matrix = np.eye(len(auction_params['licenses']))
    bid_quantities = all_bids @ bid_to_quantity_matrix.T
    licenses_in_region = bid_quantities[-1]



    risk_averse = config.get('risk_averse', False) # Right now binary, could imagine doing this better
    pricing_bonus = config.get('pricing_bonus')

    shared_bidder_args = {
        'auction_map': auction_map,
        'licenses_in_region': licenses_in_region,
        'scale': scale,
    }
    bidder_generators = {
        'local': LocalBidder,
        'regional': RegionalBidder,
        'national': NationalBidder,
        'explicit': ExplicitBidder
    }
    default_bidder_args = {
        'national': {
            'name': 'national',
            'market_share': {
                'lower': 0.08,
                'upper': 0.22,
            },
            'value_per_subscriber': {
                'lower': 700,
                'upper': 1200,
            },
            'z_spread': 0.1,
            'k_max': 2, 
            'b': {
              'lower': 0.1,
              'upper': 0.3,  
            },
            'allow_float_values': False,
        },
        'regional': {
            'name': 'regional',
            'market_share': {
                'lower': 0.04,
                'upper': 0.1,
            },
            'value_per_subscriber': {
                'lower': 500,
                'upper': 840,
            },
            'z_spread': 0.1,
            'gamma_factor': 0.42,
            'allow_float_values': False,
        }, 
        'local': {
            'name': 'local',
            'market_share': {
                'lower': 0.05,
                'upper': 0.12,
            },
            'value_per_subscriber': {
                'lower': 60,
                'upper': 100,
            },
            'z_spread': 0.1,
            'allow_float_values': False,
        },
        'explicit': {
            'name': 'explicit',
            'values': [],
        }
    }

    # Read bidders # List of lists (bidder, type)
    bidders = []

    for bidder_config in config['bidders']:
        bidder = []
        for cfg in bidder_config['types']:
            # Get type of bidder
            type_of_bidder = cfg.pop('type').lower()

            # Merge config with defaults for specific bidder
            shared_args = copy.deepcopy(shared_bidder_args)
            bidder_args = default_bidder_args[type_of_bidder]
            kwargs = {**shared_args, **bidder_args, **cfg}

            if type_of_bidder != 'explicit':
                # Build samplers
                kwargs['market_share_sampler'] = FloatSampler(rng, **(kwargs.pop('market_share')))
                kwargs['value_per_subscriber_sampler'] = FloatSampler(rng, **(kwargs.pop('value_per_subscriber')))

            # Special args for specific bidders
            if type_of_bidder == 'national':
                kwargs['b_sampler'] = FloatSampler(rng, **(kwargs.pop('b')))

            # Build bidders
            bidder_func = bidder_generators[type_of_bidder]
            bidder_count = kwargs.pop('count', 1)
            for count in range(bidder_count):
                kwargs_to_use = kwargs.copy()
                if bidder_count > 1:
                    kwargs_to_use['name'] = f"{kwargs_to_use['name']}_{count}"
                bidder.append(bidder_func(**kwargs_to_use))

        bidders.append(bidder)

    auction_params['license_names'] = [node.name for node in auction_map.nodes]

    auction_params['players'] = []
    for bidder_id, bidder in enumerate(bidders):
        player = {}
        type_list = []
        for bidder_type_index, bidder_type in enumerate(bidder):
            type_object = {
                'value': bidder_type.output_clock_auction(bid_quantities),
                'value_format': 'full',
                'budget': bidder_type.budget(bid_quantities),
                'prob': 1. / len(bidder),
                'name': bidder_type.name,
                'action_prefix': [],
                # 'drop_out_heuristic': config['bidders'][bidder_id].get('drop_out_heuristic', True)
            }
            bidder_config = config['bidders'][bidder_id]
            cfg = bidder_config['types'][bidder_type_index]
            if risk_averse: # Should come in the config at the bidder level like below, but a bit annoying b/c better access to values here
                type_object['utility_function'] = {'name': 'risk_averse', 'alpha': 1 / type_object['budget']}
            if pricing_bonus is not None:
                type_object['pricing_bonus'] = pricing_bonus
            if 'pricing_bonus' in cfg:
                type_object['pricing_bonus'] = cfg['pricing_bonus']
            if 'prob' in cfg:
                type_object['prob'] = cfg['prob']
            if 'straightforward' in cfg:
                type_object['straightforward'] = cfg['straightforward']
            if 'action_prefix' in bidder_config:
                type_object['action_prefix'] = bidder_config['action_prefix']

            type_list.append(type_object)
        player['type'] = type_list    
        auction_params['players'].append(player)

    # Write to disk
    with open(output_file, 'w') as fh:
        json.dump(auction_params, fh, cls=NpEncoder, indent=2)
    
    return auction_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output_file', type=str, default=None)
    args = parser.parse_args()


    if args.output_file is None:
        args.output_file = Path(args.config_file).stem + '.json'

    logging.info(f"Reading config from {args.config_file}")
    with open(args.config_file, 'rb') as fh: 
        config = yaml.load(fh, Loader=yaml.FullLoader)

    run_sats(config, args.output_file, seed=args.seed)

