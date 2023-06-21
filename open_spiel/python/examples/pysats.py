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

map_generators = {
    '3Province': map_one,
    'BC': map_bc,
    'OntarioQuebec': map_ontario_quebec,
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
    return 1 if n == 1 else 1.2

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

    def __init__(self, market_share_sampler, value_per_subscriber_sampler, map=None, all_bids=None, scale=1, name=None) -> None:
        self.region_to_params = dict()
        self.regions = list(map.nodes)
        self.all_bids = all_bids
        self.map = map
        self.scale = scale
        self.name = name
        for region_index, region in enumerate(self.regions):
            market_share = market_share_sampler.sample()
            licenses_in_region = self.all_bids[-1][region_index]
            max_capacity = licenses_in_region * synergy(licenses_in_region)
            region_params = BidderRegionParams(
                z_lower=(max(0, market_share - 0.3) / (market_share * region.population)) * max_capacity, 
                z_upper=(min(1, market_share + 0.3) / (market_share * region.population)) * max_capacity, 
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
            if gamma < 0:
                raise ValueError(f'Negative gamma: {gamma} for package {package} for bidder {self}')
            sv = self.independent_region_value(region_params, p)
            independent_region_value = region_params.market_share * region_params.region.population * sv
            v += independent_region_value * gamma
        return max(0, int(v / self.scale))

    def output_clock_auction(self):
        # Return my values in clock auction format as list comprehension
        return [self.value(bid) for bid in self.all_bids]

    def budget(self):
        return max(self.output_clock_auction())

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
        return self.gamma_factor**(len(nx.shortest_path(self.map, self.hq, region_params.region)) - 1)

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
        map = map_generators[config['map']]()
    except KeyError:
        raise ValueError(f"Unknown map {config['map']} not found")
    
    signal_amount = config.get('signal', 0)
    if signal_amount:
        map = add_signal(map)
        auction_params['licenses'].append(signal_amount)
        auction_params['activity'].append(1)
        auction_params['opening_price'].append(10)

    scale = config['scale']
    all_bids = generate_bids(auction_params['licenses'])

    shared_bidder_args = {
        'map': map,
        'all_bids': all_bids,
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
            'k_max': 2, 
            'b': {
              'lower': 0.1,
              'upper': 0.3,  
            } 
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
            'gamma_factor': 0.42,
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
            }
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
                # print(bidder[-1].region_to_params)
                # for region_params in bidder[-1].region_to_params.values():
                #     print(region_params.region.name)
                #     print(region_params.market_share * region_params.max_subscriber_value)
                # print()

        bidders.append(bidder)

    auction_params['license_names'] = [node.name for node in map.nodes]

    auction_params['players'] = []
    for bidder_id, bidder in enumerate(bidders):
        player = {}
        type_list = []
        for bidder_type_index, bidder_type in enumerate(bidder):
            type_object = {
                'value': bidder_type.output_clock_auction(),
                'value_format': 'full',
                'budget': bidder_type.budget(),
                'prob': 1. / len(bidder),
                'name': bidder_type.name,
                # 'drop_out_heuristic': config['bidders'][bidder_id].get('drop_out_heuristic', True)
            }
            if 'prob' in config['bidders'][bidder_id]['types'][bidder_type_index]:
                type_object['prob'] = config['bidders'][bidder_id]['types'][bidder_type_index]['prob']
            if 'straightforward' in config['bidders'][bidder_id]['types'][bidder_type_index]:
                type_object['straightforward'] = config['bidders'][bidder_id]['types'][bidder_type_index]['straightforward']

            type_list.append(type_object)
        player['type'] = type_list    
        auction_params['players'].append(player)

    # print(auction_params)

    # Write to disk
    with open(output_file, 'w') as fh:
        json.dump(auction_params, fh, cls=NpEncoder)
    
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

