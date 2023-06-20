import pyspiel
import open_spiel.python.games
from open_spiel.python.examples.ppo_utils import EnvParams
from open_spiel.python.env_decorator import *

# from auctions.webutils import *
import os

import sys
import pandas as pd
import os
import seaborn as sns

from tqdm import tqdm
from statsmodels.distributions.empirical_distribution import ECDF
from collections import defaultdict
import pickle
import numpy as np 
import pandas as pd
import tempfile
import open_spiel.python.games
from open_spiel.python.examples.ubc_utils import *

from pathlib import Path

from open_spiel.python.examples.ubc_cma import *
from open_spiel.python.examples.env_and_policy import *
import copy

import sys
sys.path.append('/apps/sats/python')
from pysats import map_generators, run_sats

from open_spiel.python.algorithms import get_all_states_with_policy
import time
import tempfile
import copy
import pickle

from open_spiel.python.algorithms.outcome_sampling_mccfr import OutcomeSamplingSolver
from open_spiel.python.algorithms.external_sampling_mccfr import ExternalSamplingSolver
import signal
import argparse
# from pympler.tracker import SummaryTracker

from distutils import util
PREFIX = 'may30'
CONFIG_DIR = os.environ['CLOCK_AUCTION_CONFIG_DIR']
PYSATS = '/apps/sats/python'

def has_non_zero_allocations(df):
    # Returns true if for each player, there exists a type combo of opponents such that it is allocated at least one item (no one should ever have 0 hope)
    df['players'] = [list(range(len(df.iloc[0]['combo']))) for _ in range(len(df))]
    df_exploded = df.explode(['allocation', 'combo', 'players'])
    df_exploded['good'] = df_exploded['allocation'].apply(np.sum) > 0
    return (df_exploded.groupby(['players', 'combo'])['good'].sum() > 0).all()

# def has_something_In_every_alloc(df):
#     #         #checking if every player gets something against every type alloc
# #         for alloc in df['allocation'].values: # Tuple where each entry is my alloc
# #             for p in alloc:
# #                 if sum(p) == 0:
# #                     return False

def run_iter(solver):
    solver.iteration()

class SignalTimeout(ValueError):
    pass


def is_mccfr_iter_slow(game, external=True, num_iters=5000, cutoff=20):
    # Note: 277 iters/s would be 1M iterations in an hour
    # Note cache needs warming up so times are a bit deceiving, especially if cutoff is too short
    
    # 25 iters/s means 1M in 11 hours

    print("Starting test for slowness")
    solver = ExternalSamplingSolver(game) if external else OutcomeSamplingSolver(game)
    start = time.time()
    MIN_TIME = 1
    
    # Alternative possibility to signal handler is to instrument a timeout WITHIN MCCFR itself, but that's annoying (and would have to be done with every variant)
    
    def signal_handler(*args):
        raise SignalTimeout()
    
    failed = False
    for num_iters_done in range(num_iters):
        
        if num_iters_done == 0:
            # Use signal to make sure the first iteration isn't overly long
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(max(MIN_TIME, cutoff // 20))
            try:
                run_iter(solver)
            except SignalTimeout as ex:
                failed = True
            finally:
                signal.alarm(0) # Clear alarm
        else:
            solver.iteration()
        
        elapsed = time.time() - start
        if failed or elapsed > cutoff:
            failed = True # You couldn't get through enough iterations before the cutoff
            break
    
    if num_iters_done > 2:
        print(f"Speed of {(num_iters_done + 1)/elapsed:.2f} iters/s")
    return failed


def all_products_overdemanded(game):
    # For each product, there should exist at least one type combo that overdemands it at the opening prices 
    good_products = [False] * game.auction_params.num_products
    for combo in type_combos(game):
        demand = np.zeros(game.auction_params.num_products)
        for bidder_type in combo:
            bidder = bidder_type['bidder']
            profits = bidder.get_profits(game.auction_params.opening_prices)
            package = bidder.all_bids[profits.argmax()]
            demand += package
        is_good_for_combo = demand > game.auction_params.licenses
        good_products |= is_good_for_combo
    return good_products.all()


def test_config_is_wieldy(config, max_game_tree_size=None, external=True):    
    # This function is really a combination of interesting and wieldy


    start = time.time()
    retval = dict(failed=False)
    # Make the file in a tmp place
    with tempfile.NamedTemporaryFile(mode='w+') as fp:
        sats_config = run_sats(config, fp.name, seed=config['sats_seed'])
        retval['sats_config'] = sats_config

        try:
            game = pyspiel.load_game('python_clock_auction', dict(filename=fp.name))
            df, combo_to_score, scorer = efficient_allocation(game, factor_in_opening_prices=True, verbose=False)

            if not (df['allocation'].apply(np.sum) == sum(game.auction_params.licenses)).all():
                # Some combos do not sell everything, let's skip those for now
                retval['failed'] = True
                retval['failure_reason'] = 'There exists a combo that does not sell everything'
                return retval
            
            if not has_non_zero_allocations(df):
                retval['failed'] = True
                retval['failure_reason'] = 'There exists a type that is hopelessly weak (never allocated)'
                return retval
            
            if not all_products_overdemanded(game):
                retval['failed'] = True
                retval['failure_reason'] = 'Not enough competition'
                return retval
        
            # Test #2: Is this game way too big? (Is this the right metric? We're using MCCFR aftearall... The whole point is NOT needing to expand)
            # This is VERY slow... (fails after 15 minutes for 5_000 nodes, say)
            if max_game_tree_size is not None:
                try:
                    print("Checking wieldy")
                    get_all_states_with_policy.get_all_info_states_with_policy(game, max_num_states=max_game_tree_size)
                except get_all_states_with_policy.TooManyStates:
                    print(f"Failed tree after {time.time() - start:.1f}s")
                    retval['failed'] = True
                    retval['failure_reason'] = 'Game tree size too large'
                    return retval
            
            # Test #3: Is this practically solvable in a reasonable amount of time?
            if is_mccfr_iter_slow(game, external=external):
                print("Cutoff exceeded")
                retval['failed'] = True
                retval['failure_reason'] = 'Slow MCCFR iters'
                return retval
        finally:
            del game

    return retval


def main(seed=1234, output='configs.pkl', external=True, geography=None, track_mem=False):
    if track_mem:
        from pympler import muppy, summary

    
    N_CONFIGS = 5
    MIN_TYPES = 2
    MAX_TYPES = 2
    MIN_BIDDERS = 3
    MAX_BIDDERS = 3
    MAX_ACTION_SPACE = 16
    MAX_NUM_LICENSES = 6

    failures = defaultdict(int)

    configs = []
    base = {
        'scale': 1_000_000,
        'auction_params': {
            'increment': .3,
            'fold_randomness': True,
            'max_rounds': 10, # TODO: Think!
        },
        'bidders': [
        ]
    }
    base['auction_params']['agent_memory'] = base['auction_params']['max_rounds']

    iters = 0
    rng = np.random.default_rng(seed=seed)

    with tqdm() as pbar:
        while len(configs) < N_CONFIGS:
            iters += 1
            pbar.update(1)
            
            sats_seed = rng.integers(int(1e9))
            x = copy.deepcopy(base)
            x['sats_seed'] = sats_seed

            if geography is None:
                x['map'] = str(rng.choice(list(map_generators.keys())))
            else:
                x['map'] = geography

            selected_map = map_generators[x['map']]()
            licenses = []
            for _ in range(len(selected_map)):
                licenses.append(rng.integers(1, MAX_NUM_LICENSES + 1))

            if np.product([l+1 for l in licenses]) > MAX_ACTION_SPACE:
                continue # This will be too big, just kill it

            if sum(licenses) < 3: # This will not be an interesting game, just kill it
                continue

            ap = x['auction_params']
            ap['licenses'] = licenses

            license_mhz = 10
            mhz_per_pop_open = 0.232 # Real value (for 3800 I think?)

            ap['opening_price'] = [int(np.round(license_mhz * mhz_per_pop_open * node.population / x['scale'])) for node in selected_map]
            ap['activity'] = [op for op in ap['opening_price']]

            bidders = x['bidders']
            n_bidders = rng.integers(MIN_BIDDERS, MAX_BIDDERS + 1)
            
            failed = False
            for j in range(n_bidders):
                bidder_types = []
                bidders.append({
                    'player': j,
                    'types': bidder_types
                })
                for _ in range(rng.integers(MIN_TYPES, MAX_TYPES + 1)):
                    bidder = {
                        'type': str(rng.choice(['national', 'regional', 'local'])),
                        'value_per_subscriber': {
                            'lower': 25,
                            'upper': 35,
                        },
                    }

                    bidder_types.append(bidder)
                    # TODO: How do I make sure the players are "reasonably" powerful relatively?
                    # TODO: Are your types meaningfully different? (Check efficient allocation make sure at least one product for everyone. This check should be first for easier rejection)
                    if bidder['type'] == 'regional':
                        bidder['market_share'] = {
                            'lower': 0.04,
                            'upper': 0.1,
                        }
                        bidder['hq'] = rng.integers(len(ap['licenses']))
                    elif bidder['type'] == 'local':
                        bidder['market_share'] = {
                            'lower': 0.05,
                            'upper': 0.12,
                        }                       

                        def generate_local_bidder(num_attempts):
                            for _ in range(num_attempts):
                                local_regions = [rng.integers(2) for _ in selected_map] # Can't have no regions or you're values are all 0
                                if 0 < sum(local_regions) < len(local_regions):
                                    return local_regions
                        
                        local_regions = generate_local_bidder(500)
                        if local_regions is not None:
                            bidder['local_regions'] = local_regions
                        else:
                            failed = True
                    elif bidder['type'] == 'national':
                        bidder['market_share'] = {
                            'lower': 0.08,
                            'upper': 0.22,
                        }

                    # Want higher marginal values on secondary licenses
                    MARKET_SHARE_BOOST = 5 # TODO: Sample this from 1 to 10?
                    bidder['market_share']['lower'] = bidder['market_share']['lower'] * MARKET_SHARE_BOOST
                    bidder['market_share']['upper'] = bidder['market_share']['upper'] * MARKET_SHARE_BOOST

            if failed:
                continue

            retval = test_config_is_wieldy(x, external=external)
            if not retval['failed']:
                print("SUCCESS")
                print(x)
                print(retval['sats_config'])
                configs.append(x)
                with open(output, 'wb') as f: # Will keep overwriting
                    pickle.dump(configs, f)
                print(f"Acceptance rate is {len(configs) / iters:.5%}")
            else:
                failures[retval['failure_reason']] += 1
                if retval['failure_reason'] == 'Slow MCCFR iters':
                    print(failures) 
            
            # # Either way, purge ca∂ches
            # import gc
            # import functools
            # gc.collect()
            # wrappers = [a for a in gc.get_objects() if isinstance(a, functools._lru_cache_wrapper)]

            # for wrapper in wrappers:
            #     if 'ClockAuction' in wrapper.__qualname__:
            #         wrapper.cache_clear()

            if track_mem:
                all_objects = muppy.get_objects()
                summ = summary.summarize(all_objects)
                summary.print_(summ)  

if __name__ == '__main__':
    # TODO: Switch prints to logs
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--geography', type=str)
    parser.add_argument('--output', type=str, default='configs.pkl')
    parser.add_argument('--external', type=util.strtobool, default=1) 
    parser.add_argument('--track_mem', type=util.strtobool, default=0) 
    args = parser.parse_args()
    main(seed=args.seed, output=args.output, external=args.external, geography=args.geography, track_mem=args.track_mem)



# map: BC
# scale: 1_000_000
# auction_params:
#   licenses: [3]
#   activity: [1]
#   opening_price: [18]
#   information_policy: show_demand
#   undersell_rule: undersell
#   increment: 1
#   fold_randomness: True
#   max_rounds: 5
#   agent_memory: 20
# bidders:
#   -
#     player: 0
#     types:
#       -
#         type: explicit
#         name: "normal"
#         values: [0, 60, 120, 120]
#   - 
#     player: 1
#     types:
#       -
#         type: explicit
#         name: "normal"
#         values: [0, 100, 250, 250]
#         straightforward: true