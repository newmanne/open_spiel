import argparse
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import pickle
import copy
import os
import time
import tempfile

import matplotlib.pyplot as plt
plt.style.use('https://raw.githubusercontent.com/gregdeon/plots/main/style.mplstyle')

from open_spiel.python.examples.pysats import map_generators, run_sats
from open_spiel.python.examples.sats_game_sampler import test_config_is_wieldy 
from open_spiel.python.games.clock_auction_base import action_to_bundles 

LICENSE_MHZ = 10

# --- plotting
FIGURE_DIR = '/global/scratch/open_spiel/open_spiel/notebooks/greg/figures/sampler'
PID_TO_STYLE = {
    0: 'solid',
    1: 'dotted',
    2: 'dashed',
}

def visualize_values(config, fname=None):
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as fp:
        sats_config = run_sats(config, fp.name, seed=config['sats_seed'])

    bundles = action_to_bundles(config['auction_params']['licenses'])
    auction_map, encumberance = map_generators[config['map']]
    spectrum_per_license = np.array(encumberance).reshape(-1, 1) * LICENSE_MHZ
    bandwidths = (bundles @ spectrum_per_license).flatten()
    idx = np.argsort(bandwidths)

    for player_id in range(len(sats_config['players'])):
        for type_id in range(len(sats_config['players'][player_id]['type'])):
            values = np.array(sats_config['players'][player_id]['type'][type_id]['value']) #- 12 * bandwidths[idx] # You aren't paying less than this
            plt.plot(bandwidths[idx], values[idx], label=f'Player {player_id} Type {type_id}', linestyle=PID_TO_STYLE[player_id])
            plt.scatter(bandwidths[idx], values[idx], s=4, clip_on=False)
    plt.legend()
    plt.xlabel('Bandwidth')
    plt.ylabel('Value')
    plt.xlim(0, None)
    plt.ylim(0, None)
    # plt.axhline(12*4*.6*1.05*1.05)
    if fname is None:
        plt.show()
    else:
        full_fname = f'{FIGURE_DIR}/{fname}'
        plt.savefig(full_fname, bbox_inches='tight')
        print(f'Saved plot to {full_fname}')
    plt.close()

# --- sampling
def print_failure_reasons(failures_dict, print_fn=print):
    print_fn('Failure reasons:')
    for failure_reason, count in sorted(failures_dict.items(), key=lambda x: x[1], reverse=True):
        print_fn(f"- {failure_reason}: {count}")

def sample_games(args):
    if args.delete_if_exists:
        try:
            os.remove(args.output)
        except FileNotFoundError:
            pass

    print(f"Sampling games with args: {vars(args)}")

    failures = defaultdict(int)

    configs = []
    base = {
        'scale': 1_000_000,
        'auction_params': {
            'increment': args.clock_increment,
            'max_rounds': args.max_rounds, 
            'heuristic_deviations': args.heuristic_deviations,
            'agent_memory': args.max_rounds,
        },
        'bidders': [],
    }

    iters = 0
    last_print_time = time.time()
    rng = np.random.default_rng(seed=args.seed)

    if args.action_prefix is not None:
        all_bids = action_to_bundles(args.licenses)
        bids_in_prefix = all_bids[args.action_prefix]
        print(f'Prefix bids:\n {bids_in_prefix}')

    with tqdm() as pbar:
        while len(configs) < args.n_configs:
            sats_seed = rng.integers(int(1e9))
            x = copy.deepcopy(base)
            x['sats_seed'] = sats_seed
            x['map'] = args.map_name

            map_generator, bid_to_quantity_matrix = map_generators[x['map']]
            selected_map = map_generator()
            if bid_to_quantity_matrix is None:
                bid_to_quantity_matrix = np.eye(len(selected_map))

            ap = x['auction_params']
            ap['licenses'] = args.licenses

            mhz_per_pop_open = 0.232 # Real value (for 3800 I think?)

            region_opening_prices = np.array([int(np.round(LICENSE_MHZ * mhz_per_pop_open * node.population / x['scale'])) for node in selected_map])
            product_opening_prices = region_opening_prices @ bid_to_quantity_matrix # opening prices of encumbered licenses are proportional to bandwidth
            product_opening_prices = np.clip(product_opening_prices, np.min(region_opening_prices) * 0.05, None) # signalling products are worth 5% of cheapest region
            product_opening_prices = np.array([int(np.round(p)) for p in product_opening_prices])
            ap['opening_price'] = product_opening_prices.tolist()
            ap['activity'] = [op for op in ap['opening_price']]

            bidders = x['bidders']
            n_bidders = rng.integers(args.min_bidders, args.max_bidders + 1)

            for j in range(n_bidders):
                bidder_types = []
                bidders.append({
                    'player': j,
                    'types': bidder_types,
                    'action_prefix': args.action_prefix,
                })
                for _ in range(rng.integers(args.min_types, args.max_types + 1)):
                    bidder = {
                        'type': 'local',
                        'value_per_subscriber': {
                            'lower': args.min_value_per_subscriber,
                            'upper': args.max_value_per_subscriber,
                        },
                        'market_share': {
                            'lower': args.min_market_share,
                            'upper': args.max_market_share,
                        },
                        'z_spread': args.z_spread,
                        'local_regions': [1],
                        'allow_float_values': args.allow_float_values,
                    }

                    bidder_types.append(bidder)

            if args.plot_values:
                visualize_values(x, fname=f'{x["sats_seed"]}.png')
                    
            retval = test_config_is_wieldy(x, external=args.use_external_mccfr, min_mccfr_iters=args.min_mccfr_iters, max_rounds=args.max_straightforward_rounds, max_rounds_alternating=args.max_straightforward_rounds_alternating, verbose=False)
            if not retval['failed']:
                pbar.write("SUCCESS")
                pbar.write(str(x))
                # pbar.write(str(retval['sats_config']))
                configs.append(x)
                with open(args.output, 'wb') as f: # Will keep overwriting
                    pickle.dump(configs, f)
            else:
                failures[retval['failure_reason']] += 1

            iters += 1
            pbar.update(1)
            pbar.set_description(f"Accepted {len(configs) / iters:.5%}")

            if time.time() - last_print_time > args.report_interval:
                print_failure_reasons(failures, print_fn=pbar.write)
                last_print_time = time.time()

    print('All done, goodbye!')
    print_failure_reasons(failures)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # auction settings
    parser.add_argument('--n_configs', type=int, default=5)
    parser.add_argument('--map_name', type=str, default='BCEncumbered')
    # parser.add_argument('--licenses', type=list, default=[1, 4])
    # parser.add_argument('--action_prefix', type=list, default=[4, 4])
    parser.add_argument('--licenses', nargs='*', type=int, default=[1, 4])
    parser.add_argument('--action_prefix', nargs='*', type=int, default=[4, 4])
    parser.add_argument('--clock_increment', type=float, default=0.05)
    parser.add_argument('--max_rounds', type=int, default=15)
    parser.add_argument('--heuristic_deviations', type=int, default=1000)

    # value model settings
    parser.add_argument('--min_types', type=int, default=3)
    parser.add_argument('--max_types', type=int, default=3)
    parser.add_argument('--min_bidders', type=int, default=2)
    parser.add_argument('--max_bidders', type=int, default=2)
    parser.add_argument('--min_value_per_subscriber', type=int, default=20)
    parser.add_argument('--max_value_per_subscriber', type=int, default=35)
    parser.add_argument('--min_market_share', type=float, default=0.35)
    parser.add_argument('--max_market_share', type=float, default=0.5)
    parser.add_argument('--z_spread', type=float, default=0.1)
    parser.add_argument('--allow_float_values', action='store_true')

    # wieldiness/interestingness settings
    parser.add_argument('--max_straightforward_rounds', type=int, default=20)
    parser.add_argument('--max_straightforward_rounds_alternating', type=int, default=25)
    parser.add_argument('--min_mccfr_iters', type=int, default=20, help='Minimum iters/s for MCCFR to be considered wieldy')
    parser.add_argument('--use_external_mccfr', type=bool, default=True)

    # other settings
    parser.add_argument('--seed', type=int, default=7777)
    parser.add_argument('--output', type=str, default='configs/debug.pkl')
    parser.add_argument('--delete_if_exists', action='store_true', help='Delete existing output file before writing new one') 
    parser.add_argument('--report_interval', type=int, default=30, help='How often to print failure reasons (in seconds)')
    parser.add_argument('--plot_values', action='store_true', help='Plot the values of the sampled games')

    args = parser.parse_args()
    sample_games(args)
