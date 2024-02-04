"""
Adapted from SATSGameSampler.ipynb.

Sample usage:
> python write_configs.py --sats_fnames /global/scratch/open_spiel/open_spiel/notebooks/greg/configs/jan19_boring.pkl --prefix jan19_boring
"""

import argparse
from collections import defaultdict
import copy
import os
import pickle
import shutil
import subprocess
from tqdm import tqdm
import yaml

from open_spiel.python.games.clock_auction_base import *

CONFIG_DIR = os.environ['CLOCK_AUCTION_CONFIG_DIR']
PYSATS_DIR = os.environ['PYSATS_DIR']
# PYSATS_DIR = '/apps/open_spiel/open_spiel/python/examples'

def convert_pesky_np(d):
    """
    TODO: import from auctions.webutils; copying here to avoid importing Django-only code
    """
    if isinstance(d, np.ndarray):
        return convert_pesky_np(d.tolist())
    elif isinstance(d, list):
        return [convert_pesky_np(x) for x in d]
    elif isinstance(d, np.int64):
        return int(d)
    elif isinstance(d, np.float32):
        return float(d)
    elif isinstance(d, dict):
        return {k: convert_pesky_np(v) for k, v in d.items()}
    else:
        return d

def generate_mods(base_config):
    """
    TODO: find a way to write configs without constantly needing to edit this function...
    """
    mods = defaultdict()
    for deviations in [1000]:
        for rho in [0, 1]:
            config = copy.deepcopy(base_config)
            n_types = len(config['bidders'][0]['types'])

            # Common settings
            config['auction_params']['agent_memory'] = 10
            config['auction_params']['max_rounds'] = 10
            config['auction_params']['heuristic_deviations'] = deviations
            config['auction_params']['sor_bid_bonus_rho'] = rho
            config['auction_params']['information_policy'] = 'show_demand'
            
            # Base config
            base_string = f'base_dev{deviations}_rho{rho}_t{n_types}'
            mods[base_string] = config

            # Tiebreaking
            tie_break = copy.deepcopy(mods[base_string])
            tie_break['auction_params']['tiebreaking_policy'] = TiebreakingPolicy.DROP_BY_LICENSE.name
            mods[f'{base_string}_tie_break'] = tie_break

    return mods


def main(args):
    prefix = args.prefix
    config_fnames = args.sats_fnames
    OUTPUT_DIR = f'{PYSATS_DIR}/{prefix}'

    # Load SATS configs
    configs = []
    for config_fname in config_fnames:
        with open(config_fname, 'rb') as f:
            configs += pickle.load(f)
    print(f'Loaded {len(configs)} samples.')

    # Generate a set of modified configs for each game
    all_mods = {i: generate_mods(config) for i, config in enumerate(configs)}
    print(f'Created {sum(len(mods) for mods in all_mods.values())} modified configs for {len(all_mods)} base games.')

    # Write out YML files for the modified games
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    shutil.rmtree(f'{CONFIG_DIR}/{prefix}', ignore_errors=True)
    os.makedirs(f'{CONFIG_DIR}/{prefix}', exist_ok=True)
    print(f'Running SATS...')
    for base_name, game_mods in tqdm(all_mods.items()):
        for mod_name, mod in game_mods.items():
            g_name = f'{base_name}_{mod_name}'
            outfile_name = f'{OUTPUT_DIR}/{g_name}.yml'
            with open(outfile_name, 'w') as f:
                yaml.dump(convert_pesky_np(mod), f)
            
            # Run SATS
            subprocess.run(f'python {PYSATS_DIR}/pysats.py --config_file {outfile_name} --seed {mod["sats_seed"]} --output_file {CONFIG_DIR}/{prefix}/{prefix}_{g_name}.json', shell=True)

    print(f'All done, goodbye!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Write game config files from PySATS samples.')
    parser.add_argument('--sats_fnames', type=str, nargs='+', help='Config file names.')
    parser.add_argument('--prefix', type=str, default='jan19boring', help='Prefix for config paths.')
    args = parser.parse_args()
    main(args)
