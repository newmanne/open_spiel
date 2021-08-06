# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example use of the C++ MCCFR algorithms on Clcok Auction.

This examples calls the underlying C++ implementations via the Python bindings.
Note that there are some pure Python implementations of some of these algorithms
in python/algorithms as well.
"""

import pickle
from absl import app
from absl import flags
from pathlib import Path
import sys
import pyspiel
import pandas as pd
import json
import re
import numpy as np
import time
import os
import itertools

from open_spiel.python import policy
from open_spiel.python.algorithms import cfr, outcome_sampling_mccfr, expected_game_score, exploitability, get_all_states_with_policy
import logging
from sharpen_solution import sharpen_solution

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS

flags.DEFINE_integer("report_freq", 100, "Report frequency")
flags.DEFINE_bool("persist", True, "Pickle the models")
flags.DEFINE_integer("persist_freq", -1, "Pickle the models every this many iterations")

flags.DEFINE_bool("python", False, "Use python CFR impls")
flags.DEFINE_bool("turn_based", True, "Convert simultaneous to turn based")
flags.DEFINE_bool("use_best",True,"Use best iterate and not last iterate")

flags.DEFINE_enum("solver", "cfr", ["cfr", "cfrplus", "cfrbr", "mccfr", "ecfr"], "CFR solver")
flags.DEFINE_enum("sampling", "external", ["external", "outcome"], "Sampling for the MCCFR solver")
flags.DEFINE_enum("metric", "max_regret", ["max_regret", "nash_conv"], "Metric to use for stopping condition")

flags.DEFINE_integer("iterations", 50, "Number of iterations")
flags.DEFINE_string("output", "output", "Name of the output folder")

flags.DEFINE_string("game", "clock_auction", "Name of the game")

# Game params for clock auction
flags.DEFINE_string("filename", 'parameters.json', "Filename with parameters")
flags.DEFINE_integer("seed", '123', "Seed for randomized algs")

# ECFR
flags.DEFINE_float("initial_eps", 1e-1, "Initial epsilon")
flags.DEFINE_float("decay_factor", 0.99, "Decay factor")
flags.DEFINE_integer("decay_freq", 500, "Decay frequency")
flags.DEFINE_float("min_eps", 1e-6, "Minimum epsilon")


price_pattern = re.compile(r'^.*(Price:.*)$', flags=re.MULTILINE)
allocation_pattern = re.compile(r'.*(Final bids:.*)$.*', flags=re.MULTILINE)
round_pattern = re.compile(r'.*(Round:.*)$.*', flags=re.MULTILINE)

def num_to_char(i):
    return chr(ord("@")+i+1)

def pretty_bid(b):
    s = ''
    for i, a in enumerate(b):
        s += str(a) + ' ' + num_to_char(i) + (', ' if i != len(b) - 1 else '')
    return s


def recurse(accum, stack, sequences, index):
    sequence = sequences[index]
    for i in sequence:
        stack.append(i)
        if index == 0:
            accum.append(list(stack))
        else:
            recurse(accum, stack, sequences, index - 1)
        stack.pop()

def simple_product(sequences):
    '''Dont use itertools.product because what if it gives a different ordering? This is a copy of the C++ code so it should give the same'''
    accum = []
    stack = []
    if len(sequences) > 0:
        recurse(accum, stack, sequences, len(sequences) - 1)
    return accum

    
def action_to_bids(licenses):
    bids = []
    for n in licenses:
        b = []
        for i in range(n + 1):
            b.append(i)
        bids.append(b)
    actions = simple_product(bids)
    return {i: pretty_bid(a) for i, a in enumerate(actions)}


def state_to_final(game, s):
    '''Convert a state into unique final outcomes (but not caring about bidding being different in the middle). i.e., the allocation and the types and the price are all the same'''
    state_str = str(s)
    info_state_string = '\n'.join(state_str.split('\n')[:game.num_players()]) + '\n' + re.findall(allocation_pattern, state_str)[0] + '\n' + re.findall(price_pattern, state_str)[0] + '\n' + re.findall(round_pattern, state_str)[0]
    return info_state_string

def parse_state_str(game, state, info_state_str):
    d = dict()
    state_str = str(state)
    price = re.findall(price_pattern, state_str)[0].replace('Price:', '')
    for i, p in enumerate(price.split(',')):
        d[f'Price {num_to_char(i)}'] = p
    round_number = int(re.findall(round_pattern, state_str)[0].replace('Round:', ''))
    d['round'] = round_number
    if state.is_terminal():
        allocation = [x for x in re.findall(allocation_pattern, state_str)[0].replace('Final bids:', '').strip().split('|')]
        for i, player_alloc in enumerate(allocation):
            for j, a in enumerate(player_alloc.split(',')):
                d[f'Allocation {i} {num_to_char(j)}'] = a
    else:
        splits = info_state_str.splitlines()
        d['type'] = splits[1]
        d['my_bids'] = splits[2] if round_number > 1 else ''
        if len(splits) > 3:
            d['total_demand'] = splits[3] if round_number > 1 else ''
        else:
            d['total_demand'] = '?' if round_number > 1 else ''
    return d


def persist_model(solver, i):
    logger.info("Persisting the model...")
    model_name = f'{FLAGS.solver}'
    if FLAGS.solver == 'mccfr':
        model_name += f'_{FLAGS.sampling}'
    
    try:
        with open(f'{FLAGS.output}/{model_name}_{i}.pkl', "wb") as f:
            pickle.dump(solver, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logger.exception("Error pickling solver!!!")

def main(_):
    start_time = time.time()
    Path(FLAGS.output).mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(f'{FLAGS.output}/{FLAGS.solver}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    # LOAD GAME
    load_function = pyspiel.load_game if not FLAGS.turn_based else pyspiel.load_game_as_turn_based
    params = dict()
    if FLAGS.game == 'clock_auction':
        params['filename'] = FLAGS.filename

    game = load_function(
        FLAGS.game,
        params,
    )

    solver_config = dict()
    solver_config['solver'] = FLAGS.solver
    solver_config['seed'] = FLAGS.seed
    solver_config['python'] = FLAGS.python

    # LOAD SOLVER
    if not FLAGS.python:
        logger.info("Using C++ implementations")
        if FLAGS.solver == "cfr":
            logger.info("Using CFR solver")
            solver = pyspiel.CFRSolver(game)
        elif FLAGS.solver == "cfrplus":
            logger.info("Using CFR+ solver")
            solver = pyspiel.CFRPlusSolver(game)
        elif FLAGS.solver == "cfrbr":
            logger.info("Using CFR-BR solver")
            solver = pyspiel.CFRBRSolver(game)
        elif FLAGS.solver == "mccfr":
            logger.info("Using MCCFR solver")
            solver_config['sampling'] = FLAGS.sampling
            if FLAGS.sampling == "external":
                logger.info("Using external sampling")
                solver = pyspiel.ExternalSamplingMCCFRSolver(game, seed=FLAGS.seed, avg_type=pyspiel.MCCFRAverageType.FULL)
            elif FLAGS.sampling == "outcome":
                logger.info("Using outcome sampling")
                solver = pyspiel.OutcomeSamplingMCCFRSolver(game)
        elif FLAGS.solver == "ecfr":
            logger.info("Using EpsilonCFR solver")
            initial_eps = FLAGS.initial_eps
            decay_factor = FLAGS.decay_factor
            decay_freq = FLAGS.decay_freq
            min_eps = FLAGS.min_eps
            solver_config['initial_eps'] = FLAGS.initial_eps
            solver_config['decay_factor'] = FLAGS.decay_factor
            solver_config['decay_freq'] = FLAGS.decay_freq
            solver_config['min_eps'] = FLAGS.min_eps
            solver = pyspiel.EpsilonCFRSolver(game, initial_eps)
    else:
        logger.info("Using python implementations")
        if FLAGS.solver == "cfr":
            logger.info("Using CFR solver")
            solver = cfr.CFRSolver(game)
        elif FLAGS.solver == "cfrplus":
            logger.info("Using CFR+ solver")
            solver = cfr.CFRPlusSolver(game)
        elif FLAGS.solver == "cfrbr":
            logger.info("Using CFR-BR solver")
            solver = cfr.CFRBRSolver(game)
        elif FLAGS.solver == "mccfr":
            logger.info("Using MCCFR solver")
            if FLAGS.sampling == "outcome":
                logger.info("Using outcome sampling")
                solver = OutcomeSamplingSolver(game)
            else:
                raise ValueError("Not external")
            
    with open(f'{FLAGS.output}/solver.json', 'w') as f:
        json.dump(solver_config, f)

    # RUN SOLVER
    run_records = []
    best_policy = None
    best_metric_value = float("inf")
    for i in range(FLAGS.iterations):
        if i % 100 == 0:
            logger.info(f"Starting iteration {i}")

        if FLAGS.solver == "mccfr":
            if FLAGS.python:
                solver.iteration()
            else:
                solver.run_iteration() 
        else:
            solver.evaluate_and_update_policy()

        if FLAGS.solver == 'ecfr' and i > 0 and i % FLAGS.decay_freq == 0:
            old_eps = solver.epsilon()
            new_eps = max(solver.epsilon() * decay_factor, min_eps)
            solver.set_epsilon(new_eps)
            logger.info(f"Changing epsilon from {old_eps} to {new_eps}")

        if i % FLAGS.report_freq == 0 or i == FLAGS.iterations - 1:
            policy = solver.average_policy() if FLAGS.solver != 'ecfr' else solver.tabular_average_policy()
            record = dict(iteration=i, walltime=time.time() - start_time)
            if FLAGS.python:
                metric = exploitability.nash_conv(game, policy)
            else:
                regrets = pyspiel.player_regrets(game, policy, False)
                max_regret = max(regrets)
                nash_conv = sum(regrets)
                record['max_on_path_regret'] = max_regret
                record['nash_conv'] = nash_conv
                if FLAGS.solver == 'ecfr':
                    br_info = pyspiel.nash_conv_with_eps(game, policy)
                    merged_table = pyspiel.merge_tables(br_info.cvtables)
                    record['max_qv_diff'] = merged_table.max_qv_diff()
                    record['avg_qv_diff'] = merged_table.avg_qv_diff()
                if FLAGS.use_best:
                    curr_metric_value = max_regret if FLAGS.metric=="max_regret" else nash_conv
                    if best_policy == None or curr_metric_value < best_metric_value:
                        best_policy = policy
                        best_metric_value = curr_metric_value
                        logger.info(f"Updated best policy on iteration {i}")

            run_records.append(record)
            logger.info(f"Iteration {i}")
            for k, v in record.items():
                if k == 'iteration':
                    continue
                logger.info(f"{k}={v:.6f}")
            # TODO: Appending would be better...
            pd.DataFrame.from_records(run_records).to_csv(f'{FLAGS.output}/run_metrics.csv', index=False)


        if FLAGS.persist and FLAGS.persist_freq > 0 and i % FLAGS.persist_freq == 0 and i > 0:
            persist_model(solver, i)

    if FLAGS.use_best:
        policy = best_policy

    if FLAGS.persist:
        persist_model(solver, i)

    pd.DataFrame.from_records(run_records).to_csv(f'{FLAGS.output}/run_metrics.csv', index=False)

    records = []
    if FLAGS.solver == 'ecfr':
        info_state_to_cve = merged_table.table()
    else:
        info_state_to_cve = None

    if FLAGS.python:
        raise ValueError("TODO")
    else:
        all_states = get_all_states_with_policy.get_all_info_states_with_policy(
            game,
            depth_limit=-1,
            include_terminals=True,
            policy=policy,
            to_string=lambda s: state_to_final(game, s),
        )

    with open(os.getenv('CLOCK_AUCTION_CONFIG_DIR') + f'/{FLAGS.filename}', 'r') as f:
        params = json.load(f)
    a2b = action_to_bids(params['licenses'])

    records = []
    for info_state_key, state_dict in all_states.items():
        state = state_dict['state']
        prob = state_dict['prob']
        record = dict(terminal=state.is_terminal(), prob=prob, player=state.current_player())
        record.update(parse_state_str(game, state, info_state_key))
        if state.is_terminal():
            record.update({f'Utility {n}': v for n,v in enumerate(state.returns())})
            splits = info_state_key.splitlines()
            for n, u in enumerate(state.returns()):
                # 1 entry per player per terminal state for better filtering
                r = dict(record)
                r.update(dict(player=n, info_state=info_state_key, type=splits[n]))
                records.append(r)
        else:
            action_probabilities = policy.action_probabilities(state)
            record['info_state'] = info_state_key[18:] # Get rid of "Current Player line"
            record.update({f'Bid ({a2b[k]})': v for k, v in action_probabilities.items()})
            records.append(record)

    df = pd.DataFrame.from_records(records).set_index('info_state')
    df['value'] = df.type.str.extract(r'v(.+)b.*')
    df['budget'] = df.type.str.extract(r'.*b(.+)$').astype(float)
    df = df.drop(['type'], axis=1)

    df = df.reindex(sorted(df.columns, key=str), axis=1) # Sort columns alphabetically
    output_path = f'{FLAGS.output}/strategy.csv'
    logger.info(f"Saving strategy to {output_path}")
    df.to_csv(output_path)

    logger.info(f"Running processing script")
    sharpen_solution(output_path)

    # logger.info("Loading the model...")
    # with open(MODEL_FILE_NAME.format(FLAGS.sampling), "rb") as file:
    #   loaded_solver = pickle.load(file)

if __name__ == "__main__":
  app.run(main)