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

from open_spiel.python import policy
from open_spiel.python.algorithms import cfr, outcome_sampling_mccfr, expected_game_score, exploitability, get_all_states
import logging

logger = logging.getLogger(__name__)

FLAGS = flags.FLAGS

flags.DEFINE_bool("persist", False, "Pickle the models")
flags.DEFINE_bool("python", False, "Use python CFR impls")
flags.DEFINE_bool("turn_based", True, "Convert simultaneous to turn based")

flags.DEFINE_enum("solver", "cfr", ["cfr", "cfrplus", "cfrbr", "mccfr"], "CFR solver")
flags.DEFINE_enum("sampling", "external", ["external", "outcome"], "Sampling for the MCCFR solver")
flags.DEFINE_integer("iterations", 50, "Number of iterations")
flags.DEFINE_string("output", "output", "Name of the output folder")

flags.DEFINE_string("game", "clock_auction", "Name of the game")

# Game params for clock auction
flags.DEFINE_string("filename", 'parameters.json', "Filename with parameters")
flags.DEFINE_integer("seed", '123', "Seed for randomized algs")


def main(_):
    Path(FLAGS.output).mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(f'{FLAGS.output}/{FLAGS.solver}.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    # LOAD GAME
    load_function = pyspiel.load_game if not FLAGS.turn_based else pyspiel.load_game_as_turn_based
    params = dict()
    if FLAGS.game == 'clock_auction':
        params['filename'] = pyspiel.GameParameter(FLAGS.filename)

    game = load_function(
        FLAGS.game,
        params,
    )

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
            if FLAGS.sampling == "external":
                logger.info("Using external sampling")
                solver = pyspiel.ExternalSamplingMCCFRSolver(game, seed=FLAGS.seed, avg_type=pyspiel.MCCFRAverageType.FULL)
            elif FLAGS.sampling == "outcome":
                logger.info("Using outcome sampling")
                solver = pyspiel.OutcomeSamplingMCCFRSolver(game)
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

    # RUN SOLVER
    nash_convs = []
    for i in range(int(FLAGS.iterations)):
        if FLAGS.solver == "mccfr":
            if FLAGS.python:
                solver.iteration()
            else:
                solver.run_iteration() 
        else:
            solver.evaluate_and_update_policy()

        policy = solver.average_policy()
        if FLAGS.python:
            nash_conv = exploitability.nash_conv(game, policy)
        else:
            nash_conv = pyspiel.nash_conv(game, policy)

        if nash_conv < 0:
            raise ValueError("NEGATIVE NASH CONV! Is your game not perfect recall? Do two different states have the same AuctionState::ToString() representation?")

        nash_convs.append(nash_conv)
        # TODO: Should probably append to a file here, or every few iters. Don't think it makes sense to wait until the end
        logger.info("Iteration {} nash_conv: {:.6f}".format(i, nash_conv))

    if FLAGS.persist:
        logger.info("Persisting the model...")
        model_name = f'{FLAGS.solver}_{FLAGS.python}'
        if FLAGS.solver == 'mccfr':
            model_name += f'_{FLAGS.sampling}'
        
        with open(f'{FLAGS.output}/{model_name}.pkl', "wb") as f:
            pickle.dump(solver, f, pickle.HIGHEST_PROTOCOL)

    pd.Series(nash_convs, name='nash_conv').to_csv(f'{FLAGS.output}/nash_conv.csv')

    records = []
    if FLAGS.python:
        raise ValueError("TODO")
    else:
        # These are all the STATES. But you want all the INFOSTATES, so make sure to reject duplicates
        all_states = get_all_states.get_all_states(
            game,
            depth_limit=-1,
            include_terminals=True,
            include_chance_states=False
        )
        records = []
        seen = set()
        pattern = re.compile(r'.*(Final bids:.*)$.*', flags=re.MULTILINE)
        for state_code, state in all_states.items():
            if state.is_terminal():
                action_probabilities = dict()
                # NOT an infostate string - but a terminal
                info_state_string = re.findall(pattern, str(state))[0]
            else:
                action_probabilities = policy.action_probabilities(state)
                info_state_string = state.information_state_string()
                info_state_string = info_state_string[18:] # Get rid of "Current Player line"
            if info_state_string not in seen:
                seen.add(info_state_string)
                record = dict(info_state=info_state_string, player=state.current_player(), terminal=state.is_terminal())
                record.update(action_probabilities)
                if state.is_terminal():
                    record.update({f'Utility {n}': u for n, u in enumerate(state.returns())})
                records.append(record)

    df = pd.DataFrame.from_records(records).set_index('info_state')
    df = df.reindex(sorted(df.columns, key=str), axis=1) # Sort columns alphabetically
    output_path = f'{FLAGS.output}/strategy.csv'
    logger.info(f"Saving strategy to {output_path}")
    df.to_csv(output_path)

    # logger.info("Loading the model...")
    # with open(MODEL_FILE_NAME.format(FLAGS.sampling), "rb") as file:
    #   loaded_solver = pickle.load(file)

if __name__ == "__main__":
  app.run(main)
