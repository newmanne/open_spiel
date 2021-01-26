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

import pyspiel
import pandas as pd
import json

from open_spiel.python import policy
from open_spiel.python.algorithms import cfr, outcome_sampling_mccfr, expected_game_score, exploitability, get_all_states


FLAGS = flags.FLAGS

flags.DEFINE_bool("python", False, "Use python CFR impls")
flags.DEFINE_bool("turn_based", True, "Convert simultaneous to turn based")

flags.DEFINE_enum("solver", "cfr", ["cfr", "cfrplus", "cfrbr", "mccfr"], "CFR solver")
flags.DEFINE_enum("sampling", "external", ["external", "outcome"], "Sampling for the MCCFR solver")
flags.DEFINE_integer("iterations", 50, "Number of iterations")
flags.DEFINE_string("output", "output", "Name of the output folder")

flags.DEFINE_string("game", "clock_auction", "Name of the game")

# Game params for clock auction
flags.DEFINE_string("filename", 'parameters.json', "Filename with parameters")


def main(_):
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
        print("Using C++ implementations")
        if FLAGS.solver == "cfr":
            print("Using CFR solver")
            solver = pyspiel.CFRSolver(game)
        elif FLAGS.solver == "cfrplus":
            print("Using CFR+ solver")
            solver = pyspiel.CFRPlusSolver(game)
        elif FLAGS.solver == "cfrbr":
            print("Using CFR-BR solver")
            solver = pyspiel.CFRBRSolver(game)
        elif FLAGS.solver == "mccfr":
            print("Using MCCFR solver")
            if FLAGS.sampling == "external":
                print("Using external sampling")
                solver = pyspiel.ExternalSamplingMCCFRSolver(game, avg_type=pyspiel.MCCFRAverageType.FULL)
            elif FLAGS.sampling == "outcome":
                print("Using outcome sampling")
                solver = pyspiel.OutcomeSamplingMCCFRSolver(game)
    else:
        print("Using python implementations")
        if FLAGS.solver == "cfr":
            print("Using CFR solver")
            solver = cfr.CFRSolver(game)
        elif FLAGS.solver == "cfrplus":
            print("Using CFR+ solver")
            solver = cfr.CFRPlusSolver(game)
        elif FLAGS.solver == "cfrbr":
            print("Using CFR-BR solver")
            solver = cfr.CFRBRSolver(game)
        elif FLAGS.solver == "mccfr":
            print("Using MCCFR solver")
            if FLAGS.sampling == "outcome":
                print("Using outcome sampling")
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
        print("Iteration {} nash_conv: {:.6f}".format(i, nash_conv))

    print("Persisting the model...")
    model_name = f'{FLAGS.solver}_{FLAGS.python}'
    if FLAGS.solver == 'mccfr':
        model_name += f'_{FLAGS.sampling}'
    with open(f'{FLAGS.output}/{model_name}.pkl', "wb") as f:
        pickle.dump(solver, f, pickle.HIGHEST_PROTOCOL)

    Path(FLAGS.output).mkdir(parents=True, exist_ok=True)
    pd.Series(nash_convs).to_csv(f'{FLAGS.output}/nash_conv.csv')

    records = []
    if FLAGS.python:
        raise ValueError("TODO")
      # # for info_state_str in tabular_policy.state_lookup.keys():
      # player = -1
      # for player_info_states in policy.states_per_player:
      #   player += 1
      #   for info_state in player_info_states:
      #     record = dict(player=player, info_state=info_state)
      #     state_policy = policy.policy_for_key(info_state)
      #     record.update(state_policy)
      #     records.append(record)

    else:
        all_states = get_all_states.get_all_states(
        game,
        depth_limit=-1,
        include_terminals=False,
        include_chance_states=False
        )
        records = []
        for info_state, state in all_states.items():
            action_probabilities = policy.action_probabilities(state)
            info_state_string = state.information_state_string()
            info_state_string = info_state_string[18:] # Get rid of "Current Player line"
            record = dict(info_state=info_state_string, player=state.current_player())
            record.update(action_probabilities)
            records.append(record)

    pd.DataFrame.from_records(records).set_index('info_state').to_csv(f'{FLAGS.output}/strategy.csv')

    # print("Loading the model...")
    # with open(MODEL_FILE_NAME.format(FLAGS.sampling), "rb") as file:
    #   loaded_solver = pickle.load(file)

if __name__ == "__main__":
  app.run(main)
