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

import pyspiel
import pandas as pd

FLAGS = flags.FLAGS

flags.DEFINE_enum("solver", "cfr", ["cfr", "cfrplus", "cfrbr", "mccfr"], "CFR solver")
flags.DEFINE_enum("sampling", "external", ["external", "outcome"], "Sampling for the MCCFR solver")
flags.DEFINE_integer("iterations", 50, "Number of iterations")
# flags.DEFINE_string("game", "clock_auction", "Name of the game")

# Game params for clock auction
flags.DEFINE_string("filename", 'parameters.json', "Filename with parameters")


MODEL_FILE_NAME = "{}_sampling_mccfr_solver.pickle"


def main(_):
    # # TODO: Load sequential

    # game = pyspiel.load_game(
    #     f"turn_based_simultaneous_game(game={FLAGS.game})" if FLAGS.turn_based else FLAGS.game,
    #     {"filename": pyspiel.GameParameter(FLAGS.filename)},
    # )

    game = pyspiel.load_game(f"turn_based_simultaneous_game(game=clock_auction(filename={FLAGS.filename}))")

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

    nash_convs = []
    for i in range(int(FLAGS.iterations)):
      if FLAGS.solver == "mccfr":
        solver.run_iteration()  
      else:
        solver.evaluate_and_update_policy()
      nash_conv = pyspiel.nash_conv(game, solver.average_policy())
      nash_convs.append(nash_conv)
      print("Iteration {} nash_conv: {:.6f}".format(i, nash_conv))

    print("Persisting the model...")
    with open(MODEL_FILE_NAME.format(FLAGS.sampling), "wb") as f:
      pickle.dump(solver, f, pickle.HIGHEST_PROTOCOL)

    pd.Series(nash_convs).to_csv('nash_conv.csv')

    # print("Loading the model...")
    # with open(MODEL_FILE_NAME.format(FLAGS.sampling), "rb") as file:
    #   loaded_solver = pickle.load(file)

if __name__ == "__main__":
  app.run(main)
