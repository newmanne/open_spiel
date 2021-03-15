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

"""CFR seq eq example."""

from absl import app
from absl import flags

import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "kuhn_poker", "Name of the game")
flags.DEFINE_integer("num_iters", 10000000, "Number of players")
flags.DEFINE_integer("report_freq", 10000, "Report frequency")
flags.DEFINE_float("initial_eps", 0.01, "Initial epsilon")
flags.DEFINE_float("decay_factor", 0.99, "Decay factor")
flags.DEFINE_integer("decay_freq", 1000, "Decay frequency")
flags.DEFINE_float("min_eps", 1e-6, "Minimum epsilon")

COUNTEREXAMPLE_EFG = """EFG 2 R "Counterexample by Rich Gibson" { "Player 1" "Player 2" } ""

p "P2" 2 1 "P2" { "l" "r" } 0
  t "l" 1 "Outcome l" { 0.0 0.0 }
  p "P1" 1 1 "P1" { "l" "m" "r" } 0
    t "rl" 2 "Outcome rl" { 0.0 0.0 }
    t "rm" 3 "Outcome rl" { 9.0 -9.0 }
    t "rr" 4 "Outcome rr" { 10.0 -10.0 }"""

GUESS_THE_ACE_EFG = """EFG 2 R "Guess the Ace game from Miltersen & Sorensen '06" { "Player 1" "Player 2" } ""

c "ROOT" 1 "ROOT" { "OOT" 51/52 "ASOT" 1/52 } 0
  p "" 1 1 "P1" { "p" "dnp" } 0
    p "" 2 1 "P2" { "go" "ga" } 0
      t "OOT-p-go" 1 "Outcome OOT-p-go" { -1000.0 1000.0 }
      t "OOT-p-ga" 2 "Outcome OOT-p-ga" { 0.0 0.0 }
    t "OOT-dnp" 3 "Outcome OOT-dnp" { 0.0 0.0 }
  p "" 1 1 "P1" { "p" "dnp" } 0
    p "" 2 1 "P2" { "go" "ga" } 0
      t "ASOT-p-go" 4 "Outcome ASOT-p-go" { 0.0 0.0 }
      t "ASOT-p-ga" 5 "Outcome ASOT-p-ga" { -1000.0 1000.0 }
    t "ASOT-dnp" 6 "Outcome ASOT-dnp" { 0.0 0.0 }"""


def main(_):
  game = pyspiel.load_game(FLAGS.game)
  # game = pyspiel.load_efg_game(COUNTEREXAMPLE_EFG)
  # game = pyspiel.load_efg_game(GUESS_THE_ACE_EFG)

  initial_eps = FLAGS.initial_eps
  decay_factor = FLAGS.decay_factor
  decay_freq = FLAGS.decay_freq
  min_eps = FLAGS.min_eps
  eps_solver = pyspiel.EpsilonCFRSolver(game, initial_eps)
  for i in range(FLAGS.num_iters):
    eps_solver.evaluate_and_update_policy()
    if i > 0 and i % decay_freq == 0:
      eps_solver.set_epsilon(max(eps_solver.epsilon() * decay_factor, min_eps))
    if i % FLAGS.report_freq == 0:
      avg_policy = eps_solver.tabular_average_policy()
      nash_conv, max_qv_diff = pyspiel.nash_conv_with_eps(game, avg_policy)
      print(f"Iter {i} Eps: {eps_solver.epsilon()} nash_conv: {nash_conv} " +
            f"max_qv_diff: {max_qv_diff}")


if __name__ == "__main__":
  app.run(main)
