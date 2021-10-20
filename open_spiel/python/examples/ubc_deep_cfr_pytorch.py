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

"""Python Deep CFR example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app, flags, logging
import time

from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
import pyspiel
from open_spiel.python.pytorch import deep_cfr

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 400, "Number of iterations")
flags.DEFINE_integer("num_traversals", 40, "Number of traversals/games")
flags.DEFINE_string("game_name", "clock_auction", "Name of the game")
flags.DEFINE_bool("turn_based", True, "Convert simultaneous to turn based")
flags.DEFINE_string("filename", 'parameters.json', "Filename with parameters")

def main(unused_argv):
  start_time = time.time()
  logging.info("Loading %s", FLAGS.game_name)

  # LOAD GAME
  load_function = pyspiel.load_game if not FLAGS.turn_based else pyspiel.load_game_as_turn_based
  params = dict()
  if FLAGS.game_name == 'clock_auction':
      params['filename'] = FLAGS.filename

  game = load_function(
      FLAGS.game_name,
      params,
    )

  deep_cfr_solver = deep_cfr.DeepCFRSolver(
      game,
      policy_network_layers=(32, 32),
      advantage_network_layers=(16, 16),
      num_iterations=FLAGS.num_iterations,
      num_traversals=FLAGS.num_traversals,
      learning_rate=1e-3,
      batch_size_advantage=None,
      batch_size_strategy=None,
      memory_capacity=int(1e7))

  _, advantage_losses, policy_loss = deep_cfr_solver.solve()
  for player, losses in advantage_losses.items():
    logging.info("Advantage for player %d: %s", player,
                 losses[:2] + ["..."] + losses[-2:])
    logging.info("Advantage Buffer Size for player %s: '%s'", player,
                 len(deep_cfr_solver.advantage_buffers[player]))
  logging.info("Strategy Buffer Size: '%s'",
               len(deep_cfr_solver.strategy_buffer))
  logging.info("Final policy loss: '%s'", policy_loss)

  average_policy = policy.tabular_policy_from_callable(
      game, deep_cfr_solver.action_probabilities)
  pyspiel_policy = policy.python_policy_to_pyspiel_policy(average_policy)
  conv = pyspiel.nash_conv(game, pyspiel_policy)
  logging.info("Deep CFR in '%s' - NashConv: %s", FLAGS.game_name, conv)

  average_policy_values = expected_game_score.policy_value(
      game.new_initial_state(), [average_policy] * 2)
  logging.info("Computed player 0 value: %.2f",
               average_policy_values[0])
  logging.info("Computed player 1 value: %.2f",
               average_policy_values[1])


if __name__ == "__main__":
  app.run(main)
