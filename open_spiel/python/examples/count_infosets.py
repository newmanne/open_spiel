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

"""Python spiel example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from absl import app
from absl import flags
import numpy as np

import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import cfr, outcome_sampling_mccfr, expected_game_score, exploitability, get_all_states_with_policy


FLAGS = flags.FLAGS

flags.DEFINE_string("game", "clock_auction", "Name of the game")
flags.DEFINE_string("filename", 'parameters.json', "Filename with parameters")
flags.DEFINE_bool("turn_based", True, "Convert simultaneous to turn based")


def main(_):
  action_string = None

  print("Creating game: " + FLAGS.game)
  load_function = pyspiel.load_game if not FLAGS.turn_based else pyspiel.load_game_as_turn_based
  params = dict()
  if FLAGS.game == 'clock_auction':
      params['filename'] = pyspiel.GameParameter(FLAGS.filename)

  game = load_function(
      FLAGS.game,
      params,
  )

  all_states = get_all_states_with_policy.get_all_info_states_with_policy(
      game,
      depth_limit=-1,
      include_terminals=False,
  )

  print(len(all_states))

if __name__ == "__main__":
  app.run(main)
