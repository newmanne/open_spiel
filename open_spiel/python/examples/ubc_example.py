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

from absl import app
from absl import flags
import numpy as np

import open_spiel.python.games # Need this import to detect python games
import pyspiel
from open_spiel.python.observation import make_observation
from open_spiel.python.examples.ubc_utils import fix_seeds
from pprint import pprint

FLAGS = flags.FLAGS

flags.DEFINE_string("game", "python_clock_auction", "Name of the game")
flags.DEFINE_string("filename", 'parameters.json', "Filename with parameters")
flags.DEFINE_string("parameters", None, "String to be evaluated")
flags.DEFINE_bool("turn_based", False, "Convert simultaneous to turn based")
flags.DEFINE_bool("show_obs", False, "Show observation")

flags.DEFINE_integer("num_plays", 1, "Number of times to play")
flags.DEFINE_integer("seed", None, "Seed")

flags.DEFINE_string("load_state", None,
                    "A file containing a string to load a specific state")


def main(_):
  if FLAGS.seed:
    print(f"FIXING SEED {FLAGS.seed}")
    fix_seeds(FLAGS.seed)

  action_string = None

  print("Creating game: " + FLAGS.game)
  load_function = pyspiel.load_game if not FLAGS.turn_based else pyspiel.load_game_as_turn_based
  params = dict()
  if FLAGS.game == 'python_clock_auction':
      params['filename'] = FLAGS.filename
  elif FLAGS.parameters:
    params = eval(FLAGS.parameters)

  game = load_function(
      FLAGS.game,
      params,
  )

  if FLAGS.show_obs:
    observation = make_observation(game, params=dict(normalize=False))

  # Get a new state
  if FLAGS.load_state is not None:
    # Load a specific state
    state_string = ""
    with open(FLAGS.load_state, encoding="utf-8") as input_file:
      for line in input_file:
        state_string += line
    state_string = state_string.rstrip()
    print("Loading state:")
    print(state_string)
    print("")
    state = game.deserialize_state(state_string)
  else:
    state = game.new_initial_state()


  num_plays = 0
  while num_plays < FLAGS.num_plays:
    # TODO: I Know I'm breaking the above load_state
    state = game.new_initial_state()

    # Print the initial state
    print(str(state))

    while not state.is_terminal():
      # The state can be three different types: chance node,
      # simultaneous node, or decision node
      if state.is_chance_node():
        # Chance node: sample an outcome
        outcomes = state.chance_outcomes()
        if isinstance(outcomes, dict):
            print("Chance node, got " + str(outcomes['upper']) + " outcomes")
            action = np.random.randint(0, high=outcomes['upper'])
            print("Sampled outcome: ", state.action_to_string(state.current_player(), action))
        else:
            num_actions = len(outcomes)
            print("Chance node, got " + str(num_actions) + " outcomes")
            action_list, prob_list = zip(*outcomes)
            action = np.random.choice(action_list, p=prob_list)
            print("Sampled outcome: ", state.action_to_string(state.current_player(), action))
        state = state.child(action)

      elif state.is_simultaneous_node():
        # Simultaneous node: sample actions for all players.
        chosen_actions = [
            np.random.choice(state.legal_actions(pid))
            for pid in range(game.num_players())
        ]

        print("Chosen actions: ", [
            state.action_to_string(pid, action)
            for pid, action in enumerate(chosen_actions)
        ])
        state.apply_actions(chosen_actions)

      else:
        if FLAGS.show_obs:
          observation.set_from(state, player=state.current_player())
          print(f"OBSERVATION DICT P{state.current_player()}")
          pprint(observation.dict)
          # print(observation.tensor)

        # Decision node: sample action for the single current player
        action = np.random.choice(state.legal_actions(state.current_player()))
        action_string = state.action_to_string(state.current_player(), action)
        print("Player", state.current_player(), "randomly sampled action: ",
              action_string)
        state = state.child(action)

      print(str(state))

    # Game is now done. Print utilities for each player
    returns = state.returns()
    for pid in range(game.num_players()):
      print("Utility for player {} is {}".format(pid, returns[pid]))

    num_plays += 1

if __name__ == "__main__":
  app.run(main)
