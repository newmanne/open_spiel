
from absl import app
from absl import flags

import numpy as np
import pandas as pd
import pyspiel
import os
import json
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_integer("player_number", 0, "What player you will control")
flags.DEFINE_string("filename", 'parameters.json', "Filename with parameters")
flags.DEFINE_string("solver", 'model.pkl', "Path to model you will play against")
flags.DEFINE_bool("show_hidden", False, "See information you shouldn't")
flags.DEFINE_string("game", "clock_auction", "Name of the game")


def main(_):

    with open(FLAGS.solver, "rb") as f:
        solver = pickle.load(f)

    policy = solver.average_policy()

    # TODO: It seems silly that I can't get the game from the solver. Doesn't it have a reference to it I could expose?
    params = dict()
    params['filename'] = pyspiel.GameParameter(FLAGS.filename)
    
    game = pyspiel.load_game_as_turn_based(
        FLAGS.game,
        params,
    )

    state = game.new_initial_state()

    print(f"You are player {FLAGS.player_number}")
    # Print the initial state
    print(str(state))

    while not state.is_terminal():
        # The state can be three different types: chance node,
        # simultaneous node, or decision node
        if state.is_chance_node():
            # Chance node: sample an outcome
            outcomes = state.chance_outcomes()
            num_actions = len(outcomes)
            # print("Chance node, got " + str(num_actions) + " outcomes")
            action_list, prob_list = zip(*outcomes)
            action = np.random.choice(action_list, p=prob_list)
            # TODO: I shouldn't be able to see my oppononent's private info
            if FLAGS.show_hidden:
                print("Sampled outcome:", state.action_to_string(state.current_player(), action))
            state.apply_action(action)

        elif state.is_simultaneous_node():
            raise
    #       # TODO: Don't think we have these nodes in our game
    #       # Simultaneous node: sample actions for CPU players.
    #         chosen_actions = [
    #           random.choice(state.legal_actions(pid))
    #           for pid in range(game.num_players())
    #         ]
    #         print("Chosen actions: ", [
    #           state.action_to_string(pid, action)
    #           for pid, action in enumerate(chosen_actions)
    #         ])
    #         state.apply_actions(chosen_actions)

        else:
            # Decision node: sample action for the single current player
            if state.current_player() == FLAGS.player_number:
                print(state.information_state_string())
                choices = state.legal_actions(state.current_player())
                valid_move = False
                while not valid_move:
                    action = int(input(f"Enter your bid: {choices} "))
                    if action not in choices:
                        print("Please enter a valid action")
                    else:
                        valid_move = True

                    
            else:
                # CPU acts according to policy
                a_and_p = policy.get_state_policy(state)
                a = [x[0] for x in a_and_p]
                p = [x[1] for x in a_and_p]
                action = np.random.choice(a, p=p)
            if state.current_player() == FLAGS.player_number or FLAGS.show_hidden:
                action_string = state.action_to_string(state.current_player(), action)
                print(f"Player {state.current_player()} played action: {action_string}")
            state.apply_action(action)

    print()
    print("GAME OVER")
    print(str(state))
    # Game is now done. Print utilities for each player
    returns = state.returns()
    for pid in range(game.num_players()):
        print("Utility for player {} is {}".format(pid, returns[pid]))


if __name__ == "__main__":
    app.run(main)