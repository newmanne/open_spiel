import pyspiel
from absl import logging
import numpy as np
import pandas as pd
import itertools
import pulp
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, LpBinary, lpSum, lpDot, LpMaximize, LpInteger
import random
import string
from open_spiel.python.rl_agent import StepOutput


CLOCK_AUCTION = 'clock_auction'
FEATURES_PER_PRODUCT = 4

def action_to_bundles(licenses):
    bids = []
    for n in licenses:
        b = []
        for i in range(n + 1):
            b.append(i)
        bids.append(b)
    actions = list(itertools.product(*bids))
    return {i: a for i, a in enumerate(actions)}



def single_action_result(legal_actions, num_actions, as_output=False):
    probs = np.zeros(num_actions)
    action = legal_actions[0]
    probs[action] = 1.0
    if as_output:
        return StepOutput(action=action, probs=probs)
    return action, probs


def get_actions(game):
    state = game.new_initial_state()
    # Skip over chance nodes
    while state.current_player() < 0:
        state = state.child(0) # Let chance choose first outcome. We're assuming all moves are possible at starting prices for all players, that may not really be true though

    # Now we are at a player state
    action_dict = dict()
    for i in range(len(state.legal_actions())): 
        action_dict[i] = state.action_to_string(i)
    return action_dict


def smart_load_sequential_game(game_name, game_parameters=dict()):
    # Stupid special case our own game because loading it twice takes time
    if game_name == CLOCK_AUCTION:
        return pyspiel.load_game_as_turn_based(game_name, game_parameters)

    game = pyspiel.load_game(game_name, game_parameters)
    game_type = game.get_type()

    if game_type.dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
        logging.warn("%s is not turn-based. Trying to reload game as turn-based.", game_name)
        game = pyspiel.load_game_as_turn_based(game_name, game_parameters)

    game_type = game.get_type()

    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError("Game must be sequential, not {}".format(game_type.dynamics))

    return game


def clock_auction_bounds(game_config, player_id):
    # MAX UTILITY
    player_id = 0
    p_open = np.array(game_config['opening_price'])
    supply = np.array(game_config['licenses'])
    bounds = []
    for t in game_config['players'][player_id]['type']:
        # What if you won the whole supply at opening prices?
        bound = (np.array(t['value']) - p_open).clip(0) @ supply
        bounds.append(bound)
    max_utility = max(bounds)
    # MIN UTILITY
    # What if you spent your entire budget and got nothing? (A tighter not implemented bound: if you got the single worst item for you)
    min_utility = -min([t['budget'] for t in game_config['players'][player_id]['type']])
    return min_utility, max_utility


def solve(problem):
    return problem.solve(solver=pulp.CPLEX_PY(msg=0, gapRel=0))
    # return problem.solve(solver=pulp.CPLEX_CMD(msg=0, options=[f'set mip tolerances mipgap 0']))
    # return problem.solve(solver=pulp.CPLEX_CMD(msg=0, options=[f'set mip tolerances mipgap 0'], keepFiles = 1))

def objective_from_lp(problem):
    return np.round(problem.objective.value())

def random_string(k):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=k))

def fail_lp(problem, save_lp, status=None):
    error_message = f"Couldn't solve allocation problem optimally. Status was {status}."
    if save_lp:
        fname = ''.join(random_string(10))
        # TODO: Pickle seems to no longer work TypeError: can't pickle SwigPyObject objects
        # with open(f'{fname}.pkl', 'wb') as f:
        #     pickle.dump(problem, f)
        problem.writeLP(f'{fname}.lp')
        error_message += f'Saved LP to {fname}.lp.'
    raise ValueError(error_message)

def pulp_solve(problem, solve_function=solve, save_if_failed=True):
    try:
        status = LpStatus[solve_function(problem)]
        if status != "Optimal":
            fail_lp(problem, save_if_failed, status=status)
    except pulp.PulpSolverError as e:
        logging.warning(e)
        fail_lp(problem, save_if_failed)
    return objective_from_lp(problem)
