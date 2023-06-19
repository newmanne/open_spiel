from ..pytorch.ppo import PPOAgent
import pyspiel
from absl import logging
import numpy as np
import pandas as pd
import itertools
import pulp
from pulp import LpStatus
import random
import string
from open_spiel.python.rl_agent import StepOutput
import torch
import humanize
import datetime as dt
import os
import json
import yaml
import shutil
from open_spiel.python.examples.ubc_math_utils import fast_choice
import signal

CONFIG_ROOT = '/apps/open_spiel/notebooks/configs'

CLOCK_AUCTION = 'clock_auction'

BR_DIR = 'best_responses'
CHECKPOINT_FOLDER = 'solving_checkpoints' # Don't use "checkpoints" because jupyter bug
EVAL_DIR = 'evaluations'

def setup_directory_structure(output_dir, warn_on_overwrite, database=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if warn_on_overwrite:
            raise ValueError("You are overwriting a folder!")
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    
    os.makedirs(os.path.join(output_dir, BR_DIR))
    os.makedirs(os.path.join(output_dir, EVAL_DIR))
    if not database:
        os.makedirs(os.path.join(output_dir, CHECKPOINT_FOLDER))

def fix_seeds(seed):
    logging.info(f"Setting numpy and torch seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True) # See https://github.com/pytorch/pytorch/issues/50469

def single_action_result(legal_actions, num_actions, as_output=False):
    probs = np.zeros(num_actions)
    action = legal_actions[0]
    probs[action] = 1.0
    if as_output:
        return StepOutput(action=action, probs=probs)
    return action, probs

def get_first_actionable_state(game, forced_types=None, player_id=None):
    state = game.new_initial_state()
    # Skip over chance nodes
    i = 0
    while state.current_player() < 0:
        state = state.child(0 if forced_types is None else forced_types[i]) # Let chance choose first outcome. We're assuming all moves are possible at starting prices for all players, that may not really be true though
        i += 1
    
    if player_id is not None:
        while state.current_player() != player_id:
            state = state.child(0)

    return state

def get_actions(game):
    state = get_first_actionable_state(game)
    action_dict = dict()
    for i in state.legal_actions(): 
        action_dict[i] = state.action_to_string(i)
    return action_dict

def load_game_config(game_name):
    game_config_path = os.path.join(os.environ['CLOCK_AUCTION_CONFIG_DIR'], game_name)
    with open(game_config_path, 'r') as f:
        game_config = json.load(f)
    return game_config

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
        return objective_from_lp(problem)
    except pulp.PulpSolverError as e:
        logging.warning(e)
        fail_lp(problem, save_if_failed) # Will reraise

def pretty_time(seconds):
    delta = dt.timedelta(seconds=seconds)
    return humanize.precisedelta(delta)

def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def apply_optional_overrides(args, argv, config):
    # Override any top-level yaml args with command line arguments
    for arg in vars(args):
        if f'--{arg}' in argv:
            name = arg
            value = getattr(args, arg)
            if name in config:
                logging.warning(f'Overriding {name} from command line')
                config[name] = value

    # These always get overridden from the command line
    config['seed'] = args.seed
    config['total_timesteps'] = args.total_timesteps
    config['use_wandb'] = args.use_wandb
    if hasattr(args, 'potential_function') and args.potential_function is not None:
        logging.warning(f'Overriding potential function from command line to {args.potential_function}')
        config['potential_function'] = args.potential_function
    if hasattr(args, 'scale_coef') and args.scale_coef is not None:
        logging.warning(f'Overriding scale coef from command line to {args.scale_coef}')
        config['scale_coef'] = args.scale_coef        


class UBCChanceEventSampler(object):
    """Default sampler for external chance events."""

    def __init__(self, seed=None) -> None:
        self.seed(seed)

    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed)

    def __call__(self, state):
        """Sample a chance event in the given state."""
        output = state.chance_outcomes()
        if isinstance(output, dict):
            return self._rng.randint(0, high=output['upper'])
        else:
            actions, probs = zip(*output)
        return fast_choice(actions, probs, rng=self._rng)


def series_to_quantiles(s: pd.Series):
    quantiles = []
    for quantile in np.arange(0, 1.01, 0.01):
        quantiles.append(s.quantile(quantile)) # TODO: Think about what interpolation method you want to use
    return quantiles

def config_path_from_config_name(config_name):
    return f'{CONFIG_ROOT}/{config_name}.yml'

def safe_config_name(name):
    return name.replace("/", "").replace('.yml', '')

def num_to_letter(i):
    '''Maps 0 to A, 1 to B etc.'''
    return chr(ord('@')+i+1)

def players_not_me(my_player_id, num_players):
    for i in range(num_players):
        if i == my_player_id:
            continue
        yield i

def factorial(k):
    return np.math.factorial(k)

def convert_seed_to_swaps(seed):
    """
    arguments:
    - seed: non-negative int
    
    returns: 
    - list (f_0, f_1, f_2, ..., f_k) such that \sum_{i=0}^k f_i * i! = seed
    """
    ret = [0]
    while seed > 0:
        divisor = len(ret) + 1
        ret.append(seed % divisor)
        seed //= divisor
    return ret

def permute_array(arr, seed):
    """
    Permute an array into a canonical permutation.

    Arguments:
    - arr: list
    - seed: int in range [0, len(arr)! - 1]
    
    Returns:
    - A permutation of arr
    """

    if len(arr) == 0:
        return arr

    if seed >= factorial(len(arr)):
        raise ValueError(f'seed {seed} too large for array of length {len(arr)}') 
    
    # Fisher-Yates shuffle
    permuted_arr = [a for a in arr]
    swap_indices = convert_seed_to_swaps(seed)
    swap_indices = swap_indices + [0] * (len(permuted_arr) - len(swap_indices))
    
    for i, swap_idx in enumerate(swap_indices):
        permuted_arr[i], permuted_arr[swap_idx] = permuted_arr[swap_idx], permuted_arr[i]
    return permuted_arr
    
def between_first(s, a, b):
    '''Returns the substring between the first occurrence of characters a and b in the string s'''
    start_index = s.find(a) + len(a) if a in s else -1
    end_index = s.find(b, start_index) if start_index != -1 else -1

    if start_index != -1 and end_index != -1:
        return s[start_index:end_index]
    else:
        return ""


class SignalTimeout(ValueError):
    pass

def signal_handler(*args):
    raise SignalTimeout()

def time_bounded_run(t, f, *args, **kwargs):
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(t)
    try:
        return True, f(*args, **kwargs)
    except SignalTimeout as ex:
        return False, None
    finally:
        signal.alarm(0) # Clear alarm
