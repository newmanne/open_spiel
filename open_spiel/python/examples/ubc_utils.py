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

CONFIG_ROOT = '/apps/open_spiel/notebooks/configs'

CLOCK_AUCTION = 'clock_auction'

BR_DIR = 'best_responses'
CHECKPOINT_FOLDER = 'solving_checkpoints' # Don't use "checkpoints" because jupyter bug
EVAL_DIR = 'evaluations'

SUBMITTED_DEMAND_INDEX = 0
PROCESSED_DEMAND_INDEX = 1
AGG_DEMAND_INDEX = 2
POSTED_PRICE_INDEX = 3

FEATURES_PER_PRODUCT = 4

# TODO: I hate this... This is super fragile

def turn_based_size(num_players):
    return 2 * num_players

def round_index(num_players):
    return turn_based_size(num_players)

def current_round(num_players, information_state_tensor): # 1-indexed
    return int(information_state_tensor[round_index(num_players)])

def prefix_size(num_types):
    return num_types

def handcrafted_size(num_actions, num_products):
    return 2 * num_actions + 3 + 3 * num_products

def clock_price_index(num_players, num_actions):
    # Round, SOR/Clock Profit
    return round_index(num_players) + 2 * num_actions + 2 + 1

def clock_profit_index(num_players, num_actions):
    return round_index(num_players) + num_actions + 1

def sor_profit_index(num_players):
    return round_index(num_players) + 1

def activity_index(num_players, num_actions):
    return clock_profit_index(num_players, num_actions) + num_actions

def sor_exposure_index(num_players, num_actions):
    return activity_index(num_players, num_actions) + 1

def recurrent_index(num_players, num_actions, num_products, num_types):
    return turn_based_size(num_players) + handcrafted_size(num_actions, num_products) + prefix_size(num_types)

def prefix_index(num_players, num_actions, num_products):
    return turn_based_size(num_players) + handcrafted_size(num_actions, num_products)

def get_player_type(num_players, num_actions, num_products, max_types, information_state_tensor):
    index = prefix_index(num_players, num_actions, num_products)
    return int(np.nonzero(information_state_tensor[index:index + max_types])[0][0])

def recurrent_round_size(num_products):
    return FEATURES_PER_PRODUCT * num_products

def round_frame_index(num_players, num_actions, num_products, r, num_types):
    r_index = recurrent_index(num_players, num_actions, num_products, num_types)
    return r_index + recurrent_round_size(num_products) * (r - 1)

def round_frame(num_players, num_actions, num_products, r, information_state_tensor, num_types):
    index = round_frame_index(num_players, num_actions, num_products, r, num_types)
    return information_state_tensor[index: index + recurrent_round_size(num_products)]

def current_round_frame(num_players, num_actions, num_products, information_state_tensor, num_types):
    r = current_round(num_players, information_state_tensor)
    return round_frame(num_players, num_actions, num_products, r, information_state_tensor, num_types)

def payment_and_allocation(num_players, num_actions, num_products, information_state_tensor, num_types):
    frame = current_round_frame(num_players, num_actions, num_products, information_state_tensor, num_types)
    allocation = frame[PROCESSED_DEMAND_INDEX * num_products : (PROCESSED_DEMAND_INDEX + 1) * num_products]
    prices = frame[POSTED_PRICE_INDEX * num_products : (POSTED_PRICE_INDEX + 1) * num_products]
    return np.array(prices) @ np.array(allocation), allocation

def parse_current_round_frame(num_players, num_actions, num_products, information_state_tensor, num_types):
    frame = current_round_frame(num_players, num_actions, num_products, information_state_tensor, num_types)
    allocation = frame[PROCESSED_DEMAND_INDEX * num_products : (PROCESSED_DEMAND_INDEX + 1) * num_products]
    agg_demand = frame[AGG_DEMAND_INDEX * num_products : (AGG_DEMAND_INDEX + 1) * num_products]
    prices = frame[POSTED_PRICE_INDEX * num_products : (POSTED_PRICE_INDEX + 1) * num_products]
    if len(prices) != num_products:
        raise ValueError("Wrong size for prices!")

    return dict(allocation=allocation, agg_demand=agg_demand, posted_prices=prices)

def num_increments(price, increment, starting_price):
    num_increments = (np.log(price) - np.log(starting_price)) / np.log(increment)
    return int(np.round(num_increments)) - 1

def action_to_bundles(licenses):
    bids = []
    for n in licenses:
        b = []
        for i in range(n + 1):
            b.append(i)
        bids.append(b)
    actions = list(itertools.product(*bids))
    return {i: a for i, a in enumerate(actions)}


def fix_seeds(seed):
    logging.info(f"Setting numpy and torch seed to {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def make_dqn_kwargs_from_config(config, game_config=None, player_id=None, include_nfsp=True):
    dqn_kwargs = {
      "replay_buffer_capacity": config['replay_buffer_capacity'],
      "epsilon_decay_duration": config['num_training_episodes'],
      "epsilon_start": config['epsilon_start'],
      "epsilon_end": config['epsilon_end'],
      "update_target_network_every": config['update_target_network_every'],
      "loss_str": config['loss_str'],
      "double_dqn": config['double_dqn'],
      "batch_size": config['batch_size'],
      "learning_rate": config['rl_learning_rate'],
      "learn_every": config['learn_every'],
      "min_buffer_size_to_learn": config['min_buffer_size_to_learn'],
      "optimizer_str": config['optimizer_str'],
      "device": config['device'],
    }
    if not include_nfsp:
        del dqn_kwargs['batch_size']
        del dqn_kwargs['min_buffer_size_to_learn']
        del dqn_kwargs['learn_every']
        del dqn_kwargs['optimizer_str']
        del dqn_kwargs['device']

    if game_config is not None and player_id is not None:
        dqn_kwargs['lower_bound_utility'], dqn_kwargs['upper_bound_utility'] = clock_auction_bounds(game_config, player_id)
        dqn_kwargs['game_config'] = game_config
    return dqn_kwargs



def single_action_result(legal_actions, num_actions, as_output=False):
    probs = np.zeros(num_actions)
    action = legal_actions[0]
    probs[action] = 1.0
    if as_output:
        return StepOutput(action=action, probs=probs)
    return action, probs


def get_first_actionable_state(game, forced_types=None):
    state = game.new_initial_state()
    # Skip over chance nodes
    i = 0
    while state.current_player() < 0:
        state = state.child(0 if forced_types is None else forced_types[i]) # Let chance choose first outcome. We're assuming all moves are possible at starting prices for all players, that may not really be true though
        i += 1
    return state


def get_actions(game):
    state = get_first_actionable_state(game)
    action_dict = dict()
    for i in state.legal_actions(): 
        action_dict[i] = state.action_to_string(i)
    return action_dict

def check_on_q_values(agent, game, state=None, infostate_tensor=None, legal_actions=None, time_step=None, return_raw_q_values=False):
    q_network = agent._q_network

    if time_step is not None:
        legal_actions = time_step.observations["legal_actions"][agent.player_id]
        it = time_step.observations["info_state"][agent.player_id]
    elif infostate_tensor is not None:
        legal_actions = legal_actions
        it = infostate_tensor
    else:
        # Extract from state
        if state is None:
            # TODO: assuming player_id on agent is 0 here, could be smarter
            state = get_first_actionable_state(game)
        legal_actions = state.legal_actions()
        it = state.information_state_tensor()

    info_state = q_network.prep_batch([q_network.reshape_infostate(it)]).to(agent._device)
    q_values = q_network(info_state).cpu().detach()[0]
    if return_raw_q_values:
        return q_values

    legal_q_values = q_values[legal_actions]
    action_dict = get_actions(game)
    return {s: q for s, q in zip(action_dict.values(), legal_q_values)}  

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

def smart_load_clock_auction(game_name):
    return smart_load_sequential_game(CLOCK_AUCTION, dict(filename=game_name))

def load_game_config(game_name):
    game_config_path = os.path.join(os.environ['CLOCK_AUCTION_CONFIG_DIR'], game_name)
    with open(game_config_path, 'r') as f:
        game_config = json.load(f)
    return game_config

def max_opponent_spend(game_config, player_id):
    max_opp_budgets = []
    for p, opponent in enumerate(game_config['players']):
        if p == player_id:
            continue
        p_budgets = []
        for t2 in opponent['type']:
            p_budgets.append(t2['budget'])
        max_opp_budgets.append(max(p_budgets))
    return np.array(max_opp_budgets).sum()

def max_budget(game_config):
    for p in game_config['players']:
        p_budgets = []
        for t in p['type']:
            p_budgets.append(t['budget'])
    return max(p_budgets)

def all_activity(game_config):
    return np.array(game_config['licenses']) @ np.array(game_config['activity'])


def my_max_pricing_bonus(game_config, player_id):
    pricing_bonuses = []
    for t in game_config['players'][player_id]['type']:
        pricing_bonuses.append(t.get('pricing_bonus', 0))
    return max(pricing_bonuses)

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

    # If pricing bonus is on, you can receive additional points for pricing opponents (up to their budgets)
    max_utility += my_max_pricing_bonus(game_config, player_id) * max_opponent_spend(game_config, player_id)


    # MIN UTILITY
    # What if you spent your entire budget and got nothing? (A tighter not implemented bound: if you got the single worst item for you, since you must be paying for something)
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


# See https://stackoverflow.com/a/58101985. Much faster than np.random.choice, at least for our current version of numpy and our distribution over the arguments
def fast_choice(options, probs):
    x = np.random.rand()
    cum = 0
    for i,p in enumerate(probs):
        cum += p
        if x < cum:
            break
    return options[i]


def pretty_time(seconds):
    delta = dt.timedelta(seconds=seconds)
    return humanize.precisedelta(delta)


def default_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def add_optional_overrides(parser):
    # Optional Overrides
    parser.add_argument('--replay_buffer_capacity', type=int, default=None)
    parser.add_argument('--reservoir_buffer_capacity', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--rl_learning_rate', type=float, default=None)
    parser.add_argument('--sl_learning_rate', type=float, default=None)
    parser.add_argument('--min_buffer_size_to_learn', type=int, default=None)
    parser.add_argument('--learn_every', type=int, default=None)
    parser.add_argument('--optimizer_str', type=str, default=None)
    parser.add_argument('--epsilon_start', type=float, default=None)
    parser.add_argument('--epsilon_end', type=float, default=None)

def apply_optional_overrides(args, argv, config):
    # Override any top-level yaml args with command line arguments
    for arg in vars(args):
        if f'--{arg}' in argv:
            name = arg
            value = getattr(args, arg)
            if name in config:
                logging.warning(f'Overriding {name} from command line')
                config[name] = value

    # This one argument we special case: I don't consider it part of the config per se, but it's too annoying to refactor out and it impacts the epsilon decay schedule
    config['num_training_episodes'] = args.num_training_episodes

class UBCChanceEventSampler(object):
    """Default sampler for external chance events."""

    def __init__(self, deterministic_types=None) -> None:
        self.deterministic_types = deterministic_types

    def __call__(self, state, reset=False):
        """Sample a chance event in the given state."""
        if reset and self.deterministic_types is not None:
            deterministic_action = self.deterministic_types[len(state.history())]
            if deterministic_action is not None:
                return deterministic_action

        actions, probs = zip(*state.chance_outcomes())
        return fast_choice(actions, probs)


def series_to_quantiles(s: pd.Series):
    quantiles = []
    for quantile in np.arange(0, 1.01, 0.01):
        quantiles.append(s.quantile(quantile)) # TODO: Think about what interpolation method you want to use
    return quantiles

def game_spec(game, game_config):
    num_players = game.num_players()
    num_actions = game.num_distinct_actions()
    num_products = len(game_config['licenses'])
    return num_players, num_actions, num_products

def config_path_from_config_name(config_name):
    return f'{CONFIG_ROOT}/{config_name}.yml'

def read_config(config_name):
    config_file = config_path_from_config_name(config_name)
    logging.info(f"Reading config from {config_file}")
    with open(config_file, 'rb') as fh: 
        config = yaml.load(fh, Loader=yaml.FullLoader)

    # DEFAULTS GO HERE
    config['replay_buffer_capacity'] = config.get('replay_buffer_capacity', 50_000)
    config['update_target_network_every'] = config.get('update_target_network_every', 10_000)
    config['loss_str'] = config.get('loss_str', 'mse')
    config['sl_loss_str'] = config.get('sl_loss_str', 'cross_entropy')
    config['double_dqn'] = config.get('double_dqn', True)
    config['device'] = config.get('device', default_device())
    config['reservoir_buffer_capacity'] = config.get('reservoir_buffer_capacity', 2_000_000)
    config['anticipatory_param'] = config.get('anticipatory_param', 0.1)
    config['sl_learning_rate'] = config.get('sl_learning_rate', 0.01)
    config['rl_learning_rate'] = config.get('rl_learning_rate', 0.01)
    config['batch_size'] = config.get('batch_size', 256)
    config['min_buffer_size_to_learn'] = config.get('min_buffer_size_to_learn', 1000)
    config['learn_every'] = config.get('learn_every', 64)
    config['optimizer_str'] = config.get('optimizer_str', 'sgd')
    config['add_explore_transitions'] = config.get('add_explore_transitions', False)

    return config

def safe_config_name(name):
    return name.replace("/", "").replace('.yml', '')

def num_to_letter(i):
    '''Maps 0 to A, 1 to B etc.'''
    return chr(ord('@')+i+1)

def make_normalizer_for_game(game, game_config):
    # Turn based crap (2 * num_players)
    # Round
    # SoR profit (num_actions)
    # Clock profit (num_actions)
    # Activity
    # SoR exposure
    # Clock prices (num_products)
    # Current holdings
    # Agg demand
    # 1 hot type (num_types)
    # While loop over rounds:
        # Submitted demand
        # Processed demand
        # Agg demand
        # Posted price

    game_max_budget = max_budget(game_config)
    num_actions = game.num_distinct_actions()
    max_activity = all_activity(game_config)
    num_types = max_num_types(game_config)
    num_products = len(game_config['licenses'])
    num_players = game.num_players()
    max_rounds = game_config.get('max_rounds', 100)

    normalizer = ([1] * (2 * num_players)) + \
        [max_rounds] + \
        ([game_max_budget] * (num_actions * 2)) + \
        [max_activity] + \
        [game_max_budget] + \
        [game_max_budget] * (num_products) + \
        game_config['licenses'] + \
        (np.array(game_config['licenses']) * num_players).tolist() + \
        [1] * num_types
    for _ in range(max_rounds):
        normalizer += game_config['licenses']
        normalizer += game_config['licenses']
        normalizer += (np.array(game_config['licenses']) * num_players).tolist()
        normalizer += [game_max_budget] * num_products

    return torch.tensor(normalizer, dtype=torch.float32)

def players_not_me(my_player_id, num_players):
    for i in range(num_players):
        if i == my_player_id:
            continue
        yield i

def max_num_types(game_config):
    n_types = []
    for player_id in range(len(game_config['players'])):
        n_types.append(len(game_config['players'][player_id]['type']))
    return max(n_types)

def type_from_index(game_config, player_id, type_index):
    return game_config['players'][player_id]['type'][type_index]