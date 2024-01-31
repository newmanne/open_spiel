import numpy as np
import pandas as pd
from collections import Counter
from open_spiel.python.examples.ubc_utils import *
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, LpBinary, lpSum, lpDot, LpMaximize, LpInteger, value
from tqdm import tqdm
import scipy.stats
from collections import defaultdict
import pickle
from compress_pickle import dumps, loads
from open_spiel.python.examples.straightforward_agent import StraightforwardAgent
from open_spiel.python.examples.ubc_decorators import TakeSingleActionDecorator, TremblingAgentDecorator, ModalAgentDecorator
from open_spiel.python.examples.ppo_utils import make_env_and_policy
from open_spiel.python.games.clock_auction_base import InformationPolicy, ActivityPolicy, UndersellPolicy, TiebreakingPolicy
from open_spiel.python.algorithms.exploitability import nash_conv

def type_combos(game):
    types = [game.auction_params.player_types[player] for player in range(game.num_players())]
    for player_type in types:
        for i, t in enumerate(player_type):
            t['index'] = i
    type_combos = list(itertools.product(*types))
    # type_combos = list(map(lambda x: (x[0]['index'], x[1]['index']), type_combos))
    return type_combos

def analyze_samples(samples, game, restrict_to_wandb=False):
    players = list(samples['raw_rewards'].keys())
    record = dict()
    welfares = np.zeros(len(samples['raw_rewards'][players[0]]))
    revenues = np.zeros(len(samples['raw_rewards'][players[0]]))
    record['total_entropy'] = 0
    record['avg_entropy'] = 0
    record['avg_mode_probabilities'] = 0
    n_players = len(samples['raw_rewards'].keys())
    unsold = np.array(game.auction_params.licenses, dtype=np.float64)

    for player in players:
        # player = str(player)
        rewards = pd.Series(samples['raw_rewards'][player])
        payments = pd.Series(samples['payments'][player])
        record[f'p{player}_expected_exposure'] = np.nan_to_num(rewards[rewards < 0].mean())
        record[f'p{player}_exposure_frac'] = np.nan_to_num((rewards < 0).mean())
        record[f'p{player}_utility'] = rewards.mean()
        record[f'p{player}_payment'] = payments.mean()
        record[f'p{player}_total_entropies'] = np.array(samples['total_entropies'][player]).mean()
        record['total_entropy'] += record[f'p{player}_total_entropies']
        record[f'p{player}_avg_entropies'] = np.mean(np.array(samples['total_entropies'][player]) / np.array(samples['auction_lengths']))
        record['avg_entropy'] += record[f'p{player}_avg_entropies'] / n_players
        record[f'p{player}_avg_mode_probabilities'] = np.array(samples['avg_mode_probabilities'][player]).mean()
        record['avg_mode_probabilities'] += record[f'p{player}_avg_mode_probabilities'] / n_players

        # TODO: Talk about efficiency of the allocation, NOT welfare. Also, does welfare still make sense with pricing bonuses?
        welfares += rewards + payments # Welfare: what you would get if you got it for free
        revenues += payments

        unsold -= np.array(samples['allocations'][player]).mean(axis=0)
    
    s = []
    for player in players:
        s.append(np.array(samples['raw_rewards'][player]) < 0)
    # Take the logical OR of the series in s together logically and compute the mean. Make sure this works it 
    record['exposure_frac'] = np.nan_to_num(np.any(np.stack(s), axis=0).mean())
    
    record['total_welfare'] = welfares.mean()
    record['total_revenue'] = revenues.mean()
    record['auction_lengths'] = np.array(samples['auction_lengths']).mean()
    record['num_lotteries'] = np.array(samples['num_lotteries']).mean()
    record['p_lottery'] = (np.array(samples['num_lotteries']) > 0).mean()
    record['unsold_activity'] = unsold @ game.auction_params.activity
    record['unsold_licenses'] = unsold.sum()
    
    # record['unsold_sor_revenue'] = unsold @ game.

    if not restrict_to_wandb:
        # only include common allocations when analyzing in, e.g., notebooks
        arr = np.array(samples['allocations'][players[0]]).astype(int) # TODO: only player 0's allocation.
        c = Counter(tuple(map(tuple, arr)))
        record['common_allocations'] = c.most_common(5)
        record['unsold'] = unsold
    return record


def allocation_scorer(game, normalizer_dict):
    def scorer(alloc, types):
        score = sum((game.auction_params.player_types[i][types[i]['bidder'].value_for_package](alloc[i]) for i in range(len(alloc))))
        return score, score / normalizer_dict[types]
        
    return scorer

def efficient_allocation(game, factor_in_opening_prices=True, verbose=True):
    # Parse all type combos
    types = [game.auction_params.player_types[player] for player in range(game.num_players())]
    for player_type in types:
        for i, t in enumerate(player_type):
            t['index'] = i

    type_combos = list(itertools.product(*types))

    records = []
    for combo in tqdm(type_combos, disable=not verbose):
        type_prob = np.product([t['prob'] for t in combo])
        score, allocation = efficient_allocation_from_types(game, combo, factor_in_opening_prices=factor_in_opening_prices)
        combo_index = tuple(t['index'] for t in combo)
        record = dict(prob=type_prob, score=score, allocation=tuple(tuple(a) for a in allocation), combo=combo_index)
        records.append(record)

    df = pd.DataFrame.from_records(records)
    combo_to_score = df[['combo', 'score']].set_index('combo')['score'].to_dict()
    scorer = allocation_scorer(game, combo_to_score)
    return df, combo_to_score, scorer

def efficient_allocation_from_types(game, types, factor_in_opening_prices=True):
    '''factor_in_opening_prices says to only consider feasible allocations at the opening prices'''
    params = game.auction_params
    p_open = params.opening_prices
    num_players = game.num_players()
    
    profits = []
    for player in range(num_players):
        profits.append(
            types[player]['bidder'].get_profits(p_open).tolist() if factor_in_opening_prices else np.maximum(0, types[player]['bidder'].get_values().tolist())
        )

    return efficient_allocation_from_profits(game, profits)

def efficient_allocation_avg_type(game, factor_in_opening_prices=True):
    '''factor_in_opening_prices says to only consider feasible allocations at the opening prices'''
    types = [game.auction_params.player_types[player] for player in range(game.num_players())]
    params = game.auction_params
    p_open = params.opening_prices

    profit_fn = lambda t: t['bidder'].get_profits(p_open) if factor_in_opening_prices else t['bidder'].get_values()
    avg_profits = []
    for player in range(len(types)):
        type_profits = np.array([profit_fn(t).tolist() for t in types[player]]) # (num_types, num_bundles)
        avg_profits.append(np.mean(type_profits, axis=0)) # avg over types

    return efficient_allocation_from_profits(game, avg_profits)

def efficient_allocation_from_profits(game, profits):
    '''factor_in_opening_prices says to only consider feasible allocations at the opening prices
    
    profits: (num_players, num_bundles)
    '''
    params = game.auction_params
    bundles = params.all_bids
    num_players = game.num_players()
    num_actions = game.num_distinct_actions()
    n_vars = num_players * num_actions
    var_id_to_player_bundle = dict() # VarId -> (player, bundle)
    
    q = 0
    for player in range(num_players):
        for bundle in bundles:
            var_id_to_player_bundle[q] = (player, bundle)
            q += 1

    problem = LpProblem(f"EfficientAllocation", LpMaximize)
    bundle_variables = LpVariable.dicts("X", np.arange(n_vars), cat=LpBinary)

    if np.asarray(profits).sum() == 0: # Nothing is profitable => no objective will get written (dummy) and causes errors
        return 0, [tuple([0] * params.num_products) for _ in range(num_players)]


    # OBJECTIVE
    problem += lpDot(np.reshape(profits, -1).tolist(), bundle_variables.values())
    
    feasible_result = True

    # Constraint: Only 1 bundle per bidder
    for i in range(num_players):
        problem += lpSum(list(bundle_variables.values())[i * num_actions: (i+1) * num_actions]) == 1, f"1-per-bidder-{i}"
        
    # Constraint: Can't overallocate any items
    supply = params.licenses
    for i in range(params.num_products):
        product_amounts = [bundle[i] for (player, bundle) in var_id_to_player_bundle.values()]
        problem += lpDot(bundle_variables.values(), product_amounts) <= supply[i], f"supply-{i}"

    allocation = []
    try: 
        # problem.writeLP(f'efficient_allocation_{random_string(10)}.lp')
        obj = pulp_solve(problem, save_if_failed=True)
        for var_id in range(n_vars):
            # print(  var_id, bundle_variables[var_id], value(bundle_variables[var_id]), var_id_to_player_bundle[var_id])
            if value(bundle_variables[var_id]) > .99: # Rounding stupidness
                allocation.append(var_id_to_player_bundle[var_id][1])
    except ValueError as e:
        # if MIP is infeasible, drop out - TODO: Should this ever happen?
        feasible_result = False
        logging.warning(f'Failed to solve MIP; dropping out')

    return obj, allocation


def compute_per_type_combo(state_fn, policy, game, **kwargs):
    """Compute some function of the initial state for each type combo.

    Args:
    - state_fn: (policy, state, **kwargs) -> value
    - policy: CFR policy
    - game: auction game
    - kwargs: passed to state_fn

    Returns:
    - results: dict of {type combination: value}
    """

    # Count number of types for each player
    # TODO: See ubc_cma type combos, this looks like a reinvention
    state = game.new_initial_state()
    num_types = []
    for player in range(game.num_players()):
        num_types.append(len(state.chance_outcomes()))
        state = state.child(0)

    # Test each type combo independently
    results = {}
    for combo in itertools.product(*[range(n) for n in num_types]):
        state = game.new_initial_state()
        for type in combo:
            state = state.child(type)

        results[combo] = state_fn(policy, state, **kwargs)
    return results

def min_prob(policy, state):
    if state.is_terminal():
        return []
    elif state.is_chance_node():
        chance_outcomes = state.chance_outcomes()
        if len(chance_outcomes) == 1:
            return min_prob(policy, state.child(0))
        else:
            return [
                min_prob(policy, state.child(action))
                for action, _ in state.chance_outcomes()
            ]
    else:
        action_probs = policy.action_probabilities(state)
        modal_action = max(action_probs, key=lambda x: action_probs[x])
        return [action_probs[modal_action]] + min_prob(policy, state.child(modal_action))

def get_demand_history(state, history_type='processed'):
    if not state.is_terminal():
        raise ValueError("State must be terminal")
    demand_fn = lambda bidder: bidder.processed_demand[1:] if history_type == 'processed' else bidder.submitted_demand[1:] # bidder -> (round, product)
    demand_histories = np.array([demand_fn(bidder) for bidder in state.bidders]) # (bidder, round, product)
    demand_histories = demand_histories.transpose(1, 0, 2) # (round, bidder, product)
    demand_histories = tuple(tuple(tuple(d) for d in round_demands) for round_demands in demand_histories) # convert to tuples for dict keys
    return demand_histories

def history_distribution(policy, state, min_prob=0.01, history_type='processed'):
    """Return the distribution over processed/submitted demand histories for the given policy and state.

    TODO: add an option to only include randomness from the policy, not from chance nodes?
    unsure if this would be interpretable (sum will be >1).
    """
    if state.is_terminal():
        history = get_demand_history(state, history_type)
        return {history: 1.0}

    else:
        dist = defaultdict(float)
        action_dist = state.chance_outcomes() if state.is_chance_node() else policy.action_probabilities(state).items()
        for action, action_prob in action_dist:
            if action_prob >= min_prob:
                dist2 = history_distribution(policy, state.child(action), min_prob=min_prob, history_type=history_type)
                for history, history_prob in dist2.items():
                    dist[history] += action_prob * history_prob
        return dist
    
def empirical_history_distribution(e):
    '''e is an Evaluation object'''
    processed_demand_series = pd.Series(zip(e.samples['processed_demands']['0'], e.samples['processed_demands']['1']))#.value_counts(normalize=True)
    type_series = pd.Series(zip(e.samples['types']['0'], e.samples['types']['1']))
    hist_df = pd.DataFrame({
        'processed_demand': processed_demand_series,
        'type': type_series
    })
    hist_df['processed_demand'] = hist_df['processed_demand'].apply(convert_nested_lists_to_tuples)
    return hist_df.groupby('type')['processed_demand'].value_counts(normalize=True)
    
def get_history_entropy(history_distribution):
    """Compute the entropy of a history distribution."""
    return scipy.stats.entropy(list(history_distribution.values()))


def convert_nested_lists_to_tuples(lst):
    if isinstance(lst, list) or isinstance(lst, tuple):
        return tuple(convert_nested_lists_to_tuples(x) for x in lst)
    return lst


def get_results(run, game_cache=None, skip_single_chance_nodes=True, load_policy=True):
    """Load the game, final checkpoint, and policy for a single run.
    """
    if game_cache is None:
        game_cache = dict()
    game = game_cache.get(run.game.name, run.game.load_as_spiel())
    game.auction_params.skip_single_chance_nodes = skip_single_chance_nodes # for backwards compatibility
    game_cache[run.game.name] = game


    for final_checkpoint in run.equilibriumsolverruncheckpoint_set.defer('policy').order_by('-t'):
        try:
            final_checkpoint.get_modal_eval()
            break
        except:
            pass
    if final_checkpoint is None:
        raise ValueError("None final checkpoint?")

    
    if load_policy:
        solver_type = run.config.get('solver_type', 'ppo')
        if solver_type == 'cfr':
            policy = loads(final_checkpoint.policy, compression='gzip')
        else:
            from auctions.webutils import ppo_db_checkpoint_loader # Get around import stuff for sats_game-sampler
            policy = ppo_db_checkpoint_loader(final_checkpoint).make_policy()
    else:
        policy = None

    return game, final_checkpoint, policy

def get_algorithm_from_run(run):
    """Get the algorithm used for a run."""
    alg = run.config.get('solver_type', 'PPO')
    if alg == 'cfr':
        alg += '_' + run.config.get('sampling_method', '')
        if run.config.get('linear_averaging'):
            alg += '_linear'
        if run.config.get('regret_maching_plus'):
            alg += '+'
    return alg

def display_history_distributions(history_dists):
    for type_combo in history_dists:
        print(type_combo)
        for history, probs in history_dists[type_combo].items():
            print(f'{probs:.3f} {history}')
        print()


def rule_set_to_value_structure(s):
    if 'spite' in s:
        return 'spite'
    elif 'risk_averse' in s:
        return 'risk_averse'
    else:
        return 'quasi_linear'

def rule_set_to_rule(s):
    if 'high_speed' in s:
        return 'high_speed'
    elif 'medium_speed' in s:
        return 'medium_speed'
    elif 'grace' in s:
        return 'grace'
    elif 'tie_break' in s:
        return 'tie_break'
    elif 'undersell_allowed' in s:
        return 'undersell_allowed'
    elif 'hide_demand' in s:
        return 'hide_demand'
    elif 'no_activity' in s:
        return 'no_activity'
    else:
        return 'base'
    
def get_game_info(game, game_db):
    game_name = game_db.name
    # Base is like sep19_encumbered_4 4 without all the crap before or afer in game_name. This will fail horribly if you don't have exactly 2 underscores.

    if '/' not in game_name:
        base_game_name = game_name
        rule_set = 'base'
        rule = 'base'
    else:
        base_game_name = '_'.join(game_name.split('/')[1].split('_')[:3]) # Stupid naming convention that will surely bite us later
        rule_set = game_name.split(base_game_name)[-1][1:-5]
        rule = rule_set_to_rule(rule_set)

    try:
        value_structure = game_db.config['players'][0]['type'][0]['utility_function']['name']    
    except:
        value_structure = 'quasi_linear'

    n_types = len(game.auction_params.player_types[0])

    return {
        'base_game_name': base_game_name,
        'information_policy': InformationPolicy(game.auction_params.information_policy).name,
        'activity_policy': ActivityPolicy(game.auction_params.activity_policy).name,
        'undersell_policy': UndersellPolicy(game.auction_params.undersell_policy).name,
        'tiebreaking_policy': TiebreakingPolicy(game.auction_params.tiebreaking_policy).name,
        'grace_rounds': game.auction_params.grace_rounds,
        'clock_speed': game.auction_params.increment,
        'value_structure': value_structure,
        'rule': rule,
        'rho': game.auction_params.sor_bid_bonus_rho,
        'n_types': n_types,
        'deviations': game.auction_params.heuristic_deviations
    }


def get_modal_nash_conv(game, policy, config):
    env_and_policy = make_env_and_policy(game, config)
    for agent in env_and_policy.agents:
        agent.policy = policy
    for player in range(game.num_players()):
        env_and_policy.agents[player] = ModalAgentDecorator(env_and_policy.agents[player])
    modal_policy = env_and_policy.make_policy()
    return get_nash_conv(game, modal_policy)

def get_nash_conv(game, policy):
    worked, time_taken, retval = time_bounded_run(300, nash_conv, game, policy, return_only_nash_conv=True, restrict_to_heuristics=False)
    if worked:
        return retval
    else:
        return None

def get_straightforward_nash_conv(game):
    env_and_policy = make_env_and_policy(game, dict())
    for player in range(game.num_players()):
        env_and_policy.agents[player] = TakeSingleActionDecorator(StraightforwardAgent(player, game), game.num_distinct_actions())
    straightforward_policy = env_and_policy.make_policy()
    return get_nash_conv(game, straightforward_policy)


'''Copy of game b/c we modify params. Use load_as_spiel() e.g.'''
def get_modal_nash_conv_new_rho(game_copy, policy, config, rho=0):
    game_copy.auction_params.sor_bid_bonus_rho = rho
    env_and_policy = make_env_and_policy(game_copy, dict(config))
    for agent in env_and_policy.agents:
        agent.policy = policy
    for player in range(game_copy.num_players()):
        env_and_policy.agents[player] = ModalAgentDecorator(env_and_policy.agents[player])
    modal_policy = env_and_policy.make_policy()
    return get_nash_conv(game_copy, modal_policy)
