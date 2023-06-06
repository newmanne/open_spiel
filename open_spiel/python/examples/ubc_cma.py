import numpy as np
import pandas as pd
from collections import Counter
from open_spiel.python.examples.ubc_utils import *
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, LpBinary, lpSum, lpDot, LpMaximize, LpInteger, value
from tqdm import tqdm

def type_combos(game):
    types = [game.auction_params.player_types[player] for player in range(game.num_players())]
    for player_type in types:
        for i, t in enumerate(player_type):
            t['index'] = i
    type_combos = list(itertools.product(*types))
    # type_combos = list(map(lambda x: (x[0]['index'], x[1]['index']), type_combos))
    return type_combos

def analyze_checkpoint(checkpoint):
    record = dict()
    samples = checkpoint.evaluation.samples
    welfares = np.zeros(len(samples['raw_rewards']['0']))
    revenues = np.zeros(len(samples['raw_rewards']['0']))
    for player in range(checkpoint.equilibrium_solver_run.game.num_players):
        player = str(player)
        rewards = pd.Series(samples['raw_rewards'][player])
        payments = pd.Series(samples['payments'][player])
        record[f'p{player}_utility'] = rewards.mean()
        record[f'p{player}_payment'] = payments.mean()
        # TODO: Talk about efficiency of the allocation, NOT welfare. Also, does welfare still make sense with pricing bonuses?
        welfares += rewards + payments # Welfare: what you would get if you got it for free
        revenues += payments
    record['total_welfare'] = welfares.mean()
    record['total_revenue'] = revenues.mean()
    record['auction_lengths'] = np.array(samples['auction_lengths']).mean()

    arr = np.array(samples['allocations']['0']).astype(int) # TODO: only player 0's allocation.
    c = Counter(tuple(map(tuple, arr)))
    record['common_allocations'] = c.most_common(5)
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