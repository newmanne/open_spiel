import numpy as np
import pandas as pd
from collections import Counter
from open_spiel.python.examples.ubc_utils import *
from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, LpBinary, lpSum, lpDot, LpMaximize, LpInteger, value

def analyze_checkpoint(checkpoint):
    record = dict()
    samples = checkpoint.evaluation.samples
    welfares = np.zeros(len(samples['rewards']['0']))
    revenues = np.zeros(len(samples['rewards']['0']))
    for player in range(checkpoint.equilibrium_solver_run.game.num_players):
        player = str(player)
        rewards = pd.Series(samples['rewards'][player])
        payments = pd.Series(samples['payments'][player])
        record[f'p{player}_utility'] = rewards.mean()
        record[f'p{player}_payment'] = payments.mean()
        # TODO: Talk about efficiency of the allocation, NOT welfare. Also, does welfare still make sense with pricing bonuses?
        welfares += rewards + payments
        revenues += payments
    record['total_welfare'] = welfares.mean()
    record['total_revenue'] = revenues.mean()
    record['auction_lengths'] = np.array(samples['auction_lengths']).mean()

    arr = np.array(samples['allocations']['0']).astype(int) # TODO: only player 0's allocation.
    c = Counter(tuple(map(tuple, arr)))
    record['common_allocations'] = c.most_common(5)
    return record


def allocation_scorer(game_config, normalizer_dict):
    value_functions = []
    for player in range(len(game_config['players'])):
        player_value_functions = []
        for player_type in game_config['players'][player]['type']:
            player_value_functions.append(value_for_bundle_function(player_type, game_config))
        value_functions.append(player_value_functions)          
        
    def scorer(alloc, types):
        score = sum((value_functions[i][types[i]](alloc[i]) for i in range(len(alloc))))
        return score, score / normalizer_dict[types]
        
    return scorer


def value_for_bundle_function(t, game_config):
    # Reinvent the C value parser
    if t.get('value_format') == 'full':
        # Enum
        all_bundles = list(map(tuple,action_to_bundles(game_config['licenses']).values()))
        def val(bundle):
            bundle_index = all_bundles.index(bundle)
            return t['value'][bundle_index]
        return val
    elif isinstance(t['value'][0], list):
        # Marginal
        def val(bundle):
            v = 0
            for j, value_for_product in enumerate(t['value']):
                v += sum(value_for_product[:bundle[j]])
            return v
        
        return val
        
    else:
        np_value = np.array(t['value'])
        return lambda b: np.array(b) @ np_value

def efficient_allocation(game, game_config):
    # Parse all type combos
    num_players, num_actions, num_products = game_spec(game, game_config)
    
    combos = []
    for player in range(num_players):
        player_types = list(range(len(game_config['players'][player]['type'])))
        combos.append(player_types)
        
    type_combos = itertools.product(*combos)
    records = []
    for combo in type_combos:
        type_prob = 1.
        for player in range(num_players):
            type_prob *= game_config['players'][player]['type'][combo[player]]['prob']
        score, allocation = efficient_allocation_from_types(game, game_config, combo)
        record = dict(prob=type_prob, score=score, allocation=allocation, combo=combo)
        records.append(record)

    df = pd.DataFrame.from_records(records)
    combo_to_score = df[['combo', 'score']].set_index('combo')['score'].to_dict()
    scorer = allocation_scorer(game_config, combo_to_score)
    return df, combo_to_score, scorer

def efficient_allocation_from_types(game, game_config, types):
    num_players, num_actions, num_products = game_spec(game, game_config)
    bundles = action_to_bundles(game_config['licenses']).values()
    
    n_vars = num_players * num_actions
    var_id_to_player_bundle = dict() # VarId -> (player, bundle)
    
    values = []
    q = 0
    for player in range(num_players):
        v = value_for_bundle_function(game_config['players'][player]['type'][types[player]], game_config)
        for bundle in bundles:
            values.append(v(bundle))
            var_id_to_player_bundle[q] = (player, bundle)
            q += 1

    problem = LpProblem(f"EfficientAllocation", LpMaximize)
    bundle_variables = LpVariable.dicts("X", np.arange(n_vars), cat=LpBinary)

    # OBJECTIVE
    problem += lpDot(values, bundle_variables.values())
    
    feasible_result = True

    # Constraint: Only 1 bundle per bidder
    for i in range(num_players):
        problem += lpSum(list(bundle_variables.values())[i * num_actions: (i+1) * num_actions]) == 1, f"1-per-bidder-{i}"
        
    # Constraint: Can't overallocate any items
    supply = game_config['licenses']
    for i in range(num_products):
        product_amounts = [bundle[i] for (player, bundle) in var_id_to_player_bundle.values()]
        problem += lpDot(bundle_variables.values(), product_amounts) <= supply[i], f"supply-{i}"

    allocation = []
    try: 
        problem.writeLP(f'efficient_allocation.lp')
        obj = pulp_solve(problem, save_if_failed=True)
        for var_id in range(n_vars):
            # print(var_id, bundle_variables[var_id], value(bundle_variables[var_id]), var_id_to_player_bundle[var_id])
            if value(bundle_variables[var_id]) > .99: # Rounding stupidness
                allocation.append(var_id_to_player_bundle[var_id][1])
    except ValueError as e:
        # if MIP is infeasible, drop out - TODO: Should this ever happen?
        feasible_result = False
        logging.warning(f'Failed to solve MIP; dropping out')

    return obj, allocation