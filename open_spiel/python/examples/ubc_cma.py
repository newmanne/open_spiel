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

def efficient_allocation(game):
    # Parse all type combos
    types = [game.auction_params.player_types[player] for player in range(game.num_players())]
    for player_type in types:
        for i, t in enumerate(player_type):
            t['index'] = i

    type_combos = list(itertools.product(*types))

    records = []
    for combo in tqdm(type_combos):
        type_prob = np.product([t['prob'] for t in combo])
        score, allocation = efficient_allocation_from_types(game, combo)
        combo_index = tuple(t['index'] for t in combo)
        record = dict(prob=type_prob, score=score, allocation=tuple(tuple(a) for a in allocation), combo=combo_index)
        records.append(record)

    df = pd.DataFrame.from_records(records)
    combo_to_score = df[['combo', 'score']].set_index('combo')['score'].to_dict()
    scorer = allocation_scorer(game, combo_to_score)
    return df, combo_to_score, scorer

def efficient_allocation_from_types(game,  types):
    params = game.auction_params
    bundles = params.all_bids
    num_players = game.num_players()
    num_actions = game.num_distinct_actions()
    n_vars = num_players * num_actions
    var_id_to_player_bundle = dict() # VarId -> (player, bundle)
    
    values = []
    q = 0
    for player in range(num_players):
        values += types[player]['bidder'].get_values().tolist()
        for bundle in bundles:
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
    supply = params.licenses
    for i in range(params.num_products):
        product_amounts = [bundle[i] for (player, bundle) in var_id_to_player_bundle.values()]
        problem += lpDot(bundle_variables.values(), product_amounts) <= supply[i], f"supply-{i}"

    allocation = []
    try: 
        # problem.writeLP(f'efficient_allocation_{random_string(10)}.lp')
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