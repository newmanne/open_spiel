import numpy as np
from collections import defaultdict

VALUES = {
    'P1':  [50, 24],
    'P2S': [50, 30],
    'P2W': [50, 14],
}

PRICES = 13 * 1.4**np.arange(4)
print(PRICES)

def profit(player, outcome, price):
    if isinstance(outcome, list): # list of (prob, qty)
        return sum([prob * profit(player, qty, price) for (prob, qty) in outcome])
    else: # prob 1 that qty = outcome
        return sum(VALUES[player][:outcome]) - price * outcome

# check everything about the setting
assert np.all(np.diff(PRICES)) > 0, "Prices must be increasing"
for (player, last_round_demand_2) in [
    ('P1', 1),
    ('P2S', 2),
    ('P2W', 0)
]:
    # print([(profit(player, 1, price), profit(player, 2, price)) for price in PRICES])
    assert profit(player, 2, PRICES[last_round_demand_2]) > profit(player, 1, PRICES[last_round_demand_2]), \
        f'Player {player} must have a truthful demand of 2 in round {last_round_demand_2}'
    assert profit(player, 2, PRICES[last_round_demand_2+1]) < profit(player, 1, PRICES[last_round_demand_2+1]), \
        f'Player {player} must have a truthful demand of 1 in round {last_round_demand_2+1}'

# notation: (X, YZ) = "player 1 drop in round X; player 2 drop in round Y when strong and Z when weak"
payoffs_p1 = defaultdict(float)
payoffs_p2 = defaultdict(float)
tiebreak_outcome = [(0.5, 1), (0.5, 2)]
for p1_strategy in [0, 1, 2]:
    for p2s_strategy in [0, 1, 2]:
        for p2w_strategy in [0, 1]:
            s = (p1_strategy, p2s_strategy, p2w_strategy)

            # when P2 is strong (50% chance)
            final_round = min(p1_strategy, p2s_strategy)
            if p1_strategy == p2s_strategy:
                payoffs_p1[s] += profit('P1',  tiebreak_outcome, PRICES[final_round]) / 2
                payoffs_p2[s] += profit('P2S', tiebreak_outcome, PRICES[final_round]) / 2
            elif p1_strategy < p2s_strategy:
                payoffs_p1[s] += profit('P1',  1, PRICES[final_round]) / 2
                payoffs_p2[s] += profit('P2S', 2, PRICES[final_round]) / 2
            else: # p1_strategy > p2s_strategy
                payoffs_p1[s] += profit('P1',  2, PRICES[final_round]) / 2
                payoffs_p2[s] += profit('P2S', 1, PRICES[final_round]) / 2

            # when P2 is weak (50% chance)
            final_round = min(p1_strategy, p2w_strategy)
            if p1_strategy == p2w_strategy:
                payoffs_p1[s] += profit('P1',  tiebreak_outcome, PRICES[final_round]) / 2
                payoffs_p2[s] += profit('P2W', tiebreak_outcome, PRICES[final_round]) / 2
            elif p1_strategy < p2w_strategy:
                payoffs_p1[s] += profit('P1',  1, PRICES[final_round]) / 2
                payoffs_p2[s] += profit('P2W', 2, PRICES[final_round]) / 2
            else: # p1_strategy > p2w_strategy
                payoffs_p1[s] += profit('P1',  2, PRICES[final_round]) / 2
                payoffs_p2[s] += profit('P2W', 1, PRICES[final_round]) / 2

# print
print('3 6')
print()
for payoff_matrix in [payoffs_p1, payoffs_p2]:
    for p1_strategy in [0, 1, 2]:
        payoffs_line = []
        for p2w_strategy in [0, 1]:
            for p2s_strategy in [0, 1, 2]:
                payoffs_line.append(payoff_matrix[(p1_strategy, p2s_strategy, p2w_strategy)])
        print(' '.join(map(str, payoffs_line)))
    print()