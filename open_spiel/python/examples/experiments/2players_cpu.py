from open_spiel.python.examples.gen_params import *

V_L = 121
B_L = 600

V_M = 150
B_M = 600

V_H = 180
B_H = 600

low = {
    "value": V_L,
    "budget": B_L,
}
medium = {
    'value': V_M,
    'budget': B_M,
}
high = {
    "value": V_H,
    "budget": B_H,
}

param_grid = [
    {'opening_price': [100], 'increment': [0.1], 'licenses': [3], 'undersell_rule': ["undersell_standard"], "information_policy": ["hide_demand", "show_demand"], "bidding": ["weakly_positive_profit"]},
]

player_grid = [
    [make_player([(low, 1.0)]), make_player([(low, 1.0)]), make_player([(low, 0.5), (high, 0.5)])]
]

solver_grid = [
    {'solver': ['cfr']},
    {'solver': ['cfrplus']},
    {'solver': ['ecfr'],'solver_args': [f'--initial_eps {initial_eps} --decay_freq {freq} --decay_factor {decay_factor}' for (initial_eps, freq, decay_factor) in itertools.product([0.01, 0.001], [1000, 2500], [0.9, 0.99])]},
]

grids_to_commands(param_grid, player_grid, solver_grid, '2players_CPU')