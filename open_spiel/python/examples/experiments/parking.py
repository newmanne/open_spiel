from open_spiel.python.examples.gen_params import *


p1 = {
        'value': [150, 0],
        'budget': 121,
     }

p2 = {
        'value': [0, 150],
        'budget': 130,
    }

p3_a = {
        'value': [150, 0],
        'budget': 110,
        }

p3_b = {
        'value': [150, 0],
        'budget': 130,
}


param_grid = [
        {'opening_price': [(100, 100)], 'increment': [0.1], 'licenses': [(1,1)],'activity': [(100,100)], 'undersell_rule': ["undersell"], "information_policy": ["hide_demand", "show_demand"]},
]

player_grid = [
    [make_player([(p1, 1.0)]), make_player([(p2, 1.0)]), make_player([(p3_a, 0.5), (p3_b, 0.5)])],
]

solver_grid = [
        {'solver': ['mccfr'], 'seed': list(range(10))}
    #{'solver': ['cfr']},
    #{'solver': ['cfrplus']},
    #{'solver': ['ecfr'],'solver_args': [f'--initial_eps {initial_eps} --decay_freq {freq} --decay_factor {decay_factor}' for (initial_eps, freq, decay_factor) in itertools.product([0.01], [2500], [0.99])]},
]

grids_to_commands(param_grid, player_grid, solver_grid, job_name='parking', mem=32)
