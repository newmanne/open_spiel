set -e 

#python /apps/open_spiel/notebooks/greg/sample_games.py --n_configs 5 --licenses 1 4 --action_prefix 3 3 --min_mccfr_iters 100 --seed 1234 --output configs/jan12_repro_3p.pkl --min_bidders 3 --max_bidders 3

#python /apps/open_spiel/notebooks/greg/sample_games.py --n_configs 5 --licenses 1 4 --action_prefix 3 3 --min_mccfr_iters 100 --seed 1234 --output configs/jan12_repro_2t.pkl --min_types 2 --max_types 2

#python /apps/open_spiel/notebooks/greg/sample_games.py --n_configs 5 --licenses 1 4 --action_prefix 3 3 --min_mccfr_iters 100 --seed 1234 --output configs/jan12_repro_4t.pkl --min_types 4 --max_types 4

python /apps/open_spiel/notebooks/greg/sample_games.py --n_configs 5 --licenses 1 4 --action_prefix 3 3 --min_mccfr_iters 100 --seed 1234 --output configs/jan12_repro_1t.pkl --min_types 1 --max_types 1
