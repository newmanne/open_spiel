import json
import argparse
from sklearn.model_selection import ParameterGrid
import os
from pathlib import Path
import logging
import itertools

logger = logging.getLogger(__name__)

CMD_FILE_NAME = 'cmds.txt'

def make_player(player_types): 
    player = dict()
    player['type'] = []
    for t, p in player_types:
        logger.info(t)
        d = dict(t)
        d['prob'] = p
        player['type'].append(d)
    return player


def grids_to_commands(param_grid, player_grid, solver_grid, root, grid_name, spiel_path, job_name='CFR', submit=True):
    grid_path = f'{root}/{grid_name}'
    i = 1
    cmds = []

    for parameterization in ParameterGrid(param_grid):
        for players in player_grid:
            parameterization['players'] = players
            if len(players) <= 1:
                raise ValueError("Fewer than one player?")
            Path(f'{grid_path}/{i}').mkdir(parents=True, exist_ok=True)
            with open(f'{grid_path}/{i}/{i}.json', 'w') as f:
                json.dump(parameterization, f)
                for j, solver_config in enumerate(ParameterGrid(solver_grid)):
                    solver = solver_config['solver']
                    seed = solver_config.get('seed', 123)
                    name = solver_config.get('name', f'{solver}_{j}')
                    solver_args = solver_config.get('solver_args', '')
                    cmd = f'cd {grid_path}/{i} && python {spiel_path}/open_spiel/python/examples/ubc_mccfr_cpp_example.py --filename={grid_path}/{i}/{i}.json --iterations 10000 --solver={solver} {solver_args} --output {grid_path}/{i}/{name}_{seed} --seed {seed}'
                    cmds.append(cmd)
                i += 1

    print(f"Dumping {i-1} configs to {grid_path}")
    with open(f'{grid_path}/{CMD_FILE_NAME}', 'w') as f:
        for cmd in cmds:
            f.write(cmd + '\n')
    print (f"{len(cmds)} commands written to {grid_path}/{CMD_FILE_NAME}")

    Path(f'{grid_path}/logs').mkdir(parents=True, exist_ok=True)

    JOB_NAME = job_name
    slurm = f"""#!/bin/sh
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name={JOB_NAME}-{grid_name}
#SBATCH --output=logs/{JOB_NAME}-%A_%a.out-o.txt
#SBATCH --error=logs/{JOB_NAME}-%A_%a.out-e.txt
#SBATCH --account=rrg-kevinlb
#SBATCH --time=3-0
#SBATCH --array=1-{len(cmds)}

source {spiel_path}/venv/bin/activate

CMD=`head -n $SLURM_ARRAY_TASK_ID {CMD_FILE_NAME} | tail -n 1`
echo $CMD
eval $CMD
"""
    JOB_FILE = 'job_runner.sh'
    with open(f'{grid_path}/{JOB_FILE}', 'w') as f:
        f.write(slurm)
    os.chmod(f"{grid_path}/{JOB_FILE}", int('777', base=8)) # Octal
    if submit:
        os.system(f'cd {grid_path} && sbatch {grid_path}/{JOB_FILE}')

def main(root, spiel_path, job_name, submit):
    logging.basicConfig()

    Path(f'{root}').mkdir(parents=True, exist_ok=True)

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
        {'opening_price': [100], 'increment': [0.1], 'licenses': [3], 'undersell_rule': ["undersell_standard"], "information_policy": ["hide_demand"]},
    ]

    player_grid = [
        [make_player([(low, 1.0)]), (low, 1.0)]), make_player([(low, 0.5), (high, 0.5)])]
    ]

    solver_grid = [
        {'solver': ['cfr']},
        {'solver': ['cfrplus']},
        {'solver': ['ecfr'],'solver_args': [f'--initial_eps {initial_eps} --decay_freq {freq} --decay_factor {decay_factor}' for (initial_eps, freq, decay_factor) in itertools.product([0.01, 0.001], [1000, 2500], [0.9, 0.99])]},
    ]

#    grids_to_commands(param_grid, player_grid, solver_grid, root, 'small', spiel_path, job_name=job_name, submit=submit)
    grids_to_commands(param_grid, player_grid, solver_grid, root, '3playershidden', spiel_path, job_name=job_name, submit=submit)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Queue up a bunch of CFR jobs')
    parser.add_argument('--root', default='/home/newmanne/scratch/cfr', type=str)
    parser.add_argument('--spiel_path', default='/project/def-kevinlb/newmanne/cfr/open_spiel', type=str)
    parser.add_argument('--job-name', default='CFR', type=str)
    parser.add_argument('--submit', default=True)
    args = parser.parse_args()
    main(args.root, args.spiel_path, args.job_name, args.submit)

    # SINGULARITY="singularity exec -B /home -B /project -B /scratch -B /localscratch /project/def-kevinlb/newmanne/openspiel.simg"
    # srun $SINGULARITY $CMD
