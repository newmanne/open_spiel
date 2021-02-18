import json
import argparse
from sklearn.model_selection import ParameterGrid
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

CMD_FILE_NAME = 'cmds.txt'

def main(root, spiel_path, job_name):
    logging.basicConfig()

    Path(f'{root}').mkdir(parents=True, exist_ok=True)
    Path(f'{root}/logs').mkdir(parents=True, exist_ok=True)

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
    very_low = {
        "value": 121,
        "budget": 900
    }
    very_high = {
        "value": 300,
        "budget": 900
    }

    p0 = {
        'value': 225,
        'budget': 525
    }
    p1_l = {
        'value': 150,
        'budget': 400
    }
    p1_h = {
        'value': 300,
        'budget': 650
    }

    def make_player(player_types): 
        player = dict()
        player['type'] = []
        for t, p in player_types:
            logger.info(t)
            d = dict(t)
            d['prob'] = p
            player['type'].append(d)
        return player

    param_grid = [
        {'opening_price': [100], 'increment': [0.1], 'licenses': [3], 'undersell_rule': ["undersell_standard", "undersell_allowed"]},
        # {'opening_price': [100], 'increment': [0.1, 0.15, 0.2], 'licenses': [3, 4, 5], 'undersell_rule': [True, False]},
    ]

    player_grid = [
        # [make_player([(low, 0.9), (high, 0.1)]), make_player([(medium, 1.0)])],
        # [make_player([(low, 0.5), (high, 0.5)]), make_player([(medium, 1.0)])],
        # [make_player(((low, 0.1), (high, 0.9))), make_player(((medium, 1.0),))],
        [make_player([(very_low, 1.0)]), make_player([(very_high, 1.0)])], # Trying to recreate 2b
        [make_player([(p0, 1.0)]), make_player([(p1_l, 0.5), (p1_h, 0.5)])],
    ]

    i = 1
    cmds = []


    solver_grid = [
        # {'solver': ['cfr', 'cfrplus', 'cfrbr']},
        {'solver': ['cfr']},
        {'solver': ['mccfr --sampling external'], 'name': ['mccfr_ext'], 'seed': [i for i in range(2,4)]}
    ]

    for parameterization in ParameterGrid(param_grid):
        for players in player_grid:
            parameterization['players'] = players
            if len(players) <= 1:
                raise ValueError("Fewer than one player?")
            Path(f'{root}/{i}').mkdir(parents=True, exist_ok=True)
            with open(f'{root}/{i}/{i}.json', 'w') as f:
                json.dump(parameterization, f)
                for solver_config in ParameterGrid(solver_grid):
                    solver = solver_config['solver']
                    seed = solver_config.get('seed', 123)
                    name = solver_config.get('name', solver)
                    cmd = f'cd {root}/{i} && python {spiel_path}/open_spiel/python/examples/ubc_mccfr_cpp_example.py --filename={root}/{i}/{i}.json --iterations 10000 --solver={solver} --output {root}/{i}/{name}_{seed} --seed {seed}'
                    cmds.append(cmd)
                i += 1

    print(f"Dumped {i-1} configs to {root}")
    with open(f'{root}/{CMD_FILE_NAME}', 'w') as f:
        for cmd in cmds:
            f.write(cmd + '\n')
    print (f"{len(cmds)} commands written to {root}/{CMD_FILE_NAME}")

    JOB_NAME = job_name
    slurm = f"""#!/bin/sh
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name={JOB_NAME}
#SBATCH --output=logs/{JOB_NAME}-%A_%a.out-o.txt
#SBATCH --error=logs/{JOB_NAME}-%A_%a.out-e.txt
#SBATCH --account=rrg-kevinlb
#SBATCH --time=1-0
#SBATCH --array=1-{len(cmds)}

source {spiel_path}/venv/bin/activate

CMD=`head -n $SLURM_ARRAY_TASK_ID {root}/{CMD_FILE_NAME} | tail -n 1`
echo $CMD
eval $CMD
"""
    JOB_FILE = 'job_runner.sh'
    with open(f'{root}/{JOB_FILE}', 'w') as f:
        f.write(slurm)
    os.chmod(f"{root}/{JOB_FILE}", int('777', base=8)) # Octal


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Queue up a bunch of CFR jobs')
    parser.add_argument('--root', default='/home/newmanne/scratch/cfr', type=str)
    parser.add_argument('--spiel_path', default='/project/def-kevinlb/newmanne/cfr/open_spiel/', type=str)
    parser.add_argument('--job-name', default='CFR', type=str)
    args = parser.parse_args()
    main(args.root, args.spiel_path, args.job_name)

    # SINGULARITY="singularity exec -B /home -B /project -B /scratch -B /localscratch /project/def-kevinlb/newmanne/openspiel.simg"
    # srun $SINGULARITY $CMD
