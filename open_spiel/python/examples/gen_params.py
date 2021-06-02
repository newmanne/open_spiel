import json
import argparse
from sklearn.model_selection import ParameterGrid
import os
from pathlib import Path
import logging
import itertools

logger = logging.getLogger(__name__)

CMD_FILE_NAME = 'cmds.txt'
OUTPUT_FILE_NAME = 'outputs.txt'

def make_player(player_types): 
    player = dict()
    player['type'] = []
    for t, p in player_types:
        logger.info(t)
        d = dict(t)
        d['prob'] = p
        player['type'].append(d)
    return player


def grids_to_commands(param_grid, player_grid, solver_grid, grid_name, job_name='CFR', submit=True, mem=16, time_limit='3-0'):
    SPIEL_PATH = os.environ.get('OPENSPIEL_PATH', '/project/def-kevinlb/newmanne/cfr/open_spiel')
    ROOT = os.environ.get('SPIEL_ROOT', '/home/newmanne/scratch/cfr')
    CONFIG_DIR = os.environ.get('CLOCK_AUCTION_CONFIG_DIR', '/home/newmanne/scratch/cfr/configs')

    grid_path = f'{ROOT}/{grid_name}'
    i = 1
    cmds = []
    outputs = []
    for parameterization in ParameterGrid(param_grid):
        for players in player_grid:
            parameterization['players'] = players
            if len(players) <= 1:
                raise ValueError("Less than one player?")
            Path(f'{grid_path}/{i}').mkdir(parents=True, exist_ok=True)
            with open(f'{CONFIG_DIR}/{job_name}_{i}.json', 'w') as f:
                json.dump(parameterization, f)
                for j, solver_config in enumerate(ParameterGrid(solver_grid)):
                    solver = solver_config['solver']
                    seed = solver_config.get('seed', 123)
                    name = solver_config.get('name', f'{solver}_{j}')
                    iterations = solver_config.get('iterations', 100_000)
                    solver_args = solver_config.get('solver_args', '')
                    output = f'{grid_path}/{i}/{name}_{seed}'
                    outputs.append(output)
                    cmd = f'cd {grid_path}/{i} && python {SPIEL_PATH}/open_spiel/python/examples/ubc_mccfr_cpp_example.py --filename={job_name}_{i}.json --iterations {iterations} --solver={solver} {solver_args} --output {output} --seed {seed}'
                    cmds.append(cmd)
                i += 1

    print(f"Dumping {i-1} configs to {grid_path}")
    with open(f'{grid_path}/{CMD_FILE_NAME}', 'w') as f:
        for cmd in cmds:
            f.write(cmd + '\n')
    print (f"{len(cmds)} commands written to {grid_path}/{CMD_FILE_NAME}")
    with open(f'{grid_path}/{OUTPUT_FILE_NAME}', 'w') as f:
        for output in outputs:
            f.write(output + '\n')


    Path(f'{grid_path}/logs').mkdir(parents=True, exist_ok=True)

    JOB_NAME = job_name
    slurm = f"""#!/bin/sh
#SBATCH --cpus-per-task={int(mem/4)}
#SBATCH --mem-per-cpu={mem}G
#SBATCH --job-name={JOB_NAME}-{grid_name}
#SBATCH --output=logs/{JOB_NAME}-%A_%a.out-o.txt
#SBATCH --error=logs/{JOB_NAME}-%A_%a.out-e.txt
#SBATCH --account=rrg-kevinlb
#SBATCH --time={time_limit}
#SBATCH --array=1-{len(cmds)}

source {SPIEL_PATH}/venv/bin/activate

CMD=`head -n $SLURM_ARRAY_TASK_ID {CMD_FILE_NAME} | tail -n 1`
echo $CMD
eval $CMD
OUTPUT=`head -n $SLURM_ARRAY_TASK_ID {OUTPUT_FILE_NAME} | tail -n 1`
ln -s logs/{JOB_NAME}-$SLURM_ARRAY_JOB_ID_$SLURM_ARRAY_TASK_ID.out-o.txt $OUTPUT/cfr-o.log
ln -s logs/{JOB_NAME}-$SLURM_ARRAY_JOB_ID_$SLURM_ARRAY_TASK_ID.out-e.txt $OUTPUT/cfr-e.log
"""
    JOB_FILE = 'job_runner.sh'
    with open(f'{grid_path}/{JOB_FILE}', 'w') as f:
        f.write(slurm)
    os.chmod(f"{grid_path}/{JOB_FILE}", int('777', base=8)) # Octal
    if submit:
        os.system(f'cd {grid_path} && sbatch {grid_path}/{JOB_FILE}')