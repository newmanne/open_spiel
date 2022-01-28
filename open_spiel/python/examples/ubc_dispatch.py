import pandas as pd
import numpy as np
from absl import logging
import glob
import os
from pathlib import Path

def verify_config():
    spiel_path = os.environ.get('OPENSPIEL_PATH')
    if spiel_path is None:
        raise ValueError("Need to set the OPENSPIEL_PATH env variable")

    config_dir = os.environ.get('CLOCK_AUCTION_CONFIG_DIR')
    if config_dir is None:
        raise ValueError("Need to set CLOCK_AUCTION_CONFIG_DIR env variable")
    
    pydir = f'{spiel_path}/open_spiel/python/examples'
    return spiel_path, config_dir, pydir


def dispatch_experiments(yml_config_dir, single_config=None, base_job_name=None, game_name='parking_1', submit=True, mem=16, overrides='', cfr_also=False):
    if base_job_name is None:
        base_job_name = game_name
    
    key = '*' if single_config is None else single_config
    experiments = glob.glob(f'{yml_config_dir}/{key}.yml')
    experiments = [ # Reserving the ability for a nicer format later
        {
            'name': Path(experiment).stem,
            'config': Path(experiment).stem,
            'game_name': game_name,
        }
        for experiment in experiments
    ]
    if len(experiments) == 0:
        raise ValueError("No experiments found!")
        
    if cfr_also:
        experiments.append({
            'name': 'cfr',
            'solver': 'cfr',
            'game_name': game_name,
        })

    spiel_path, config_dir, pydir = verify_config()

    experiment_output_dir = f'/shared/outputs/{base_job_name}'

    for experiment in experiments:
        solver = experiment.get('solver', 'nfsp')
        game_name = experiment.get('game_name', 'parking_1')
        experiment_name = experiment['name']
        output_dir = f'{experiment_output_dir}/{experiment_name}'

        if solver == 'cfr':
            # TODO: Better integrate the CFR stuff into the same python script
            command = f'python {pydir}/ubc_mccfr_cpp_example.py --filename {game_name}.json --iterations 150000 --report_freq 10000 --output {output_dir}' # TODO: Would be good to flag not to do the post-processing
        else:
            config = experiment['config']
            command = f'python {pydir}/ubc_nfsp_example.py --alsologtostderr -- --filename {game_name}.json --network_config_file {yml_config_dir}/{config}.yml --output_dir {output_dir} --dispatch_br true {overrides} --job_name {experiment_name}'

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        slurm_job_name = experiment_name + '_' + base_job_name
        job_file_text = f"""#!/bin/sh
#SBATCH --cpus-per-task={int(mem/4)}
#SBATCH --job-name={slurm_job_name}
#SBATCH --time=5-0:00:00 # days-hh:mm:ss
CMD=`{command}`
echo $CMD
eval $CMD
"""
        job_file_path = f'{experiment_output_dir}/{experiment_name}.sh'
        write_job_file(job_file_path, job_file_text)
        if submit:
            os.system(f'cd {experiment_output_dir} && sbatch {job_file_path}')

    print(f"Dispatched {len(experiments)} experiments!")


def dispatch_br(experiment_dir, br_player=0, checkpoint='checkpoint_latest', submit=True, mem=16, overrides=''):
    spiel_path, config_dir, pydir = verify_config()
    command = f'python {pydir}/ubc_br.py --alsologtostderr -- --experiment_dir {experiment_dir} --br_player {br_player} --checkpoint {checkpoint} --dispatch_rewards True {overrides}'
    
    basename = os.path.basename(experiment_dir)
    slurm_job_name = f'br_{br_player}_{checkpoint}_{basename}'
    job_file_text = f"""#!/bin/sh
#SBATCH --cpus-per-task={int(mem/4)}
#SBATCH --job-name={slurm_job_name}
#SBATCH --time=5-0:00:00 # days-hh:mm:ss
CMD=`{command}`
echo $CMD
eval $CMD
"""
    job_file_path = f'{experiment_dir}/br_{br_player}_{checkpoint}.sh'
    write_job_file(job_file_path, job_file_text)
    if submit:
        os.system(f'cd {experiment_dir} && sbatch {job_file_path}')

    print(f"Dispatched experiment!")

def dispatch_eval(experiment_dir, br_name=None, straightforward_player=None, checkpoint='checkpoint_latest', submit=True, mem=16, overrides=''):
    spiel_path, config_dir, pydir = verify_config()
    command = f'python {pydir}/ubc_evaluate_policy.py --alsologtostderr -- --experiment_dir {experiment_dir} --checkpoint {checkpoint} {overrides}'
    if br_name is not None:
        command += f' --br_name {br_name}'
    elif straightforward_player is not None:
        command += f' --straightforward_player {straightforward_player}'
    
    basename = os.path.basename(experiment_dir)
    if br_name:
        slurm_job_name = f'eval_{br_name}_{checkpoint}_{basename}'
    elif straightforward_player is not None:
        slurm_job_name = f'eval_straightforward_{straightforward_player}_{checkpoint}_{basename}'
    else:
        slurm_job_name = f'eval_{checkpoint}_{basename}'
    job_file_text = f"""#!/bin/sh
#SBATCH --cpus-per-task={int(mem/4)}
#SBATCH --job-name={slurm_job_name}
#SBATCH --time=5-0:00:00 # days-hh:mm:ss
CMD=`{command}`
echo $CMD
eval $CMD
"""
    job_file_path = f'{experiment_dir}/{slurm_job_name}.sh'
    write_job_file(job_file_path, job_file_text)
    if submit:
        os.system(f'cd {experiment_dir} && sbatch {job_file_path}')

    print(f"Dispatched experiment!")

def write_job_file(job_file_path, job_file_text):
    with open(job_file_path, 'w') as f:
        f.write(job_file_text)
    os.chmod(job_file_path, int('777', base=8)) # Octal
