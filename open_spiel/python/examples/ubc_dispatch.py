import pandas as pd
import numpy as np
from absl import logging
import glob
import os
from pathlib import Path


def dispatch_experiments(yml_config_dir, base_job_name='auctions', submit=True, mem=16, overrides='', cfr_also=True):
    experiments = glob.glob(f'{yml_config_dir}/*.yml')
    experiments = [ # Reserving the ability for a nicer format later
        {
            'name': Path(experiment).stem,
            'config': Path(experiment).stem
        }
        for experiment in experiments
    ]
    if len(experiments) == 0:
        raise ValueError("No experiments found!")
        
    if cfr_also:
        experiments.append({
            'name': 'cfr',
            'solver': 'cfr'
        })

    spiel_path = os.environ.get('OPENSPIEL_PATH')
    if spiel_path is None:
        raise ValueError("Need to set the OPENSPIEL_PATH env variable")

    config_dir = os.environ.get('CLOCK_AUCTION_CONFIG_DIR')
    if config_dir is None:
        raise ValueError("Need to set CLOCK_AUCTION_CONFIG_DIR env variable")

    pydir = f'{spiel_path}/open_spiel/python/examples'

    experiment_output_dir = f'/shared/outputs/{base_job_name}'

    for experiment in experiments:
        solver = experiment.get('solver', 'nfsp')
        game_name = experiment.get('game_name', 'parking_1')
        experiment_name = experiment['name']
        output_dir = f'{experiment_output_dir}/{experiment_name}'

        if solver == 'cfr':
            # TODO: Better integrate the CFR stuff into the same python script
            command = f'python {pydir}/ubc_mccfr_cpp_example.py --filename parking_1.json --iterations 150000 --report_freq 10000 --output {output_dir}' # TODO: Would be good to flag not to do the post-processing
        else:
            config = experiment['config']
            command = f'python {pydir}/ubc_nfsp_example.py -- --filename {game_name}.json --network_config_file {yml_config_dir}/{config}.yml --output_dir {output_dir} {overrides} --job_name {experiment_name}'

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        slurm_job_name = experiment_name + '_' + base_job_name
        slurm = f"""#!/bin/sh
#SBATCH --cpus-per-task={int(mem/4)}
#SBATCH --job-name={slurm_job_name}

CMD=`{command}`
echo $CMD
eval $CMD
"""
        job_file = f'{experiment_output_dir}/{slurm_job_name}.sh'
        with open(job_file, 'w') as f:
            f.write(slurm)
        os.chmod(job_file, int('777', base=8)) # Octal
        if submit:
            os.system(f'cd {experiment_output_dir} && sbatch {job_file}')

    print(f"Dispatched {len(experiments)} experiments!")