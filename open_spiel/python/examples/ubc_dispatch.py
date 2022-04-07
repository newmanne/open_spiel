import pandas as pd
import numpy as np
from absl import logging
import glob
import os
from pathlib import Path
from open_spiel.python.examples.ubc_utils import EVAL_DIR, BR_DIR, CHECKPOINT_FOLDER, CONFIG_ROOT, config_path_from_config_name, safe_config_name

def verify_config():
    spiel_path = os.environ.get('OPENSPIEL_CLUSTER_PATH')
    if spiel_path is None:
        raise ValueError("Need to set the OPENSPIEL_CLUSTER_PATH env variable")

    config_dir = os.environ.get('CLOCK_AUCTION_CONFIG_DIR')
    if config_dir is None:
        raise ValueError("Need to set CLOCK_AUCTION_CONFIG_DIR env variable")
    
    pydir = f'{spiel_path}/open_spiel/python/examples'
    manage_path = f'{spiel_path}/web/auctions/manage.py'
    return spiel_path, config_dir, pydir, manage_path

def write_job_file(job_file_path, job_file_text):
    with open(job_file_path, 'w') as f:
        f.write(job_file_text)
    os.chmod(job_file_path, int('777', base=8)) # Octal

def write_and_submit(experiment_output_dir, experiment_name, job_file_text, submit):
    job_file_path = f'{experiment_output_dir}/{experiment_name}.sh'
    write_job_file(job_file_path, job_file_text)
    if submit:
        os.system(f'cd {experiment_output_dir} && sbatch {job_file_path}')

def dispatch_experiments(yml_config, base_job_name=None, game_name='parking_1', submit=True, mem=16, overrides='', cfr_also=False, database=True, n_seeds=1, start_seed=100):
    '''yml_config is either a folder or a single config'''

    if base_job_name is None:
        base_job_name = game_name
    
    single_config = '/' in yml_config
    if single_config:
        key = ''
    else:
        yml_config = yml_config + '/'
        key = '*'
    experiment_configs = glob.glob(f'{CONFIG_ROOT}/{yml_config}{key}.yml')
    experiments = []
    for experiment_config in experiment_configs:
        for seed in range(start_seed, start_seed + n_seeds):
            config = str(Path(experiment_config).relative_to(CONFIG_ROOT))
            experiments.append(
                {
                    'name': game_name + '-' + safe_config_name(config) + '-' + str(seed),
                    'config': config.replace('.yml', ''),
                    'game_name': game_name,
                    'seed': seed
                }
            )

    if len(experiments) == 0:
        raise ValueError("No experiments found!")
        
    if cfr_also:
        experiments.append({
            'name': 'cfr',
            'solver': 'cfr',
            'game_name': game_name,
        })

    spiel_path, config_dir, pydir, manage_path = verify_config()

    experiment_output_dir = f'/shared/outputs/{base_job_name}'

    for experiment in experiments:
        solver = experiment.get('solver', 'nfsp')
        game_name = experiment.get('game_name', 'parking_1')
        experiment_name = experiment['name']
        seed = experiment['seed']
        output_dir = f'{experiment_output_dir}/{experiment_name}'

        if solver == 'cfr':
            command = f'python {pydir}/ubc_mccfr_cpp_example.py --filename {game_name}.json --iterations 150000 --report_freq 10000 --output {output_dir}' # TODO: Would be good to flag not to do the post-processing
        else:
            config = experiment['config']
            if database:
                # Note that seed will not get passed onwards to BR/eval. I don't think this matters.
                command = f'python {manage_path} nfsp --seed {seed} --filename {game_name}.json --network_config_file {config} --experiment_name {base_job_name} --job_name "{experiment_name}" --dispatch_br true {overrides}'
            else:
                command = f'python {pydir}/ubc_nfsp_example.py --alsologtostderr -- --filename {game_name}.json --network_config_file {yml_config_dir}/{config}.yml --output_dir {output_dir} --dispatch_br true {overrides} --job_name {experiment_name}'

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        slurm_job_name = experiment_name + '_' + base_job_name
        job_file_text = f"""#!/bin/sh
#SBATCH --cpus-per-task={int(mem/4)}
#SBATCH --job-name={slurm_job_name}
#SBATCH --time=5-0:00:00 # days-hh:mm:ss
#SBATCH -e slurm-%j-{slurm_job_name}.err
#SBATCH -o slurm-%j-{slurm_job_name}.out

export OPENSPIEL_PATH={spiel_path}
export PYTHONPATH=${{OPENSPIEL_PATH}}:$PYTHONPATH
export PYTHONPATH=${{OPENSPIEL_PATH}}/build/python:$PYTHONPATH

CMD=`{command}`
echo $CMD
eval $CMD
"""
        write_and_submit(experiment_output_dir, experiment_name, job_file_text, submit)
    logging.info(f"Dispatched {len(experiments)} experiments!")


def dispatch_br_database(experiment_name, run_name, t, br_player, configs, submit=True, mem=16, overrides=''):
    if os.path.exists(config_path_from_config_name(configs)):
        dispatch_single_br_database(experiment_name, run_name, t, br_player, configs, submit, mem, overrides)
    else:
        # Multiple configs
        for br_config_path in glob.glob(f'{CONFIG_ROOT}/{configs}/*.yml'):
            dispatch_single_br_database(experiment_name, run_name, t, br_player, br_config_path.replace(f'{CONFIG_ROOT}/', '').replace('.yml', ''), submit, mem, overrides)


def dispatch_single_br_database(experiment_name, run_name, t, br_player, config, submit, mem, overrides):
    spiel_path, config_dir, pydir, manage_path = verify_config()
    command = f'python {manage_path} bestrespond --experiment_name {experiment_name} --run_name {run_name} --t {t} --br_player {br_player} --dispatch_rewards True {overrides} --config {config}'

    slurm_job_name = f'br_{br_player}_{experiment_name}_{run_name}_{t}_{config.replace("/", "_")}'
    job_file_text = f"""#!/bin/sh
#SBATCH --cpus-per-task={int(mem/4)}
#SBATCH --job-name={slurm_job_name}
#SBATCH --time=5-0:00:00 # days-hh:mm:ss
#SBATCH -e slurm-%j-{slurm_job_name}.err
#SBATCH -o slurm-%j-{slurm_job_name}.out

export OPENSPIEL_PATH={spiel_path}
export PYTHONPATH=${{OPENSPIEL_PATH}}:$PYTHONPATH
export PYTHONPATH=${{OPENSPIEL_PATH}}/build/python:$PYTHONPATH
CMD=`{command}`
echo $CMD
eval $CMD
"""

    experiment_dir = f'/shared/outputs/{experiment_name}/{run_name}/{BR_DIR}'
    write_and_submit(experiment_dir, slurm_job_name, job_file_text, submit)

    logging.info(f"Dispatched experiment!")

def dispatch_eval_database(experiment_name, run_name, t, br_player, br_name, submit=True, mem=8, overrides=''):
    spiel_path, config_dir, pydir, manage_path = verify_config()

    slurm_job_name = f'eval_{run_name}_{t}_{experiment_name}'
    command = f'python {manage_path} evaluate --experiment_name {experiment_name} --run_name {run_name} --t {t} {overrides}'
    if br_player is not None and br_name is not None:
        command += f' --br_name {br_name} --br_player {br_player}'
        slurm_job_name += f'_{safe_config_name(br_name)}_{br_player}'

    experiment_dir = f'/shared/outputs/{experiment_name}/{run_name}/{EVAL_DIR}'

    job_file_text = f"""#!/bin/sh
#SBATCH --cpus-per-task={int(mem/4)}
#SBATCH --job-name={slurm_job_name}
#SBATCH --time=5-0:00:00 # days-hh:mm:ss
#SBATCH -e slurm-%j-{slurm_job_name}.err
#SBATCH -o slurm-%j-{slurm_job_name}.out

export OPENSPIEL_PATH={spiel_path}
export PYTHONPATH=${{OPENSPIEL_PATH}}:$PYTHONPATH
export PYTHONPATH=${{OPENSPIEL_PATH}}/build/python:$PYTHONPATH
CMD=`{command}`
echo $CMD
eval $CMD
"""
    write_and_submit(experiment_dir, slurm_job_name, job_file_text, submit)
    logging.info(f"Dispatched experiment!")