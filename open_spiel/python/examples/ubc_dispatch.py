import pandas as pd
import numpy as np
from absl import logging
import glob
import os
from pathlib import Path
from open_spiel.python.examples.ubc_utils import EVAL_DIR, BR_DIR, CONFIG_ROOT, config_path_from_config_name, safe_config_name, random_string
import time

BASE_OUTPUT_DIR = os.environ['CLOCK_AUCTION_OUTPUT_ROOT']
CLUSTER = os.environ.get('SPIEL_CLUSTER', 'ada')

def get_cluster_details(cluster):
    if cluster == 'ada_vickrey':
        return {
            'preamble': """#SBATCH --partition=ada_cpu_long,vickrey
#SBATCH --cpus-per-task=4
#SBATCH --mem 20G""",
            'load_py': """source ~/.bashrc
source activate py38""",
            'shell': '#!/bin/bash'
        }
    elif cluster == 'ada':
        return {
            'preamble': """#SBATCH --partition=ada_cpu_long
#SBATCH --cpus-per-task=4
#SBATCH --mem 20G""",
            'load_py': """source ~/.bashrc
source activate py38""",
            'shell': '#!/bin/bash'
        }
    elif cluster == 'ada_cpu_short':
        return {
            'preamble': """#SBATCH --partition=ada_cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem 20G""",
            'load_py': """source ~/.bashrc
source activate py38""",
            'shell': '#!/bin/bash'
        }
    else: # RONIN
        # Slurm on RONIN doesn't repsect memory issues, so we just isolate one job per node
        return {
            'preamble': '#SBATCH --cpus-per-task=8',
            'load_py': """export PYTHONPATH=${OPENSPIEL_PATH}:$PYTHONPATH
export PYTHONPATH=${OPENSPIEL_PATH}/build/python:$PYTHONPATH""",
            'shell': '#!/bin/sh'
        }
    
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
    # print(f"Writing job to {job_file_path}")
    if submit:
        if CLUSTER == 'ada':
            command = f'cd {experiment_output_dir} && /opt/slurm/bin/sbatch {job_file_path}'
            # print(command)
            os.system(command)
        else:
            os.system(f'cd {experiment_output_dir} && sbatch {job_file_path}')

def dispatch_experiments(yml_config, base_job_name=None, game_name='parking_1', submit=True, overrides='', cfr_also=False, database=True, n_seeds=1, start_seed=100, alg='ppo', extra_name='', cluster=CLUSTER):
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
                    'name': game_name.replace('/', '_') + '-' + safe_config_name(config) + ('-' + extra_name if extra_name else '') + '-' + str(seed),
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
    cluster_details = get_cluster_details(cluster)

    experiment_output_dir = f'{BASE_OUTPUT_DIR}/{base_job_name}'

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
                command = f'python {manage_path} {alg} --seed {seed} --filename {game_name}.json --network_config_file {config} --experiment_name {base_job_name} --job_name "{experiment_name}" {overrides}'
            else:
                command = f'python {pydir}/ubc_nfsp_example.py --alsologtostderr -- --filename {game_name}.json --network_config_file {yml_config_dir}/{config}.yml --output_dir {output_dir} --dispatch_br true {overrides} --job_name {experiment_name}'

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if CLUSTER == 'ada':
            time.sleep(0.3) # FS is stupid and sometimes you need to wait or the file won't exist.

        slurm_job_name = experiment_name + '_' + base_job_name
        job_file_text = f"""{cluster_details['shell']}
{cluster_details['preamble']}
#SBATCH --job-name={slurm_job_name}
#SBATCH --time=5-0:00:00 # days-hh:mm:ss
#SBATCH -e slurm-%j-{slurm_job_name}.err
#SBATCH -o slurm-%j-{slurm_job_name}.out

set -e

{cluster_details['load_py']}
export OPENSPIEL_PATH={spiel_path}

CMD=`{command}`
echo $CMD
eval $CMD
"""
        write_and_submit(experiment_output_dir, experiment_name, job_file_text, submit)
    logging.info(f"Dispatched {len(experiments)} experiments!")


def dispatch_br_database(experiment_name, run_name, t, br_player, configs, submit=True, overrides=''):
    if os.path.exists(config_path_from_config_name(configs)):
        dispatch_single_br_database(experiment_name, run_name, t, br_player, configs, submit, overrides)
    else:
        # Multiple configs
        for br_config_path in glob.glob(f'{CONFIG_ROOT}/{configs}/*.yml'):
            dispatch_single_br_database(experiment_name, run_name, t, br_player, br_config_path.replace(f'{CONFIG_ROOT}/', '').replace('.yml', ''), submit, overrides)


def dispatch_single_br_database(experiment_name, run_name, t, br_player, config, submit, overrides, django_command='ppo_br', cluster=CLUSTER):
    spiel_path, config_dir, pydir, manage_path = verify_config()
    cluster_details = get_cluster_details(cluster)
    command = f'python {manage_path} {django_command} --experiment_name {experiment_name} --run_name {run_name} --t {t} --br_player {br_player} --dispatch_rewards True {overrides} --config {config}'

    slurm_job_name = f'br_{br_player}_{experiment_name}_{run_name}_{t}_{config.replace("/", "_")}'
    job_file_text = f"""{cluster_details['shell']}
{cluster_details['preamble']}
#SBATCH --job-name={slurm_job_name}
#SBATCH --time=0-2:00:00 # days-hh:mm:ss
#SBATCH -e slurm-%j-{slurm_job_name}.err
#SBATCH -o slurm-%j-{slurm_job_name}.out

{cluster_details['load_py']}
export OPENSPIEL_PATH={spiel_path}

CMD=`{command}`
echo $CMD
eval $CMD
"""

    experiment_dir = f'{BASE_OUTPUT_DIR}/{experiment_name}/{run_name}/{BR_DIR}'
    write_and_submit(experiment_dir, slurm_job_name, job_file_text, submit)

    logging.info(f"Dispatched experiment!")

def dispatch_eval_database(t, experiment_name, run_name, br_mapping=dict(), submit=True, overrides='', django_command='ppo_eval', cluster=CLUSTER):
    spiel_path, config_dir, pydir, manage_path = verify_config()
    cluster_details = get_cluster_details(cluster)

    slurm_job_name = f'eval_{run_name}_{t}_{experiment_name}'
    br_mapping_str = str(br_mapping)
    command = f'python {manage_path} {django_command} --experiment_name {experiment_name} --run_name {run_name} --t {t} {overrides} --br_mapping "{br_mapping_str}"'
    slurm_job_name += '_' + random_string(10)

    experiment_dir = f'{BASE_OUTPUT_DIR}/{experiment_name}/{run_name}/{EVAL_DIR}'
    Path(experiment_dir).mkdir(parents=True, exist_ok=True) 
    print(experiment_dir)

    job_file_text = f"""{cluster_details['shell']}
{cluster_details['preamble']}
#SBATCH --job-name={slurm_job_name}
#SBATCH --time=5-0:00:00 # days-hh:mm:ss
#SBATCH -e slurm-%j-{slurm_job_name}.err
#SBATCH -o slurm-%j-{slurm_job_name}.out

{cluster_details['load_py']}
export OPENSPIEL_PATH={spiel_path}

CMD=`{command}`
echo $CMD
eval $CMD
"""
    write_and_submit(experiment_dir, slurm_job_name, job_file_text, submit)
    logging.info(f"Dispatched experiment!")

def dispatch_from_checkpoint(checkpoint_pk, game_name, config, experiment_name, base_job_name, overrides='', cluster=CLUSTER):
    overrides += f' --parent_checkpoint_pk {checkpoint_pk}'
    spiel_path, config_dir, pydir, manage_path = verify_config()
    cluster_details = get_cluster_details(cluster)

    experiment_output_dir = f'{BASE_OUTPUT_DIR}/{base_job_name}'
    output_dir = f'{experiment_output_dir}/{experiment_name}'

    command = f'python {manage_path} ppo --seed 1234 --filename {game_name}.json --network_config_file {config} --experiment_name {base_job_name} --job_name "{experiment_name}" {overrides}'

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    slurm_job_name = experiment_name + '_' + base_job_name
    job_file_text = f"""{cluster_details['shell']}
{cluster_details['preamble']}
#SBATCH --job-name={slurm_job_name}
#SBATCH --time=5-0:00:00 # days-hh:mm:ss
#SBATCH -e slurm-%j-{slurm_job_name}.err
#SBATCH -o slurm-%j-{slurm_job_name}.out

{cluster_details['load_py']}
export OPENSPIEL_PATH={spiel_path}

CMD=`{command}`
echo $CMD
eval $CMD
"""
    write_and_submit(experiment_output_dir, experiment_name, job_file_text, True)
