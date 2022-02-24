from open_spiel.python.examples.ubc_utils import *
from open_spiel.python.examples.ubc_dispatch import *


def dispatch_single_br(experiment_dir, br_player, checkpoint, submit, mem, overrides, config_path=None):
    spiel_path, config_dir, pydir = verify_config()
    command = f'python {pydir}/ubc_br.py --alsologtostderr -- --experiment_dir {experiment_dir} --br_player {br_player} --checkpoint {checkpoint} --dispatch_rewards True {overrides}'
    if config_path is not None:
        command += f' --config {config_path}'

    basename = os.path.basename(experiment_dir)
    slurm_job_name = f'br_{br_player}_{checkpoint}_{basename}'
    if config_path is not None:
        slurm_job_name += f'_{Path(config_path).stem}'
    job_file_text = f"""#!/bin/sh
#SBATCH --cpus-per-task={int(mem/4)}
#SBATCH --job-name={slurm_job_name}
#SBATCH --time=5-0:00:00 # days-hh:mm:ss
CMD=`{command}`
echo $CMD
eval $CMD
"""
    suffix = '' if config_path is None else f'_{Path(config_path).stem}'
    job_file_path = str(Path(f'{experiment_dir}/br_{br_player}_{checkpoint}{suffix}.sh').resolve())
    write_job_file(job_file_path, job_file_text)
    if submit:
        os.system(f'cd {experiment_dir} && sbatch {job_file_path}')

    print(f"Dispatched experiment!")

def dispatch_br(experiment_dir, br_player=0, checkpoint='checkpoint_latest', submit=True, mem=16, overrides='', br_portfolio_path=None):
    if br_portfolio_path is not None:
        for br_config_path in glob.glob(f'{br_portfolio_path}/*.yml'):
            dispatch_single_br(experiment_dir, br_player=br_player, checkpoint=checkpoint, submit=submit, mem=mem, overrides=overrides, config_path=br_config_path)
    else:
        dispatch_single_br(experiment_dir, br_player=br_player, checkpoint=checkpoint, submit=submit, mem=mem, overrides=overrides)

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
    job_file_path = str(Path(f'{experiment_dir}/{slurm_job_name}.sh').resolve())
    write_job_file(job_file_path, job_file_text)
    if submit:
        os.system(f'cd {experiment_dir} && sbatch {job_file_path}')

    print(f"Dispatched experiment!")
