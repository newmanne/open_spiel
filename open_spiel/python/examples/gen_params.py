import json
import argparse
from sklearn.model_selection import ParameterGrid
import os
from pathlib import Path


CMD_FILE_NAME = 'cmds.txt'

def main(root, spiel_path):
    Path(f'{root}').mkdir(parents=True, exist_ok=True)
    Path(f'{root}/logs').mkdir(parents=True, exist_ok=True)

    V_L = 150
    B_L = 350
    V_H = 300
    B_H = 700

    low = {
      "value": [V_L],
      "value_probs": [1.],
      "budget": [B_L],
      "budget_probs": [1.]
    }
    high = {
      "value": [V_H],
      "value_probs": [1.],
      "budget": [B_H],
      "budget_probs": [1.]
    }
    mixed = {
      "value": [V_L, V_H],
      "value_probs": [0.5, 0.5],
      "budget": [B_L, B_H],
      "budget_probs": [0.5, 0.5]
    }

    param_grid = [
        {'opening_price': [100], 'increment': [0.05, 0.1, 0.2], 'licenses': [3, 4, 5], 'undersell_rule': [True, False]},
    ]
    i = 1
    cmds = []
    for players in [(low, mixed), (high, mixed), (mixed, mixed)]:
        for parameterization in ParameterGrid(param_grid):
            parameterization['players'] = players
            Path(f'{root}/{i}').mkdir(parents=True, exist_ok=True)
            with open(f'{root}/{i}/{i}.json', 'w') as f:
                json.dump(parameterization, f)
                for solver in ["cfr", "cfrplus", "cfrbr", "mccfr --sampling external", "mccfr --sampling outcome"]:
                    cmd = f'cd {root}/{i} && python {spiel_path}/open_spiel/python/examples/ubc_mccfr_cpp_example.py --filename={root}/{i}/{i}.json --iterations 10000 --solver={solver} --output {root}/{i}'
                    cmds.append(cmd)
            i += 1

    print(f"Dumped {i} configs to {root}")
    with open(f'{root}/{CMD_FILE_NAME}', 'w') as f:
        for cmd in cmds:
            f.write(cmd + '\n')
    print (f"Commands written to {root}/{CMD_FILE_NAME}")

    JOB_NAME = 'CFR'
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
export PYTHONPATH=${{PYTHONPATH}}:{spiel_path}
export PYTHONPATH=${{PYTHONPATH}}:{spiel_path}/build/python

CMD=`head -n $SLURM_ARRAY_TASK_ID {root}/{CMD_FILE_NAME} | tail -n 1`
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
    args = parser.parse_args()
    main(args.root, args.spiel_path)

    # SINGULARITY="singularity exec -B /home -B /project -B /scratch -B /localscratch /project/def-kevinlb/newmanne/openspiel.simg"
    # srun $SINGULARITY $CMD
