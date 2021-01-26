import json
import argparse
from sklearn.model_selection import ParameterGrid
import os
from pathlib import Path


BASE_DIR = 'configs'
CMD_FILE_NAME = 'cmds.txt'

def main():
    if not os.path.exists(BASE_DIR):
        os.mkdir(BASE_DIR)

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
        {'opening_price': [100], 'increment': [0.05, 0.1, 0.2], 'licenses': [3, 4, 5], 'undersell_rule': [False]},
    ]
    i = 1
    cmds = []
    for players in [(low, mixed), (high, mixed), (mixed, mixed)]:
        for parameterization in ParameterGrid(param_grid):
            parameterization['players'] = players
            Path(f'{BASE_DIR}/{i}').mkdir(parents=True, exist_ok=True)
            with open(f'{BASE_DIR}/{i}/{i}.json', 'w') as f:
                json.dump(parameterization, f)
                for solver in ["cfr", "cfrplus", "cfrbr", "mccfr --sampling external", "mccfr --sampling outcome"]:
                    cmd = f'python open_spiel/python/examples/ubc_mccfr_cpp_example.py --filename={BASE_DIR}/{i}.json --iterations 10000 --solver={solver} --output {BASE_DIR}/{i}'
                    cmds.append(cmd)
            i += 1

    print(f"Dumped {i} configs to {BASE_DIR}")
    with open(f'{BASE_DIR}/{CMD_FILE_NAME}', 'w') as f:
        for cmd in cmds:
            f.write(cmd + '\n')
    print (f"Commands written to {BASE_DIR}/{CMD_FILE_NAME}")

    JOB_NAME = 'CFR'
    slurm = f"""#!/bin/sh
    #SBATCH --cpus-per-task=1
    #SBATCH --mem-per-cpu=4G
    #SBATCH --job-name={JOB_NAME}
    #SBATCH --output={JOB_NAME}-%A_%a.out-o.txt
    #SBATCH --error={JOB_NAME}-%A_%a.out-e.txt
    #SBATCH --account=rrg-kevinlb
    #SBATCH --time=1-0
    #SBATCH --array=1-{len(cmds)}

    SINGULARITY="singularity exec -B /home -B /project -B /scratch -B /localscratch /project/def-kevinlb/newmanne/openspiel.simg"
    CMD=`head -n $SLURM_ARRAY_TASK_ID {BASE_DIR}/{CMD_FILE_NAME} | tail -n 1`
    srun $SINGULARITY $CMD
    """
    JOB_FILE = 'job_runner.sh'
    with open(f'{BASE_DIR}/{JOB_FILE}', 'w') as f:
        f.write(slurm)
    os.chmod(f"{BASE_DIR}/{JOB_FILE}", 777)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Queue up a bunch of CFR jobs')
    args = parser.parse_args()
    main()