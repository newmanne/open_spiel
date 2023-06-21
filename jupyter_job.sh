#!/bin/bash
#SBATCH --partition ada_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 32
#SBATCH --mem 80G
#SBATCH --time UNLIMITED
#SBATCH --job-name jupyter
#SBATCH --output /global/scratch/open_spiel/jupyter/logs/slurm-%j.out
 
################################################################################

# Override Jupyter folders to use scratch instead of read-only user dir
export JUPYTER_BASE_DIR=/global/scratch/open_spiel/jupyter
export JUPYTER_RUNTIME_DIR=$JUPYTER_BASE_DIR/runtime 
export JUPYTER_CONFIG_DIR=$JUPYTER_BASE_DIR/config
export JUPYTER_DATA_DIR=$JUPYTER_BASE_DIR/data
export JUPYTER_TMP_DIR=$JUPYTER_BASE_DIR/tmp
 
# Load Python
source activate py38
 
# Set up environment variables (port, token)
export RANDFILE=$JUPYTER_TMP_DIR/.rnd
export JUPYTER_TOKEN=$(openssl rand -base64 15)
readonly PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export XDG_RUNTIME_DIR=""

# Write diagnostics + SSH command to output
cd $JUPYTER_BASE_DIR
cat > connection.txt <<END
Job info:
$(date)
$(pwd)
Job ID: $SLURM_JOB_ID
Node: $HOSTNAME

SSH info and token:
ssh -NL 8888:$HOSTNAME:$PORT $USER@$(hostname)
$HOSTNAME:$PORT
$JUPYTER_TOKEN
http://localhost:8888/?token=$JUPYTER_TOKEN

To end job:
scancel $SLURM_JOB_ID
END

#python ${OPENSPIEL_PATH}/web/auctions/manage.py shell_plus --lab --no-browser
jupyter lab --no-browser --ip=0.0.0.0 --port=$PORT --notebook-dir=/global/scratch/open_spiel/open_spiel/notebooks
