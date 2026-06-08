#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /scratch/aburger/hip-mace
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:h100:1 #gpu:l40s:1
#SBATCH --mem=128GB
#SBATCH --job-name=hip-mace
#SBATCH --requeue
# Jobs must write their output to your scratch or project directory (home is read-only on compute nodes).
#SBATCH --output=/scratch/aburger/hip-mace/outslurm/slurm-%j.txt
#SBATCH --error=/scratch/aburger/hip-mace/outslurm/slurm-%j.txt

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export WANDB_RESUME=allow

requeue_requested=0
child_pid=""

request_requeue() {
  if [[ "$requeue_requested" -eq 1 ]]; then
    return
  fi
  requeue_requested=1
  echo "$(date): Received requeue signal for job ${SLURM_JOB_ID}; terminating child and requeueing."
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}

trap request_requeue USR1

# activate venv
#source .venv/bin/activate
uv sync

which python

# get environment variables
# source .env
# export JAX_PLATFORM_NAME=gpu

#module load cuda/12.6
#module load gcc/12.3

# append command to slurmlog.txt
echo "sbatch scripts/killarney.sh $@ # $SLURM_JOB_ID" >> slurmlog.txt

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo "Inside slurm_launcher.slrm ($0). received arguments: $@"

# hand over all arguments to the script
pwd
echo "Submitting $@"

srun uv run "$@" &
child_pid=$!
wait "$child_pid"
