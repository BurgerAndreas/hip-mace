#!/bin/bash
#SBATCH -A aip-aspuru
#SBATCH -D /scratch/aburger/hip-mace
#SBATCH --time=71:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=256GB
#SBATCH --job-name=hip-mace
# Jobs must write their output to your scratch or project directory (home is read-only on compute nodes).
#SBATCH --output=/scratch/aburger/hip-mace/outslurm/slurm-%j.txt
#SBATCH --error=/scratch/aburger/hip-mace/outslurm/slurm-%j.txt

# activate venv
source .venv/bin/activate

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

srun uv run "$@"
