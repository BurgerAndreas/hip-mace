#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J hip_mace
#SBATCH -p batch_singlenode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --output=/lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip-mace/outslurm/slurm-%j.out

set -euo pipefail

cd /lustre/fs12/portfolios/nvr/projects/nvr_qualg_lmbm/users/anburger/hip-mace

export HIP_MACE_PROJECT_DIR="$PWD"
export HIP_MACE_SCRATCH_DIR="$PWD"
export HIP_MACE_VENV="$PWD/.venv"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

source scripts/setup_cuda.sh

if [[ "$#" -eq 0 ]]; then
  echo "Usage: sbatch $0 <uv-run-args...>" >&2
  exit 2
fi

exec uv run "$@"
