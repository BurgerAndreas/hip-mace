#!/usr/bin/env bash
#SBATCH -A nvr_qualg_lmbm
#SBATCH -J hip_mace
#SBATCH -p batch_singlenode
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
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

args=("$@")
is_training_run=false
has_restart_arg=false
for arg in "${args[@]}"; do
  if [[ "$arg" == "mace/cli/run_train.py" || "$arg" == */mace/cli/run_train.py ]]; then
    is_training_run=true
  fi
  if [[ "$arg" == "--restart_latest" || "$arg" == "--no_restart_latest" ]]; then
    has_restart_arg=true
  fi
done

if [[ "$is_training_run" == true && "${SLURM_RESTART_COUNT:-0}" -gt 0 ]]; then
  for idx in "${!args[@]}"; do
    if [[ "${args[$idx]}" == "--no_restart_latest" ]]; then
      echo "SLURM_RESTART_COUNT=${SLURM_RESTART_COUNT}; replacing --no_restart_latest with --restart_latest for requeued training job." >&2
      args[$idx]="--restart_latest"
      has_restart_arg=true
    fi
  done
fi

if [[ "$is_training_run" == true && "$has_restart_arg" == false ]]; then
  args+=("--restart_latest")
fi

child_pid=""
preemption_grace_seconds="${HIP_MACE_PREEMPTION_GRACE_SECONDS:-240}"
requeue_before_timeout() {
  trap - USR1 TERM
  echo "Received pre-timeout signal for job ${SLURM_JOB_ID}; asking child to checkpoint before requeueing." >&2
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    if kill -0 -- "-$child_pid" 2>/dev/null; then
      kill -USR1 -- "-$child_pid" 2>/dev/null || true
    else
      kill -USR1 "$child_pid" 2>/dev/null || true
    fi

    deadline=$((SECONDS + preemption_grace_seconds))
    while kill -0 "$child_pid" 2>/dev/null && (( SECONDS < deadline )); do
      sleep 5
    done

    if kill -0 "$child_pid" 2>/dev/null; then
      echo "Child did not exit after ${preemption_grace_seconds}s; terminating before requeue." >&2
      if kill -0 -- "-$child_pid" 2>/dev/null; then
        kill -TERM -- "-$child_pid" 2>/dev/null || true
      else
        kill -TERM "$child_pid" 2>/dev/null || true
      fi
    fi
    wait "$child_pid" 2>/dev/null || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}

trap requeue_before_timeout USR1

setsid uv run "${args[@]}" &
child_pid=$!
wait "$child_pid"
