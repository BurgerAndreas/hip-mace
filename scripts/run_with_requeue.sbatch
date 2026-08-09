#!/usr/bin/env bash
# Portable Slurm launcher with preemption/time-limit requeue support.
# Submit from the repository root. Pass account/partition/time/resources via
# sbatch flags (or site defaults), for example:
#   sbatch -A ACCOUNT -p PARTITION -J JOBNAME scripts/run_with_requeue.sbatch \
#     mace/cli/run_train.py --config=configs/CONFIG.yaml --no_restart_latest
#
#SBATCH -J hip_mace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@1800
#SBATCH --output=outslurm/slurm-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p outslurm

export HIP_MACE_PROJECT_DIR="${HIP_MACE_PROJECT_DIR:-$PWD}"
export HIP_MACE_SCRATCH_DIR="${HIP_MACE_SCRATCH_DIR:-$PWD}"
export HIP_MACE_VENV="${HIP_MACE_VENV:-$PWD/.venv}"
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
preemption_grace_seconds="${HIP_MACE_PREEMPTION_GRACE_SECONDS:-1500}"
termination_grace_seconds="${HIP_MACE_TERMINATION_GRACE_SECONDS:-30}"
preemption_exit_code="${HIP_MACE_PREEMPTION_EXIT_CODE:-75}"

signal_child() {
  local signal_name="$1"
  if [[ -z "$child_pid" ]] || ! kill -0 "$child_pid" 2>/dev/null; then
    return
  fi

  if kill -0 -- "-$child_pid" 2>/dev/null; then
    kill "-$signal_name" -- "-$child_pid" 2>/dev/null || true
  else
    kill "-$signal_name" "$child_pid" 2>/dev/null || true
  fi
}

wait_for_child_exit() {
  local deadline="$1"
  while kill -0 "$child_pid" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 5
  done
}

finish_after_preemption_signal() {
  local child_status="$1"
  if [[ "$is_training_run" == true && "$child_status" -eq "$preemption_exit_code" ]]; then
    echo "Child saved a preemption checkpoint and exited with code ${child_status}; requeueing job ${SLURM_JOB_ID}." >&2
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi

  echo "Child exited with code ${child_status} after pre-timeout signal; not requeueing." >&2
  exit "$child_status"
}

requeue_before_timeout() {
  trap - USR1
  echo "Received pre-timeout signal for job ${SLURM_JOB_ID}; asking child to checkpoint." >&2
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    signal_child USR1

    deadline=$((SECONDS + preemption_grace_seconds))
    wait_for_child_exit "$deadline"

    if kill -0 "$child_pid" 2>/dev/null; then
      echo "Child did not exit after ${preemption_grace_seconds}s; terminating without automatic requeue unless it saves a checkpoint." >&2
      signal_child TERM
      deadline=$((SECONDS + termination_grace_seconds))
      wait_for_child_exit "$deadline"
      if kill -0 "$child_pid" 2>/dev/null; then
        echo "Child did not exit after TERM; killing process group and not requeueing." >&2
        signal_child KILL
      fi
    fi

    set +e
    wait "$child_pid"
    child_status=$?
    set -e
    finish_after_preemption_signal "$child_status"
  fi

  echo "No running child found after pre-timeout signal; not requeueing." >&2
  exit 0
}

trap requeue_before_timeout USR1

setsid uv run "${args[@]}" &
child_pid=$!
wait "$child_pid"
