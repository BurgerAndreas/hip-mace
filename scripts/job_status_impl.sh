#!/usr/bin/env bash
# Fallback impl when skill script is not installed.
set -euo pipefail
USER_NAME="${USER:-$(whoami)}"
JOBID="${1:-}"
LOG_DIR="${2:-${HIP_MACE_PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/outslurm}"
TAIL_LINES="${TAIL_LINES:-8}"
fmt='%.10i %.9P %.25j %.8T %.10M %.6D %R'

if [[ -z "$JOBID" ]]; then
  timeout 10s squeue -u "$USER_NAME" -o "$fmt" 2>/dev/null || squeue -u "$USER_NAME"
  exit 0
fi

timeout 10s squeue -j "$JOBID" -o "$fmt" 2>/dev/null || echo "(not in queue)"
timeout 15s sacct -j "$JOBID" --format=JobID,JobName,Partition,State,Elapsed,ExitCode,SubmitLine -P -n 2>/dev/null | head -5
LOG="${LOG_DIR%/}/slurm-${JOBID}.out"
if [[ -f "$LOG" ]]; then
  tail -n "$TAIL_LINES" "$LOG"
  grep -m1 'Command line arguments:' "$LOG" 2>/dev/null || true
fi
