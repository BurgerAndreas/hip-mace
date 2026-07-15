#!/usr/bin/env bash
# Repo-local wrapper — delegates to the personal skill script when present.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HIP_MACE_PROJECT_DIR="${HIP_MACE_PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SKILL_SCRIPT="$HOME/.cursor/skills/nvr-slurm-partitions/scripts/job_status.sh"
if [[ -x "$SKILL_SCRIPT" ]]; then
  exec "$SKILL_SCRIPT" "$@"
fi
exec "$SCRIPT_DIR/job_status_impl.sh" "$@"
