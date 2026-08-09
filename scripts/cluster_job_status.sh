#!/usr/bin/env bash
# Thin wrapper around scripts/job_status_impl.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export HIP_MACE_PROJECT_DIR="${HIP_MACE_PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
exec "$SCRIPT_DIR/job_status_impl.sh" "$@"
