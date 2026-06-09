#!/usr/bin/env bash
# Configure dynamic library paths for CUDA packages installed in the project venv.
#
# Usage:
#   source scripts/setup_cuda.sh
#   uv run python -c "import torch; print(torch.cuda.is_available())"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this script instead of executing it:"
  echo "  source scripts/setup_cuda.sh"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${HIP_MACE_VENV:-$REPO_ROOT/.venv}"
SITE_PACKAGES_BASE="$VENV_DIR/lib"

if [[ ! -d "$SITE_PACKAGES_BASE" ]]; then
  echo "Could not find venv lib directory at $SITE_PACKAGES_BASE." >&2
  echo "Set HIP_MACE_VENV or run uv sync first." >&2
  return 1 2>/dev/null || exit 1
fi

NVPKG=""
for d in "$SITE_PACKAGES_BASE"/python*/site-packages/nvidia; do
  if [[ -d "$d" ]]; then
    NVPKG="$d"
    break
  fi
done

if [[ -z "$NVPKG" ]]; then
  echo "Could not find nvidia libs in $VENV_DIR site-packages." >&2
  echo "Install GPU deps in the project environment first." >&2
  return 1 2>/dev/null || exit 1
fi

_append_path_unique() {
  local add_path="$1"
  local current="${2:-}"
  case ":${current}:" in
    *":${add_path}:"*) printf "%s" "$current" ;;
    *)
      if [[ -n "$current" ]]; then
        printf "%s:%s" "$add_path" "$current"
      else
        printf "%s" "$add_path"
      fi
      ;;
  esac
}

_filter_existing_path_list() {
  local current="${1:-}"
  local filtered=""
  local entry
  IFS=":" read -r -a entries <<< "$current"
  for entry in "${entries[@]}"; do
    [[ -z "$entry" ]] && continue
    case "$entry" in
      *"/hip-mace/.venv/"*|*"/hip-mace/.venv-py312/"*) continue ;;
    esac
    [[ -e "$entry" ]] || continue
    if [[ -n "$filtered" ]]; then
      filtered="${filtered}:$entry"
    else
      filtered="$entry"
    fi
  done
  printf "%s" "$filtered"
}

LD_LIBRARY_PATH="$(_filter_existing_path_list "${LD_LIBRARY_PATH:-}")"
LIBRARY_PATH="$(_filter_existing_path_list "${LIBRARY_PATH:-}")"
LD_PRELOAD="$(_filter_existing_path_list "${LD_PRELOAD:-}")"

lib_dirs=()
for subdir in cublas cuda_cupti cuda_nvrtc cuda_runtime cudnn cufft curand cusolver cusparse nccl nvjitlink; do
  if [[ -d "$NVPKG/$subdir/lib" ]]; then
    lib_dirs+=("$NVPKG/$subdir/lib")
  fi
done
for d in "$NVPKG"/*/lib; do
  if [[ -d "$d" ]]; then
    lib_dirs+=("$d")
  fi
done

for d in "${lib_dirs[@]}"; do
  LD_LIBRARY_PATH="$(_append_path_unique "$d" "${LD_LIBRARY_PATH:-}")"
  LIBRARY_PATH="$(_append_path_unique "$d" "${LIBRARY_PATH:-}")"
done
export LD_LIBRARY_PATH
export LIBRARY_PATH

if [[ -f "$NVPKG/cusolver/lib/libcusolver.so.11" ]]; then
  LD_PRELOAD="$(_append_path_unique "$NVPKG/cusolver/lib/libcusolver.so.11" "${LD_PRELOAD:-}")"
  export LD_PRELOAD
fi

echo "Configured CUDA library paths from:"
echo "  $NVPKG"
