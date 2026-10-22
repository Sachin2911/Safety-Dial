#!/usr/bin/env bash
# Usage: bash scripts/baseline_unisafe.sh [NEW_OUTPUT_DIRECTORY] [RUNNER_OPTIONS...]
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
run_dir="${1:-runs/unisafe/baseline-$(date -u +%Y%m%dT%H%M%SZ)}"
if (( $# > 0 )); then shift; fi
if [[ -e "$run_dir" ]]; then
    echo "Output already exists; choose a new directory: $run_dir" >&2
    exit 1
fi
/venv/main/bin/python -u experiments/scripts/unisafe_baseline.py --output "$run_dir" "$@"
/venv/main/bin/python experiments/scripts/unisafe_analyze.py "$run_dir"
printf 'UNISafe baseline and analysis completed: %s\n' "$run_dir"
