#!/usr/bin/env bash
# Usage: bash scripts/smoke_unisafe.sh [NEW_OUTPUT_DIRECTORY] [SEED]
# Runs only frozen recorded-data inference and the simulator acceptance tests.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
run_dir="${1:-runs/unisafe/smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
seed="${2:-0}"
if [[ -e "$run_dir" ]]; then
    echo "Output already exists; choose a new directory: $run_dir" >&2
    exit 1
fi
mkdir -p "$run_dir"
/venv/main/bin/python -u experiments/scripts/unisafe_inference.py \
    --seed "$seed" --output "$run_dir/inference" 2>&1 | tee "$run_dir/inference.log"
/venv/main/bin/python -u experiments/scripts/unisafe_environment.py \
    --seed "$seed" --output "$run_dir/environment" 2>&1 | tee "$run_dir/environment.log"
printf 'Both UNISafe checks passed. Results: %s\n' "$run_dir"
