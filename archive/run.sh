#!/usr/bin/env bash
# Run a historical command with the legacy layout and shared root environment.
set -euo pipefail

if [[ $# -eq 0 || "${1:-}" == "--help" ]]; then
  echo "Usage: bash archive/run.sh COMMAND [ARGUMENTS...]"
  echo "Example: bash archive/run.sh python -m pytest -q"
  exit 0
fi

archive_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${archive_dir}/.." && pwd)"

cd -- "${archive_dir}/legacy"
exec uv run --project "${project_root}" --no-sync -- "$@"
