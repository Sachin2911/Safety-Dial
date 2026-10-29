#!/usr/bin/env bash
# Operational repair for the task 15 rental (25 Oct 2026): the first behavior_install
# attempt finished its conda env, torch and curobo steps, then BEHAVIOR's setup.sh lost
# one Isaac Sim wheel download to a transient network error (its curl has no retry).
# This resumes only the upstream installer with the identical flags, with curl retries
# set in ~/.curlrc, marks the phase done on success and reruns the preparation wrapper,
# which skips every finished phase.
set -euo pipefail
ROOT=/workspace/Safety-Dial
OUT="$ROOT/outputs/regrasp15"
mkdir -p "$OUT/logs/attempt1"
for f in prepare-failed.txt prepare-ended.txt; do
    if [ -f "$OUT/$f" ]; then mv "$OUT/$f" "$OUT/logs/attempt1/$f"; fi
done
cp "$ROOT/third_party/logs/prepare-behavior_install.log" "$OUT/logs/attempt1/" 2>/dev/null || true
printf 'retry = 8\nretry-delay = 5\nretry-all-errors\nconnect-timeout = 30\n' > ~/.curlrc
(
    set -x
    export CONDA_ENVS_PATH=/opt/miniforge3/envs CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
    set +u
    source /opt/miniforge3/etc/profile.d/conda.sh
    conda activate behavior
    set -u
    [[ "$CONDA_DEFAULT_ENV" == "behavior" ]]
    cd "$ROOT/third_party/ManiGuard/behavior-1k"
    ./setup.sh --omnigibson --bddl --joylo --dataset --eval --primitives \
        --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos --confirm-no-conda
    echo "SETUP_DONE (resumed)"; date -u
) > "$ROOT/third_party/logs/prepare-behavior_install-resume.log" 2>&1
touch "$ROOT/third_party/.prepare/behavior_install"
cd "$ROOT"
bash experiments/scripts/prepare_regrasp15.sh > "$OUT/logs/preparation-resume.log" 2>&1
