#!/bin/bash
# Unattended end of task-14 stage C. When the remote runner exits: sync, stop the rental, and,
# only if stage C completed cleanly, analyze it, preserve the evidence privately by pinned
# revision and destroy the instance with the preservation marker. Any failure halts the
# remaining steps (the instance stays stopped, never destroyed without a marker). Log:
# outputs/cube14/control/finalize.log. Launch detached so it survives the session:
#   setsid nohup bash experiments/scripts/finalize_cube14.sh > /dev/null 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
LOG=outputs/cube14/control/finalize.log
PY="env -u PYTHONPATH .venv/bin/python"
M="$PY experiments/scripts/manage_cube14.py"
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

log "finalizer started (pid $$)"
state=""
for i in $(seq 1 200); do
    state=$(timeout 90 $M ssh --timeout 60 "cd /workspace/Safety-Dial/outputs/cube14 && if [ -f run-C-ended.txt ]; then if [ -f stage-C-complete.json ]; then echo COMPLETE; else echo ENDED; fi; else echo RUNNING; fi" 2>/dev/null | tail -1)
    if [ "$state" = COMPLETE ] || [ "$state" = ENDED ]; then
        break
    fi
    if [ $((i % 6)) -eq 0 ]; then
        timeout 1800 $M sync >> "$LOG" 2>&1 || log "periodic sync failed"
    fi
    sleep 300
done
log "runner state: ${state:-unknown}"
if [ "$state" != COMPLETE ] && [ "$state" != ENDED ]; then
    log "runner never reported an end; leaving the rental to its watchdogs"
    exit 1
fi

if ! timeout 1800 $M sync >> "$LOG" 2>&1; then
    log "final sync failed; instance left running for manual recovery"
    exit 1
fi
log "final sync done"
if ! timeout 600 $M stop >> "$LOG" 2>&1; then
    log "stop failed"
    exit 1
fi
log "instance stopped"
if [ "$state" != COMPLETE ] || [ ! -f outputs/cube14/remote/stage-C-complete.json ]; then
    log "stage C did not complete cleanly; evidence kept locally, instance stopped, not destroyed"
    exit 1
fi

$PY experiments/scripts/analyze_cube14.py --stage C >> "$LOG" 2>&1 || log "stage C analysis failed; preserving the raw evidence anyway"
if ! $PY experiments/scripts/preserve_cube14.py --upload >> "$LOG" 2>&1; then
    log "preservation failed; instance stopped, not destroyed"
    exit 1
fi
if [ ! -f progress/receipts/14-cube-preservation.json ]; then
    log "no preservation marker; instance stopped, not destroyed"
    exit 1
fi
log "preserved"
if timeout 600 $M destroy --marker progress/receipts/14-cube-preservation.json >> "$LOG" 2>&1; then
    log "destroyed"
else
    log "destroy failed; instance is stopped"
    exit 1
fi
log "finalizer done"
