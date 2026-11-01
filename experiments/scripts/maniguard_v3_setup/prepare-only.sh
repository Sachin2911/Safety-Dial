#!/usr/bin/env bash
set -euo pipefail
SETUP=/workspace/setup-maniguard-v3-20261030
cd "$SETUP"
mkdir -p logs
mkdir -p /workspace/runtime
chmod 700 /workspace/runtime
trap 'date -u +%FT%TZ > "$SETUP/preparation.failed"' ERR
bash stage-a.sh > logs/stage-a.log 2>&1
bash stage-b.sh > logs/stage-b.log 2>&1 &
MANIGUARD_V3_B_JOB=$!
bash stage-c.sh > logs/stage-c.log 2>&1 &
MANIGUARD_V3_C_JOB=$!
(
  while [[ ! -f "$SETUP/download-env.ready" ]]; do
    [[ ! -f "$SETUP/stage-b.failed" ]] || exit 1
    sleep 5
  done
  HF_HOME=/workspace/.cache/huggingface /workspace/envs/downloads/bin/python -u downloads.py
) > logs/downloads.log 2>&1 &
MANIGUARD_V3_DOWNLOAD_JOB=$!
wait "$MANIGUARD_V3_DOWNLOAD_JOB"
/workspace/envs/downloads/bin/python -u extract-assets.py > logs/extract-assets.log 2>&1
wait "$MANIGUARD_V3_B_JOB"
wait "$MANIGUARD_V3_C_JOB"
/workspace/miniforge3/envs/behavior/bin/python - <<'META'
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
setup=Path('/workspace/setup-maniguard-v3-20261030')
result={'status':'prepared_not_launched','experiment_started':False,
        'simulator_scenes_started':0,'model_queries':0,'episode_starts':0,
        'python':sys.version,
        'packages':sorted([[d.metadata['Name'],d.version] for d in metadata.distributions()
                           if d.metadata['Name']],key=lambda x:x[0].lower())}
for name,path in {'maniguard':'/workspace/Safety-Dial/third_party/ManiGuard',
                  'behavior_1k':'/workspace/Safety-Dial/third_party/ManiGuard/behavior-1k',
                  'openpi':'/workspace/Safety-Dial/third_party/openpi'}.items():
    result[name]=subprocess.run(['git','-C',path,'rev-parse','HEAD'],check=True,
                                capture_output=True,text=True).stdout.strip()
(setup/'prepared.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'experiment_started':False,
                  'simulator_scenes_started':0,'model_queries':0,'episode_starts':0}))
META
date -u +%FT%TZ > preparation.complete
