#!/usr/bin/env bash
set +x
set -euo pipefail
umask 077
: "${RUNPOD_POD_ID:?Setup requires the rented RunPod host}"
export LC_ALL=C.UTF-8 LANG=C.UTF-8 DEBIAN_FRONTEND=noninteractive
export UV_CACHE_DIR=/tmp/pusht-uv-cache
cd /workspace/Safety-Dial
receipt_dir=outputs/runpod-pusht-tuning-20261029/environment
mkdir -p "$receipt_dir"
apt-get update -qq
apt-get install -y -qq --no-install-recommends libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 libegl1 libosmesa6 rsync zstd > "$receipt_dir/system-packages.log" 2>&1
python3 -m pip install --quiet uv
uv --no-config venv --python 3.12 /opt/lewm
uv --no-config pip install --python /opt/lewm/bin/python --torch-backend cu126 -r configs/lewm18/requirements.txt pytest ruff > "$receipt_dir/python-packages.log" 2>&1
uv --no-config pip check --python /opt/lewm/bin/python
uv --no-config pip freeze --python /opt/lewm/bin/python > "$receipt_dir/installed-freeze.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total,uuid --format=csv > "$receipt_dir/gpus.csv"
cat > /workspace/activate-lewm.sh <<'ACTIVATE'
source /opt/lewm/bin/activate
export STABLEWM_HOME=/workspace/Safety-Dial/data/safecem10/stablewm
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /workspace/Safety-Dial
ACTIVATE
source /workspace/activate-lewm.sh
python - <<'PY'
import hashlib, importlib.metadata as metadata, json, os, platform, torch
from pathlib import Path
root=Path.cwd()
design=json.loads((root/'configs/pusht_tuning_20261029/design-remote-v1.json').read_text())
assert os.environ['RUNPOD_POD_ID']==design['execution_pod_id']
assert torch.cuda.is_available()
assert float(torch.arange(64,device='cuda',dtype=torch.float32).sum())==2016
for name, expected in (('safety_checkpoint','safety_checkpoint_sha256'),('world_model_checkpoint','world_model_sha256')):
 assert hashlib.sha256((root/design[name]).read_bytes()).hexdigest()==design[expected]
record={'pod_id':os.environ['RUNPOD_POD_ID'],'python':platform.python_version(),'cuda_available':True,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'checkpoint_hashes_verified':True,'versions':{name:metadata.version(name) for name in ('stable-worldmodel','stable-pretraining','torch','torchvision','transformers','numpy','pymunk','gymnasium')}}
p=root/'outputs/runpod-pusht-tuning-20261029/environment/ready.json'
p.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
PY
