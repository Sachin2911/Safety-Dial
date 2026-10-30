# Source this file for paths shared by the simulator and policy commands.
export LC_ALL=C.UTF-8 LANG=C.UTF-8
export XDG_CACHE_HOME=/workspace/.cache
export HF_HOME=/workspace/.cache/huggingface
export OMNI_KIT_ACCEPT_EULA=YES PRIVACY_CONSENT=Y
export XDG_RUNTIME_DIR=/workspace/runtime
if [[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
elif [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi
export PYTHONPATH=/workspace/Safety-Dial/third_party/ManiGuard
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export PATH=/workspace/bin:$PATH
export MANIGUARD_PYTHON=/workspace/miniforge3/envs/behavior/bin/python
export OPENPI_PYTHON=/workspace/Safety-Dial/third_party/openpi/.venv/bin/python
