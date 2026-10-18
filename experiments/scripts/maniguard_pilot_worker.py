"""Run the unchanged pilot episode with exactly one visible physical GPU.

Each simulator sees only its routed GPU, reproducing the single-GPU host the pilot was
frozen on: CUDA device 0 is that GPU for PyTorch, the CLIP encoder, PhysX and Kit's
CUDA interop, and Kit selects the matching renderer by itself. With every GPU visible,
PhysX resets the CUDA current device to 0 between operations while the frozen encoder
follows the current device, which splits tensors across GPUs. OMNIGIBSON_GPU_ID must
therefore stay unset; it would pass one physical index to both Kit and PhysX.

Concurrent OmniGibson launches also copy the same experience and icon files into the
shared Isaac Sim apps directory, truncating them in place. Copies there are made
race-free: an identical target is left untouched and a differing one is replaced
atomically. Every other copy, and the episode itself, are unchanged.
"""

import argparse
import json
import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path


def guard_shared_kit_copies(copyfile=shutil.copyfile, environ=os.environ):
    def copy(source, destination, *, follow_symlinks=True):
        apps = environ.get('EXP_PATH')
        target = Path(destination)
        if not apps or target.resolve().parent != Path(apps).resolve():
            return copyfile(source, destination, follow_symlinks=follow_symlinks)
        data = Path(source).read_bytes()
        if target.is_file() and target.read_bytes() == data:
            return destination
        descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix=target.name + '.',
                                                 suffix='.tmp')
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o644)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return destination

    return copy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('episode', type=Path)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    expected = root / 'experiments/scripts/maniguard_phase2_episode.py'
    if args.episode.resolve() != expected or args.gpu < 0:
        raise ValueError('Worker requires the pinned pilot episode and a physical GPU')
    if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise ValueError('CUDA ordinals must follow the nvidia-smi PCI order')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.gpu):
        raise ValueError('The simulator must see exactly its routed physical GPU')
    if 'OMNIGIBSON_GPU_ID' in os.environ:
        raise ValueError('OMNIGIBSON_GPU_ID would give Kit and PhysX inconsistent ordinals')
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one CUDA device must be visible')
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    if 'GPU-' + uuid != args.gpu_uuid:
        raise RuntimeError('Visible CUDA device differs from the routed physical GPU')
    probe = torch.ones(1, device='cuda')
    torch.cuda.synchronize()
    print(json.dumps({'physical_gpu': args.gpu, 'gpu_uuid': 'GPU-' + uuid,
                      'torch_device': str(probe.device), 'gpu_name': torch.cuda.get_device_name(0),
                      'pid': os.getpid(), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                      'appdata': os.environ.get('OMNIGIBSON_APPDATA_PATH'),
                      'shared_kit_copy_guard': True}), flush=True)
    del probe
    shutil.copyfile = guard_shared_kit_copies()
    sys.argv = [str(expected), *args.arguments]
    runpy.run_path(str(expected), run_name='__main__')


if __name__ == '__main__':
    main()
