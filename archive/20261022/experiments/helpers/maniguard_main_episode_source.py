"""Checked main hooks layered on the existing phase-two/original/AAC generators."""
import hashlib
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def instrument_source(source, backend):
    def replace(before, after):
        nonlocal source
        if source.count(before) != 1:
            raise ValueError('Main source anchor is not unique: ' + before[:100])
        source = source.replace(before, after, 1)

    if backend == 'aac':
        replace('from helpers.maniguard_aac_runtime import get_runtime\nAAC = get_runtime()',
                'from helpers.maniguard_main_episode_runtime import get_runtime\n'
                'MAIN = get_runtime()\nAAC = MAIN.study')
        replace('    AAC.validate_config(cfg)\n',
                '    AAC.validate_config(cfg)\n    MAIN.validate_config(cfg)\n')
    elif backend in ('phase2', 'benchmark_original'):
        module = 'maniguard_phase2_runtime' if backend == 'phase2' else 'maniguard_runtime'
        replace(f'from helpers.{module} import STUDY',
                'from helpers.maniguard_main_episode_runtime import get_runtime\n'
                'MAIN = get_runtime()\nSTUDY = MAIN.study')
        replace('    resolve_source_config(cfg)\n',
                '    resolve_source_config(cfg)\n    MAIN.validate_config(cfg)\n')
        if backend == 'benchmark_original':
            replace('STUDY.start(scene_info, obs)', 'STUDY.start(scene_info, obs, env, robot, monitor)')
    else:
        raise ValueError('Unsupported main evaluator backend')
    replace('                    rollout_diagnostics["phase"] = "env_step"\n',
            '                    MAIN.before_environment_action(step_idx, chunk[ci], ctrl_action,\n'
            '                                                   action_clipped, action_space.low, action_space.high)\n'
            '                    rollout_diagnostics["phase"] = "env_step"\n')
    replace('                    rollout_diagnostics["env_steps_returned"] += 1\n',
            '                    MAIN.after_environment_action(step_idx)\n'
            '                    rollout_diagnostics["env_steps_returned"] += 1\n')
    replace('        os._exit(1)\n', '        MAIN.finalize_exit()\n        os._exit(1)\n')
    replace('    os._exit(1 if _failed else 0)\n',
            '    MAIN.finalize_exit()\n    os._exit(1 if _failed else 0)\n')
    compile(source, 'benchmark_main.py', 'exec')
    return source


def prepare_runtime(output, backend, *, root=ROOT):
    output, root = Path(output), Path(root)
    if backend == 'aac':
        from helpers.maniguard_aac_source import prepare_runtime as prepare
        base = prepare(output, root=root)
        provenance_path = output / 'aac-source-provenance.json'
    else:
        # Proven generators currently bind their checkout through module ROOT. Never
        # silently generate from a different repository than the authorized root.
        from helpers import maniguard_source
        if root.resolve() != maniguard_source.ROOT.resolve():
            raise ValueError('Legacy source generator requires its actual source checkout')
        if backend == 'phase2':
            from helpers.maniguard_phase2_source import prepare_runtime as prepare
        elif backend == 'benchmark_original':
            prepare = maniguard_source.prepare_runtime
        else:
            raise ValueError('Unsupported main evaluator backend')
        base = prepare(output)
        provenance_path = output / 'runtime-provenance.json'
    source = instrument_source(base.read_text(), backend)
    destination = output / 'benchmark_main.py'
    with destination.open('x') as stream:
        stream.write(source)
    provenance = json.loads(provenance_path.read_text())
    provenance.update(base_runtime_sha256=provenance['runtime_sha256'], backend=backend,
                      runtime_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      main_source_hook_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    final_provenance = output / 'main-source-provenance.json'
    with final_provenance.open('x') as stream:
        json.dump(provenance, stream, sort_keys=True)
        stream.write('\n')
    return destination


def reconstruct_runtime(root, backend):
    """Rebuild checked source in a private temporary directory; never execute it."""
    with tempfile.TemporaryDirectory(prefix='maniguard-main-source-') as directory:
        return prepare_runtime(directory, backend, root=root).read_text()
