"""Exact-job child preparation and backend selection for a frozen main protocol.

The CLI has no independent scene/actor/horizon switches and no boolean reserved-data
escape hatch. Authorization, inherited parent ownership, source/checkpoint checks and
backend capability checks precede output creation. Native execution remains closed
until the separately pinned physics driver accepts its qualification contract.
"""
import hashlib
import json
import math
import os
import runpy
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

from helpers import maniguard_main_protocol as protocol_api
from helpers.maniguard_main_episode_meter import atomic, counts, signed
from helpers.maniguard_phase2_model import genome_hash, identity_genome, load_model, save_model
from helpers.maniguard_phase2_optim import record_hash

ROOT = Path(__file__).resolve().parents[2]
REQUEST_SCHEMA = 'maniguard-main-episode-request-v1'
RUNTIME_SCHEMA = 'maniguard-main-runtime-contract-v1'
PROFILES = {'phase2_learned', 'phase2_identity', 'phase2_fixed', 'phase2_contextual',
            'benchmark_original', 'aac_joint20', 'aac_fixed8_joint20'}
COMMON_SOURCES = (
    'experiments/helpers/maniguard_main_episode.py',
    'experiments/helpers/maniguard_main_episode_meter.py',
    'experiments/helpers/maniguard_main_episode_runtime.py',
    'experiments/helpers/maniguard_main_episode_source.py',
    'experiments/helpers/maniguard_main_protocol.py',
    'experiments/helpers/maniguard_main_launcher.py',
    'experiments/helpers/maniguard_main_dispatch.py',
    'experiments/helpers/maniguard_main_physics.py',
    'experiments/helpers/maniguard_phase2_services.py',
    'experiments/helpers/maniguard_phase2_aac.py',
    'experiments/scripts/maniguard_main_episode.py',
    'experiments/helpers/maniguard_runtime.py',
    'experiments/helpers/maniguard_source.py',
    'experiments/helpers/maniguard_adaptation.py',
    'experiments/helpers/maniguard_client.py',
    'experiments/helpers/maniguard_policy_provenance.py',
    'experiments/helpers/maniguard_phase2_model.py',
    'experiments/helpers/maniguard_phase2_optim.py',
    'experiments/helpers/maniguard_phase2_outcomes.py',
    'experiments/helpers/maniguard_phase2_families.py',
    'experiments/scripts/maniguard_serve.py',
    'experiments/helpers/maniguard_policy.py',
    'experiments/helpers/maniguard_parallel_policy.py',
    'configs/maniguard/sources.json',
)
PHASE2_SOURCES = ('experiments/helpers/maniguard_phase2_runtime.py',
                  'experiments/helpers/maniguard_phase2_source.py',
                  'experiments/helpers/maniguard_phase2_adapter.py',
                  'experiments/helpers/maniguard_phase2_features.py')
AAC_SOURCES = ('experiments/helpers/maniguard_aac_runtime.py',
               'experiments/helpers/maniguard_aac_source.py',
               'experiments/helpers/maniguard_aac_sampling.py',
               'experiments/helpers/maniguard_phase2_aac.py')


class UnsupportedMainExecution(RuntimeError):
    """A planning profile does not establish a complete enforceable native backend."""


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _inside(path, root):
    path = (Path(root) / path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Request path escapes the source-bound study root')
    return path


def _verify_repository_tree(path, revision, source_subtrees, frozen_sources=None):
    """Read-only Git checks, including staged edits and import-shadowing sources."""
    path = Path(path)

    def git(*args):
        return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()

    if git('rev-parse', 'HEAD') != revision:
        raise ValueError('Upstream HEAD differs from its frozen revision: ' + str(path))
    # A submodule's tracked modifications / different commit also dirty its parent.
    # Untracked bytecode caches are not source edits; source additions are checked
    # explicitly below, including files hidden by .gitignore.
    dirty = git('status', '--porcelain=v1', '--untracked-files=no', '--ignore-submodules=untracked')
    if dirty:
        raise ValueError('Upstream tracked tree has local edits: ' + str(path))
    untracked = git('ls-files', '--others', '-z', '--', *source_subtrees)
    unexpected = [name for name in untracked.split('\0') if name and (
        Path(name).suffix in {'.py', '.pyi', '.so', '.pyd'}
        or (Path(name).suffix == '.pyc' and '__pycache__' not in Path(name).parts))]
    frozen_sources = frozen_sources or {}
    for name in unexpected:
        source = (path / name).resolve()
        if frozen_sources.get(str(source)) != _hash(source):
            raise ValueError('Upstream importable source is untracked and unpinned: ' + name)


def verify_upstream_trees(protocol, *, root=ROOT):
    """Pin evaluator/monitor/goal, robot/simulator and serving-library source trees."""
    references = [reference for reference in protocol['input_files']
                  if Path(reference['path']).name == 'sources.json']
    if len(references) != 1:
        raise ValueError('Exactly one frozen upstream-source manifest is required')
    sources = _read(protocol_api._file(references[0], root))
    trees = (('maniguard', 'third_party/ManiGuard', ('maniguard',)),
             ('behavior', 'third_party/ManiGuard/behavior-1k', ('OmniGibson/omnigibson',)),
             ('openpi', 'third_party/openpi', ('src/openpi', 'packages/openpi-client/src/openpi_client')))
    frozen_sources = {str(_inside(reference['path'], root)): reference['sha256']
                      for reference in protocol.get('spec', {}).get('implementation_files', [])}
    for name, relative, subtrees in trees:
        _verify_repository_tree(Path(root) / relative, sources[name]['revision'], subtrees, frozen_sources)


def preflight_backend(protocol, job, runtime_contract, *, root=ROOT, config_dir=None):
    """Pure capability check; parent must call this before claim/service side effects."""
    _verify_sources(protocol, job, root)
    verify_upstream_trees(protocol, root=root)
    contract = runtime_contract
    protocol_api._check_signed(contract)
    required = {'schema', 'supported_profiles', 'episode_timeout_seconds',
                'policy_startup_timeout_seconds', 'termination_grace_seconds',
                'cleanup_reserve_seconds', 'audit_timeout_seconds', 'audit_cleanup_reserve_seconds',
                'per_episode_startup_counts',
                'per_episode_failed_work_counts', 'backend', 'sha256'}
    if (set(contract) != required or contract['schema'] != RUNTIME_SCHEMA
            or not isinstance(contract['supported_profiles'], list)
            or not set(contract['supported_profiles']) <= PROFILES
            or len(set(contract['supported_profiles'])) != len(contract['supported_profiles'])):
        raise ValueError('Unsupported exact main runtime contract')
    for name in ('episode_timeout_seconds', 'policy_startup_timeout_seconds',
                 'termination_grace_seconds', 'cleanup_reserve_seconds',
                 'audit_timeout_seconds', 'audit_cleanup_reserve_seconds'):
        value = contract[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Positive finite main runtime timeouts are required')
    if contract['cleanup_reserve_seconds'] < 3 * contract['termination_grace_seconds']:
        raise ValueError('Cleanup reserve cannot fund the termination sequence')
    if contract['audit_cleanup_reserve_seconds'] < 3 * contract['termination_grace_seconds']:
        raise ValueError('Audit cleanup reserve cannot fund the termination sequence')
    for name in ('per_episode_startup_counts', 'per_episode_failed_work_counts'):
        if counts(contract[name])['policy_control_steps']:
            raise ValueError('Extra reservations cannot create additional policy-action slots')
    if job['resource_contract']['profile'] not in contract['supported_profiles']:
        raise UnsupportedMainExecution('Controller profile is not in the pinned runtime contract')
    try:
        from helpers.maniguard_main_physics import preflight_driver
    except ImportError as error:
        raise UnsupportedMainExecution('No reviewed pre-advance physics driver is installed') from error
    return preflight_driver(contract['backend'], root=root, protocol=protocol)


def resolve_execution(job, *, root=ROOT):
    """Preserve scientific identity; a baseline carrier is explicitly execution-only."""
    profile = job['resource_contract']['profile']
    if profile not in PROFILES:
        raise UnsupportedMainExecution('Unimplemented controller profile')
    model = None
    choices = None
    if profile == 'phase2_learned':
        model_path = protocol_api._file(job['actor'], root)
        genome, _, _ = load_model(model_path)
        if genome_hash(genome) != job['actor_sha256']:
            raise ValueError('Selected actor genome differs from compiled job')
        mode, backend, model = 'learned', 'phase2', {'source': str(model_path), 'role': 'selected_actor',
                                                    'genome_sha256': genome_hash(genome)}
    else:
        controller = protocol_api._record(job['controller'], root)
        if (controller['sha256'] != job['controller_sha256'] or job['actor_sha256'] is not None
                or protocol_api._comparator_resources(controller, root) != job['resource_contract']):
            raise ValueError('Scientific comparator identity differs from its pinned controller')
        if profile.startswith('phase2_'):
            mode = 'contextual' if profile == 'phase2_contextual' else 'fixed'
            choices = controller.get('choices')
            backend = 'phase2'
            model = {'source': None, 'role': 'identity_carrier', 'genome_sha256': genome_hash(identity_genome())}
        elif profile == 'benchmark_original':
            backend = mode = 'benchmark_original'
        else:
            backend, mode = 'aac', 'aac' if profile == 'aac_joint20' else 'fixed8'
    return {'backend': backend, 'mode': mode, 'fixed_choices': choices, 'model': model}


def _source_references(protocol, job, root):
    refs = [protocol['planner_source'], *protocol['input_files'], *protocol['spec']['implementation_files'],
            *protocol['spec']['preprocessing_files']]
    for key in ('actor', 'selection', 'controller'):
        if key in job:
            refs.append(job[key])
    if 'controller' in job:
        controller = protocol_api._record(job['controller'], root)
        if 'aac_config' in controller:
            refs.append(controller['aac_config'])
    return {ref['path']: ref for ref in refs}


def _verify_sources(protocol, job, root):
    refs = _source_references(protocol, job, root)
    required = set(COMMON_SOURCES)
    required.update(PHASE2_SOURCES if job['resource_contract']['profile'].startswith('phase2_') else ())
    required.update(AAC_SOURCES if job['resource_contract']['profile'].startswith('aac_') else ())
    missing = required - set(refs)
    if missing:
        raise ValueError('Runtime dependencies are absent from frozen source pins: ' + ', '.join(sorted(missing)))
    for ref in refs.values():
        protocol_api._file(ref, root)
    return refs


def preflight_request(request, *, root=ROOT, config_dir=None):
    """Validate the exact request without output creation, model load or policy access."""
    root = Path(root).resolve()
    config_dir = root / 'configs/maniguard/phase2' if config_dir is None else Path(config_dir)
    if set(request) != {'schema', 'protocol_path', 'job', 'output', 'parent_binding',
                        'reservation', 'policy', 'runtime_contract', 'command', 'sha256'} or request['schema'] != REQUEST_SCHEMA:
        raise ValueError('Use the exact main child request; independent overrides are forbidden')
    protocol_api._check_signed(request)
    protocol = _read(_inside(request['protocol_path'], root))
    job = deepcopy(request['job'])
    # This is deliberately the first gate before any output, simulator or policy work.
    authorization = protocol_api.authorize_job(protocol, job, config_dir=config_dir, root=root)
    if authorization != {'authorized': True, 'protocol_sha256': protocol['sha256'],
                         'episode_id': job['episode_id'], 'job_sha256': record_hash(job)}:
        raise ValueError('Exact main authorization did not reproduce')
    from helpers.maniguard_main_launcher import verify_parent_binding
    claim = verify_parent_binding(request)
    if claim['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Claim belongs to a different authorized protocol')
    command = request['command']
    if (not isinstance(command, list) or len(command) != 4 or command[2] != '--request'
            or Path(command[1]).resolve() != root / 'experiments/scripts/maniguard_main_episode.py'
            or _read(_inside(command[3], root)) != request):
        raise ValueError('Parent command is not the exact archived main child request')
    contract_path = protocol_api._file(request['runtime_contract'], root)
    if request['runtime_contract'] not in protocol['spec']['implementation_files']:
        raise ValueError('Runtime contract must be an exact frozen implementation reference')
    contract = _read(contract_path)
    driver = preflight_backend(protocol, job, contract, root=root, config_dir=config_dir)
    reservation = request['reservation']
    if (set(reservation) != {'counts', 'startup_counts', 'failed_work_counts', 'deadline'}
            or counts(reservation['counts']) != job['resource_envelope']['successful_work']
            or counts(reservation['startup_counts']) != contract['per_episode_startup_counts']
            or counts(reservation['failed_work_counts']) != contract['per_episode_failed_work_counts']):
        raise ValueError('Child reservations differ from the exact job and runtime contract')
    deadline = reservation['deadline']
    if (type(deadline) not in (int, float) or not math.isfinite(deadline)
            or deadline <= time.time()
            or deadline > claim['claimed_at'] + contract['policy_startup_timeout_seconds']
                + contract['episode_timeout_seconds']):
        raise ValueError('Child deadline exceeds its bounded parent reservation')
    refs = _verify_sources(protocol, job, root)
    route = resolve_execution(job, root=root)
    registry = _read(config_dir / 'families.json')['families']
    family = registry[job['family']]
    eval_config = root / 'third_party/ManiGuard' / family['eval_config']
    if _hash(eval_config) != job['eval_config_sha256'] or family['max_steps'] != job['action_horizon']:
        raise ValueError('Native evaluator configuration differs from the compiled case')
    refs[str(eval_config.relative_to(root))] = {
        'path': str(eval_config.relative_to(root)), 'sha256': job['eval_config_sha256']}
    scene_root = root / 'data/maniguard/benchmark' / family['benchmark_dir']
    scene = (scene_root / job['scene']).resolve()
    if not scene.is_relative_to(scene_root.resolve()) or not (scene / 'scene_ep1.json').is_file():
        raise ValueError('Exact benchmark scene is not locally prepared')
    policy = request['policy']
    if (set(policy) != {'host', 'port', 'handshake_path', 'handshake_sha256'}
            or policy['host'] != '127.0.0.1' or type(policy['port']) is not int
            or not 1 <= policy['port'] <= 65535):
        raise ValueError('Use an explicitly bound local policy endpoint')
    handshake_path = _inside(policy['handshake_path'], root)
    if _hash(handshake_path) != policy['handshake_sha256']:
        raise ValueError('Parent policy handshake receipt changed')
    handshake = _read(handshake_path)
    from helpers.maniguard_policy_provenance import load_policy_provenance
    folder = root / 'data/maniguard/policies' / job['family']
    checkpoint, provenance = load_policy_provenance(folder, job['checkpoint'], family=job['family'])
    expected = {'serve_config': job['checkpoint']['config'], 'checkpoint_revision': job['checkpoint']['revision'],
                'checkpoint': str(checkpoint.resolve()), 'episode_seed_reset': 'study_episode_id',
                'rng_isolation': 'per_episode', 'checkpoint_provenance': provenance}
    if any(handshake.get(key) != value for key, value in expected.items()):
        raise ValueError('Policy handshake does not identify the verified checkpoint and isolated RNG')
    server = ('experiments/scripts/maniguard_serve.py', 'experiments/helpers/maniguard_policy_provenance.py',
              'experiments/helpers/maniguard_policy.py', 'experiments/helpers/maniguard_parallel_policy.py')
    if handshake.get('server_source_hashes') != {name: refs[name]['sha256'] for name in server}:
        raise ValueError('Policy server implementation differs from the frozen pins')
    output = _inside(request['output'], root)
    if output.exists():
        raise ValueError('A claimed episode output cannot be reused')
    return {'root': str(root), 'output': str(output), 'protocol': protocol, 'job': job,
            'job_sha256': record_hash(job), 'protocol_sha256': protocol['sha256'],
            'driver': driver, 'route': route, 'sources': refs, 'handshake': handshake,
            'policy': policy, 'policy_provenance_source': folder / 'download-provenance.json',
            'eval_config': str(eval_config), 'scene_root': str(scene_root),
            'request': deepcopy(request), 'contract': contract,
            'encoder_device': _read(config_dir / 'encoder.json')['device']}


def prepare_episode(request, *, root=ROOT, config_dir=None):
    """After every gate succeeds, create a single immutable execution directory."""
    prepared = preflight_request(request, root=root, config_dir=config_dir)
    root, output = Path(prepared['root']), Path(prepared['output'])
    output.mkdir(parents=True, exist_ok=False)
    job, route = prepared['job'], prepared['route']
    atomic(output / 'main_job.json', signed({'schema': 'maniguard-main-job-v1',
        'protocol_sha256': prepared['protocol_sha256'], 'job_sha256': prepared['job_sha256'],
        'job': job, 'output': str(output)}))
    atomic(output / 'main_request.json', request)
    atomic(output / 'main_protocol.json', prepared['protocol'])
    sources = []
    for original, reference in sorted(prepared['sources'].items()):
        source = protocol_api._file(reference, root)
        destination = output / 'study_source' / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if _hash(destination) != reference['sha256']:
            raise ValueError('Source changed while its archive was being copied')
        sources.append({'original': original, 'path': str(destination.relative_to(output)), 'sha256': _hash(destination)})
    shutil.copyfile(prepared['policy_provenance_source'], output / 'policy-provenance.json')
    prepared['policy_provenance'] = {'path': 'policy-provenance.json', 'sha256': _hash(output / 'policy-provenance.json')}
    model = None
    if route['model'] is not None:
        destination = output / 'actor.npz'
        if route['model']['source'] is None:
            save_model(destination, identity_genome())
        else:
            shutil.copyfile(route['model']['source'], destination)
        if route['model']['role'] == 'selected_actor' and _hash(destination) != job['actor']['sha256']:
            raise ValueError('Selected actor changed while being copied')
        model = {'path': destination.name, 'sha256': _hash(destination),
                 'genome_sha256': route['model']['genome_sha256'], 'role': route['model']['role']}
    resolved = {key: job[key] for key in ('family', 'scene', 'purpose', 'seed', 'sampling_seed',
                                         'episode_id', 'action_horizon', 'stochastic')}
    resolved.update(mode=route['mode'], fixed_choices=route['fixed_choices'], policy=job['checkpoint'])
    carrier = {**resolved, 'phase': 'phase2', 'output': str(output), 'random_policy': False,
               'model': None if model is None else str(output / model['path']),
               'actor_sha256': None if model is None else model['genome_sha256'],
               'adapter': {'kind': 'phase2', 'mode': route['mode']}, 'max_steps_override': None,
               'encoder_device': prepared['encoder_device'], 'scientific_actor_sha256': job['actor_sha256'],
               'scientific_controller_sha256': job['controller_sha256'],
               'code_hashes': {item['original']: item['sha256'] for item in sources}}
    if route['backend'] == 'benchmark_original':
        from dataclasses import asdict

        from helpers.maniguard_adaptation import AdapterConfig
        carrier['phase'] = 'main_original'
        carrier['adapter'] = asdict(AdapterConfig(kind='identity', horizon=8))
    atomic(output / 'job.json', carrier)
    from helpers.maniguard_main_episode_source import prepare_runtime
    evaluator = prepare_runtime(output, route['backend'], root=root)
    execution = signed({'schema': 'maniguard-main-execution-v1', 'output': str(output),
        'protocol_sha256': prepared['protocol_sha256'], 'job_sha256': prepared['job_sha256'],
        'backend': route['backend'], 'resolved': resolved, 'execution_model': model,
        'sources': sources, 'request': {'path': 'main_request.json', 'sha256': _hash(output / 'main_request.json')},
        'generated': {'path': evaluator.name, 'sha256': _hash(evaluator),
                      'provenance_path': 'main-source-provenance.json',
                      'provenance_sha256': _hash(output / 'main-source-provenance.json')},
        'physics_enforcement': prepared['driver'].describe()})
    atomic(output / 'main_execution.json', execution)
    prepared.update(execution=execution, evaluator=str(evaluator), carrier=carrier)
    return prepared


def execute_episode(prepared):
    """Bootstrap one fresh child; the caller must not reuse this process for episodes."""
    from helpers.maniguard_main_episode_meter import OperationMeter
    from helpers.maniguard_main_episode_runtime import MainRuntime, install_runtime
    output = Path(prepared['output'])
    os.environ['SAFETY_DIAL_MANIGUARD_JOB'] = str(output / 'job.json')
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    os.environ.setdefault('OMNIGIBSON_HEADLESS', '1')
    os.environ['OMNIGIBSON_DATA_PATH'] = str(Path(prepared['root']) / 'data/maniguard/datasets')
    sys.path.insert(0, str(Path(prepared['root']) / 'third_party/ManiGuard'))
    meter = OperationMeter(output, prepared['job_sha256'], prepared['request']['reservation'])
    runtime = MainRuntime(prepared, meter, prepared['driver'])
    install_runtime(runtime)
    job = prepared['job']
    sys.argv = [prepared['evaluator'], '--config', prepared['eval_config'],
                '--benchmark-root', prepared['scene_root'], '--scenes', job['scene'],
                '--seed', str(job['seed']), '--host', prepared['policy']['host'],
                '--port', str(prepared['policy']['port']), '--source-profile', 'legacy',
                '--output-dir', str(output), '--run-name', 'episode', '--execute-horizon',
                '8' if prepared['route']['backend'] == 'benchmark_original' else '16']
    try:
        runpy.run_path(prepared['evaluator'], run_name='__main__')
    finally:
        runtime.close()
