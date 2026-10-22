"""Source-bound native PhysX pre/post guards for the main child.

Importing or preflighting this module never imports Isaac/OmniGibson. Native callbacks
are attached immediately after the existing app-launch/accounting wrapper returns,
matching the historical instrumented-physics scope. App startup before attachment is
explicitly outside that counter, not asserted to have zero physical work.

Exceptions raised by native event callbacks need not propagate to the caller. On any
reservation/persistence/callback failure the isolated child therefore exits immediately;
returning to PhysX would incorrectly authorize the pending advance.
"""
import hashlib
import importlib
import json
import logging
import math
import os
import re
from copy import deepcopy
from pathlib import Path

from helpers.maniguard_phase2_optim import record_hash

ROOT = Path(__file__).resolve().parents[2]
KIND = 'physx-prepost-v1'
SCOPE = 'post_app_launch_physx_callbacks'
EXIT_CODE = 86
LOG = logging.getLogger(__name__)
ROLES = {'api_stub', 'python_binding', 'extension_manifest', 'physx_plugin', 'simulator',
         'simulation_context', 'physics_context'}
QUALIFICATION_PATHS = {'update_simulation', 'simulation_context_step', 'app_update'}
PHYSX_DOCS = ('https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/'
              'extensions/runtime/source/omni.physx/docs/api/python.html')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _reference(value, root, pinned):
    if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
        raise ValueError('Physics sources require exact path/hash references')
    path = Path(value['path'])
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    path.relative_to(root)
    if (not re.fullmatch('[0-9a-f]{64}', value['sha256']) or
            (str(path), value['sha256']) not in pinned or _sha(path) != value['sha256']):
        raise ValueError('Physics source is absent from the frozen pins or has changed')
    return path


def _pins(protocol, root):
    pins = set()
    for item in protocol.get('spec', {}).get('implementation_files', []) + protocol.get('input_files', []):
        path = Path(item['path'])
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        pins.add((str(path), item['sha256']))
    return pins


def preflight_driver(backend_contract, *, root=ROOT, protocol):
    """Validate a qualified frozen driver before output, imports or policy work.

    A qualification report is empirical evidence with an explicit source graph. It is
    not supplied by this factory and cannot be replaced by a caller's readiness boolean.
    The parent/child must independently validate the complete main protocol as well.
    """
    root = Path(root).resolve()
    if (not isinstance(backend_contract, dict) or
            set(backend_contract) != {'kind', 'physics_driver'} or
            backend_contract['kind'] != 'maniguard-main-guarded-v1'):
        raise ValueError('Unsupported main backend contract')
    declared = backend_contract['physics_driver']
    if (not isinstance(declared, dict) or
            set(declared) != {'kind', 'accounting_scope', 'native_sources', 'qualification'} or
            declared['kind'] != KIND or declared['accounting_scope'] != SCOPE or
            set(declared['native_sources']) != ROLES):
        raise ValueError('Unsupported physics driver, accounting scope or source roles')
    pins = _pins(protocol, root)
    own_path = Path(__file__).resolve()
    if (str(own_path), _sha(own_path)) not in pins:
        raise ValueError('The running physics driver must be frozen with the protocol')
    paths = {name: _reference(reference, root, pins)
             for name, reference in declared['native_sources'].items()}
    stub = paths['api_stub'].read_text()
    if ('def subscribe_physics_on_step_events(' not in stub or
            'pre_step: bool' not in stub or 'order: int' not in stub):
        raise ValueError('Pinned PhysX API does not provide ordered pre/post step callbacks')
    if not re.search(r'^version\s*=\s*"106\.5\.7"\s*$',
                     paths['extension_manifest'].read_text(), re.MULTILINE):
        raise ValueError('This driver requires the reviewed installed PhysX 106.5.7')
    qualification_path = _reference(declared['qualification'], root, pins)
    evidence = json.loads(qualification_path.read_text())
    body = {key: value for key, value in evidence.items() if key != 'sha256'}
    if evidence.get('sha256') != record_hash(body):
        raise ValueError('Physics qualification content hash differs')
    if (evidence.get('schema') != 'maniguard-physx-qualification-v1' or
            evidence.get('driver_source_sha256') != _sha(own_path) or
            evidence.get('native_sources_sha256') != record_hash(declared['native_sources']) or
            evidence.get('accounting_scope') != SCOPE or evidence.get('passed') is not True or
            evidence.get('task_episodes') != 0):
        raise ValueError('Native physics qualification does not bind this driver and source set')
    checks = evidence.get('checks')
    if not isinstance(checks, dict) or set(checks) != QUALIFICATION_PATHS:
        raise ValueError('All declared native advancement paths need qualification')
    sources = evidence.get('sources')
    if not isinstance(sources, list) or not sources:
        raise ValueError('Native qualification needs independently inspectable source receipts')
    # Receipt files are frozen through the qualification graph, not silently fetched.
    for source in sources:
        source_pins = _pins({'spec': {'implementation_files': [source]}}, root)
        _reference(source, root, source_pins)
    for name, check in checks.items():
        if (not isinstance(check, dict) or check.get('pre_post_order_verified') is not True or
                check.get('hard_exit_before_overbudget_advance') is not True or
                type(check.get('completed_ticks')) is not int or check['completed_ticks'] < 1 or
                check.get('overbudget_process_exit_code') != EXIT_CODE):
            raise ValueError('Native physics guard qualification incomplete: ' + name)
    return NativePhysXDriver(declared['native_sources'], root=root,
                             qualification=declared['qualification'])


class NativePhysXDriver:
    """Native implementation; direct construction is for bounded qualification only.

    Main execution must obtain this object from preflight_driver. No physics is changed:
    pre-step reserves one tick, post-step settles it, and abnormal paths terminate the
    child. Retain both subscription handles through process termination.
    """

    def __init__(self, native_sources, *, root=ROOT, qualification=None):
        self.native_sources = deepcopy(native_sources)
        self.root = Path(root).resolve()
        self.qualification = deepcopy(qualification)
        self.meter = self.category = None
        self.pending = None
        self.subscriptions = []
        self.installed = self.attached = self.failed = False
        self.pre_events = self.post_events = 0
        self._terminate = os._exit

    def describe(self):
        return {'kind': KIND, 'accounting_scope': SCOPE,
                'unobserved_startup': 'App launch before callback registration is outside this counter.',
                'native_sources': deepcopy(self.native_sources),
                'qualification': deepcopy(self.qualification),
                'qualified': self.qualification is not None,
                'installed': self.installed, 'attached': self.attached,
                'pre_events': self.pre_events, 'post_events': self.post_events,
                'pending_tick': self.pending is not None, 'failed': self.failed,
                'abort_exit_code': EXIT_CODE, 'callback_order': 0,
                'native_api_documentation': PHYSX_DOCS}

    def _abort(self, error):
        self.failed = True
        try:
            if self.pending is not None:
                try:
                    self.meter.fail(self.pending, error)
                except BaseException:
                    LOG.exception('Could not settle failed physics work; durable intent remains unresolved')
            # Process status remains authoritative if this diagnostic also fails.
            try:
                path = Path(self.meter.output) / 'main_physics_failure.json'
                descriptor = self.describe() | {'error_type': type(error).__name__}
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, 'w') as stream:
                    json.dump(descriptor, stream, sort_keys=True, allow_nan=False)
                    stream.write('\n')
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                LOG.exception('Could not persist physics failure diagnostic; terminating child')
        finally:
            # Even logging/diagnostic failure cannot resume the native advance.
            self._terminate(EXIT_CODE)
            raise SystemExit(EXIT_CODE)

    @staticmethod
    def _valid_dt(dt):
        return type(dt) in (int, float) and math.isfinite(dt) and dt > 0

    def _before(self, dt):
        try:
            if self.failed or self.pending is not None or not self._valid_dt(dt):
                raise RuntimeError('Unexpected nested/repeated physics pre-step callback')
            self.pending = self.meter.begin('physics_tick', category=self.category(),
                                            returns={'physics_steps': 1})
            self.pre_events += 1
        except BaseException as error:
            self._abort(error)
            raise  # Defensive if a substituted terminator unexpectedly returns.

    def _after(self, dt):
        try:
            if self.failed or self.pending is None or not self._valid_dt(dt):
                raise RuntimeError('Physics post-step has no matching reserved tick')
            self.meter.end(self.pending)
            self.pending = None
            self.post_events += 1
        except BaseException as error:
            self._abort(error)
            raise  # Defensive if a substituted terminator unexpectedly returns.

    def _attach(self, interface, binding_module):
        reference = self.native_sources['python_binding']
        expected = (self.root / reference['path']).resolve()
        observed = Path(binding_module.__file__).resolve()
        if observed != expected or _sha(observed) != reference['sha256']:
            raise ValueError('Loaded PhysX binding differs from the qualified native source')
        if self.attached or self.subscriptions:
            raise RuntimeError('Physics guard cannot be attached twice')
        before = interface.subscribe_physics_on_step_events(self._before, True, 0)
        if before is None:
            raise RuntimeError('PhysX did not retain the pre-step subscription')
        self.subscriptions.append(before)
        after = interface.subscribe_physics_on_step_events(self._after, False, 0)
        if after is None:
            raise RuntimeError('PhysX did not retain the post-step subscription')
        self.subscriptions.append(after)
        self.attached = True

    def install(self, og, *, meter, category):
        if self.installed or not callable(category):
            raise ValueError('Install one physics driver with an explicit cost category provider')
        if getattr(og, 'sim', None) is not None:
            raise RuntimeError('Cannot install a main physics guard after simulator creation')
        simulator = importlib.import_module('omnigibson.simulator')
        original_launch = simulator._launch_app
        self.meter, self.category = meter, category

        def launch(*args, **kwargs):
            if self.attached:
                raise RuntimeError('A main episode may launch only one simulator app')
            app = original_launch(*args, **kwargs)
            physx = importlib.import_module('omni.physx')
            binding = importlib.import_module('omni.physx.bindings._physx')
            try:
                self._attach(physx.get_physx_interface(), binding)
            except BaseException as error:
                # A partially attached native callback must not survive into cleanup.
                self._abort(error)
                raise
            return app

        simulator._launch_app = launch
        self.installed = True
