"""Main-only guarded wrappers around the established controller implementations.

No simulator, model loader or policy client is imported until explicit bootstrap.
The source hook enters action intent immediately before env.step. Physics enforcement
is supplied by a separately reviewed driver; completed callbacks alone are insufficient.
"""
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from helpers.maniguard_main_episode_meter import atomic, signed

_ACTIVE = None


def get_runtime():
    if _ACTIVE is None:
        raise RuntimeError('Main runtime was not installed by an authorized episode child')
    return _ACTIVE


def install_runtime(runtime):
    global _ACTIVE
    if _ACTIVE is not None or not isinstance(runtime, MainRuntime):
        raise ValueError('Exactly one main runtime is allowed per child')
    _ACTIVE = runtime


class GuardedClient:
    def __init__(self, client, meter):
        self.client, self.meter = client, meter

    def get_server_metadata(self):
        return self.client.get_server_metadata()

    def infer(self, observation):
        return self.meter.call('policy_rpc', lambda: self.client.infer(observation),
                               attempts={'policy_queries': 1})


class GuardedEncoder:
    def __init__(self, encoder, meter):
        self.encoder, self.meter = encoder, meter

    def __getattr__(self, name):
        return getattr(self.encoder, name)

    def begin_episode(self, prompt):
        return self.meter.call('encoder_text', lambda: self.encoder.begin_episode(prompt),
                               attempts={'encoder_text_attempts': 1}, returns={'encoder_text_calls': 1})

    def encode_images(self, overview, wrist):
        return self.meter.call('encoder_images', lambda: self.encoder.encode_images(overview, wrist),
                               attempts={'encoder_image_attempts': 1},
                               returns={'encoder_image_calls': 1, 'encoder_images': 2})


class MainRuntime:
    def __init__(self, prepared, meter, driver, *, client_factory=None, encoder_factory=None):
        self.prepared, self.meter, self.driver = prepared, meter, driver
        self.output, self.job = Path(prepared['output']), prepared['job']
        self.backend = prepared['execution']['backend']
        self.client_factory, self.encoder_factory = client_factory, encoder_factory
        self.action_pending = None
        self._encoder_restore = None
        self.action_index = 0
        self.actions = (self.output / 'main_actions.jsonl').open('x')
        self.evidence = {'schema': 'maniguard-main-runtime-evidence-v1',
                         'protocol_sha256': prepared['protocol_sha256'],
                         'job_sha256': prepared['job_sha256'], 'output': str(self.output),
                         'synthetic': driver.describe().get('synthetic') is True,
                         'handshake': None, 'loaded_modules': {},
                         'policy_provenance': prepared['policy_provenance']}
        self.study = self._build_study()
        self._evidence()

    def _build_study(self):
        # Avoid either module's optional environment-triggered singleton. Instantiate
        # one explicit backend only after imports, with the verified execution job.
        previous = os.environ.pop('SAFETY_DIAL_MANIGUARD_JOB', None)
        try:
            legacy = importlib.import_module('helpers.maniguard_runtime')
            module = (importlib.import_module('helpers.maniguard_phase2_runtime')
                      if self.backend == 'phase2' else None)
        finally:
            if previous is not None:
                os.environ['SAFETY_DIAL_MANIGUARD_JOB'] = previous
        main = self
        if self.backend == 'aac':
            from helpers.maniguard_aac_runtime import AACRuntime
            study = AACRuntime(self.job, self.output, protocol_sha256=self.prepared['protocol_sha256'],
                               root=self.prepared['root'])
        elif self.backend == 'phase2':
            factory = self.encoder_factory or module.FrozenContextEncoder

            def guarded_encoder(*args, **kwargs):
                encoder = main.meter.call('encoder_load', lambda: factory(*args, **kwargs),
                                          attempts={'encoder_load_attempts': 1})
                return GuardedEncoder(encoder, main.meter)

            # This is a fresh, one-episode child. No active source file is changed.
            self._encoder_restore = (module, module.FrozenContextEncoder)
            module.FrozenContextEncoder = guarded_encoder
            study = module.Phase2Study()
            decide = study.adapter.decide
            study.adapter.decide = lambda *args, **kwargs: main.meter.call(
                'adapter_decision', lambda: decide(*args, **kwargs), returns={'adapter_decisions': 1})
        else:
            class OriginalStudy(legacy.Study):
                def start(self, scene, obs, env=None, robot=None, monitor=None):
                    self.env, self.robot = env, robot
                    super().start(scene, obs)
                    self.observed(0, obs, monitor)

                def observed(self, step, obs, monitor):
                    super().observed(step, obs, monitor)
                    from helpers.maniguard_phase2_outcomes import evaluation_state
                    state = evaluation_state(self.env, self.scene, self.job['family'], step)
                    if state:
                        self.write('evaluation_only', **state)
            study = OriginalStudy()
        original_accounting = study.install_accounting

        def install_accounting(og):
            original_accounting(og)
            main.driver.install(og, meter=main.meter, category=main.physics_category)

        study.install_accounting = install_accounting
        validate = study.validate_policy

        def connect(cfg, *_):
            if self.client_factory is None:
                from helpers.maniguard_client import StudyClient
                factory = StudyClient
            else:
                factory = self.client_factory
            client = factory(host=cfg.host, port=cfg.port)
            metadata = client.get_server_metadata()
            if metadata != self.prepared['handshake']:
                raise ValueError('Actual policy handshake differs from the preflight receipt')
            self.evidence['handshake'] = metadata
            self._evidence()
            guarded = GuardedClient(client, self.meter)
            if self.backend == 'aac':
                validate(guarded, 'openpi')
            return guarded, 'openpi'

        study.connect_policy = connect
        finish = study.finish

        def finish_guarded(result):
            if self.action_pending is not None:
                error = RuntimeError('Benchmark ended with unresolved environment action')
                self.meter.fail(self.action_pending['operation_id'], error)
            finish(result)
            self.meter.phase = result.get('status', 'unknown')
            self.meter.snapshot()
            self._evidence()

        study.finish = finish_guarded
        return study

    def _evidence(self):
        root = Path(self.prepared['root']).resolve()
        loaded = {}
        for name, module in list(sys.modules.items()):
            path = getattr(module, '__file__', None)
            if name.startswith('helpers.maniguard_') and path:
                path = Path(path).resolve()
                if path.is_relative_to(root):
                    loaded[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        pinned = {item['original']: item['sha256'] for item in self.prepared['execution'].get('sources', [])}
        if (self.evidence['synthetic'] is False
                and (not pinned or any(pinned.get(name) != digest for name, digest in loaded.items()))):
            raise ValueError('Loaded runtime source path/bytes differ from frozen archives')
        self.evidence['loaded_modules_scope'] = (
            'Module __file__ identity and source bytes checked at bootstrap/handshake/finalization; '
            'does not cryptographically attest in-memory Python code objects.')
        self.evidence['loaded_modules'] = loaded
        self.evidence['physics_enforcement'] = self.driver.describe()
        atomic(self.output / 'main_runtime_evidence.json', signed(self.evidence))

    def validate_config(self, cfg):
        expected_horizon = 8 if self.backend == 'benchmark_original' else 16
        if (cfg.random_policy or not cfg.use_openpi_client or cfg.action_dim != 8
                or cfg.ik_eef_to_joint or not cfg.gripper_binarize
                or cfg.max_steps != self.job['action_horizon'] or cfg.seed != self.job['seed']
                or list(cfg.scenes) != [self.job['scene']] or cfg.execute_horizon != expected_horizon
                or cfg.host != self.prepared['policy']['host'] or cfg.port != self.prepared['policy']['port']):
            raise ValueError('Resolved benchmark configuration differs from the exact frozen job')

    def physics_category(self):
        if self.meter.broken:
            return 'failed_work'
        return 'slot' if self.action_pending is not None else 'startup'

    def _action_event(self, value):
        self.actions.write(json.dumps({'index': self.action_index, **value}, sort_keys=True,
                                      allow_nan=False) + '\n')
        self.actions.flush()
        os.fsync(self.actions.fileno())
        self.action_index += 1

    def before_environment_action(self, step, raw, adapted, executed, low, high):
        if (self.action_pending is not None or step >= self.job['action_horizon']
                or step != self.meter.by_category['slot']['policy_control_steps']):
            raise ValueError('Action is pending or lies outside the full frozen horizon')
        values = [np.asarray(value) for value in (raw, adapted, executed, low, high)]
        if any(value.shape != (8,) or not np.isfinite(value).all() for value in values):
            raise ValueError('Main action boundary requires finite eight-dimensional vectors')
        if not np.array_equal(values[2], np.clip(values[1], values[3], values[4])):
            raise ValueError('Environment input differs from declared clipping')
        token = self.meter.begin('environment_action', returns={'policy_control_steps': 1})
        self.action_pending = {'step': step, 'operation_id': token}
        self._action_event({'type': 'intent', **self.action_pending,
                            **{key: value.tolist() for key, value in zip(
                                ('raw', 'adapted', 'environment_input', 'action_space_low', 'action_space_high'),
                                values, strict=True)}})

    def after_environment_action(self, step):
        if self.action_pending is None or self.action_pending['step'] != step:
            raise ValueError('Environment return does not match its action intent')
        token = self.action_pending['operation_id']
        self._action_event({'type': 'returned', 'step': step, 'operation_id': token})
        self.meter.end(token)
        self.action_pending = None

    def finalize_exit(self):
        # The native evaluator deliberately bypasses atexit/finally via os._exit.
        self.close()

    def close(self):
        if self.actions.closed:
            return
        self.study.close()
        self._evidence()
        self.actions.close()
        self.meter.close()
        if self._encoder_restore is not None:
            module, factory = self._encoder_restore
            module.FrozenContextEncoder = factory
            self._encoder_restore = None
