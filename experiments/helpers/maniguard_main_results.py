"""Independent main-episode identity, action, outcome and accounting audit.

No simulator, policy service, image encoder or reserved-output discovery is used.
Only explicitly supplied episode evidence is opened. An invalid episode retains
known costs and its requested ID, but never acquires verified plan fields or a
fabricated failure reward. Source hashes attest recorded bytes, not a cryptographic
proof of process memory; policy checkpoint verification is the pinned server's
startup attestation bound to its archived download receipt.
"""

import hashlib
import json
import math
import zlib
from copy import deepcopy
from pathlib import Path

import numpy as np

from helpers import maniguard_main_protocol as protocol_api
from helpers.maniguard_adaptation import Frame
from helpers.maniguard_phase2_adapter import Phase2Adapter
from helpers.maniguard_phase2_model import genome_hash, identity_genome, load_model
from helpers.maniguard_phase2_optim import record_hash
from helpers.maniguard_phase2_outcomes import derive_outcome
from helpers.maniguard_phase2_results import _array, _audit_causality, _audit_decisions, _steps
from helpers.maniguard_policy_provenance import validate_provenance_record

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/maniguard/phase2'
SCHEMA = 'maniguard-main-results-v1'
WORK_COUNTS = protocol_api.WORK_COUNTS
CATEGORIES = ('slot', 'startup', 'failed_work')
AUDIT_ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, ImportError)
PHASE2_PROFILES = {'phase2_learned', 'phase2_identity', 'phase2_fixed', 'phase2_contextual'}
SUPPORTED_PROFILES = PHASE2_PROFILES | {'benchmark_original', 'aac_joint20', 'aac_fixed8_joint20'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def _signed(path, schema=None):
    value = _read(path)
    if not isinstance(value, dict) or value.get('sha256') != record_hash(
            {key: item for key, item in value.items() if key != 'sha256'}):
        raise ValueError('Invalid content-hashed record: ' + str(path))
    if schema is not None and value.get('schema') != schema:
        raise ValueError('Wrong evidence schema: ' + str(path))
    return value


def _inside(output, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('Evidence paths must be nonempty output-relative paths')
    if '..' in Path(relative).parts:
        raise ValueError('Evidence path contains parent traversal')
    path = (output / relative).resolve()
    if not path.is_relative_to(output):
        raise ValueError('Evidence path escapes episode output')
    return path


def _reference(output, reference):
    path = _inside(output, reference['path'])
    if _sha(path) != reference['sha256']:
        raise ValueError('Evidence file hash mismatch: ' + reference['path'])
    return path


def _events(path):
    rows = [_read_line(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError('Trace contains a non-object event')
    return rows


def _read_line(line):
    return json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError('Nonfinite trace value: ' + value)))


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _counter_map(value, *, complete=False):
    if (not isinstance(value, dict) or set(value) - set(WORK_COUNTS)
            or (complete and set(value) != set(WORK_COUNTS))
            or any(type(item) is not int or item < 0 for item in value.values())):
        raise ValueError('Invalid explicit work-counter map')
    return value


class JournalAuditError(ValueError):
    """Retain measured prefix counts when a later durable record is malformed."""

    def __init__(self, message, counts, categories):
        super().__init__(message)
        self.known_counts = deepcopy(counts)
        self.by_category = deepcopy(categories)


def _journal_lower_bounds(path):
    events = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = _read_line(line)
            if not isinstance(event, dict):
                break
            events.append(event)
        except AUDIT_ERRORS:
            break
    try:
        replay = audit_operation_journal(events)
        return replay['counts'], replay['by_category']
    except JournalAuditError as error:
        return error.known_counts, error.by_category


def audit_operation_journal(events):
    """Recompute paid attempts/returns; unsuccessful returns stay unresolved."""
    counts = dict.fromkeys(WORK_COUNTS, 0)
    categories = {name: dict.fromkeys(WORK_COUNTS, 0) for name in CATEGORIES}
    pending, seen, failures, all_intents = {}, set(), [], {}
    for index, event in enumerate(events):
        try:
            if event.get('index') != index or type(event.get('index')) is not int:
                raise ValueError('Operation journal has missing/duplicate/reordered indices')
            identifier, kind = event.get('id'), event.get('type')
            if not isinstance(identifier, str) or not identifier:
                raise ValueError('Operation journal has an invalid operation ID')
            if kind == 'intent':
                if identifier in seen or event.get('category') not in CATEGORIES:
                    raise ValueError('Duplicate operation intent or invalid category')
                seen.add(identifier)
                attempts = _counter_map(event.get('attempt_counts'))
                _counter_map(event.get('return_counts'))
                pending[identifier] = deepcopy(event)
                all_intents[identifier] = deepcopy(event)
                for name, count in attempts.items():
                    counts[name] += count
                    categories[event['category']][name] += count
            elif kind in ('returned', 'failed'):
                if identifier not in pending or pending[identifier].get('failed'):
                    raise ValueError('Operation terminal lacks one unmatched intent')
                intent = pending[identifier]
                returned = _counter_map(event.get('return_counts'))
                if kind == 'returned':
                    if returned != intent['return_counts']:
                        raise ValueError('Returned work differs from its reserved operation')
                    pending.pop(identifier)
                else:
                    if returned:
                        raise ValueError('Failed operation cannot assert unverified returned work')
                    intent['failed'] = True
                    failures.append(identifier)
                for name, count in returned.items():
                    counts[name] += count
                    categories[intent['category']][name] += count
            else:
                raise ValueError('Unsupported operation journal event')
        except AUDIT_ERRORS as error:
            raise JournalAuditError(str(error), counts, categories) from error
    reserved = dict.fromkeys(WORK_COUNTS, 0)
    for event in pending.values():
        for name, count in event['return_counts'].items():
            reserved[name] += count
    return {'counts': counts, 'by_category': categories, 'reserved_inflight': reserved,
            'pending_ids': sorted(pending), 'failed_ids': failures, 'operations': all_intents,
            'unknown_fields': sorted({name for event in pending.values()
                for name in (*event['attempt_counts'], *event['return_counts'])})}


def _costs(output, job, row):
    """Keep ledger lower bounds even when independent replay fails."""
    ledger = _read(output / 'main_costs.json')
    if not isinstance(ledger, dict):
        raise TypeError('Accounting ledger must be an object')
    for name, value in ledger.get('counts', {}).items():
        if name in WORK_COUNTS and type(value) is int and value >= 0:
            row['costs'][name] = value
    if _number(ledger.get('elapsed_seconds')):
        row['costs']['elapsed_seconds'] = ledger['elapsed_seconds']
    categories = ledger.get('by_category', {})
    for category in CATEGORIES:
        source = categories.get(category, {}) if isinstance(categories, dict) else {}
        row['cost_breakdown'][category] = {name: value for name, value in source.items()
            if name in WORK_COUNTS and type(value) is int and value >= 0}
    _signed(output / 'main_costs.json', 'maniguard-main-costs-v1')
    if ledger.get('job_sha256') != record_hash(job):
        raise ValueError('Accounting belongs to another compiled job')
    journal_path = output / 'main_operations.jsonl'
    if _sha(journal_path) != ledger.get('journal_sha256'):
        raise ValueError('Operation journal differs from its terminal ledger hash')
    journal = audit_operation_journal(_events(journal_path))
    for name, measured in journal['counts'].items():
        if measured > row['costs'].get(name, -1):
            row['costs'][name] = measured
            row['cost_issues'].append('Ledger below journal lower bound: ' + name)
    if ledger.get('counts') != journal['counts'] or categories != journal['by_category']:
        raise ValueError('Ledger totals/category allocation differ from operation replay')
    if ledger.get('reserved_inflight') != journal['reserved_inflight']:
        raise ValueError('In-flight reservation differs from operation replay')
    if (journal['pending_ids'] or ledger.get('pending') != [] or ledger.get('issues') != []
            or ledger.get('phase') != 'completed'):
        row['unknown_cost_fields'] = journal['unknown_fields']
        raise ValueError('Unresolved, failed or incomplete attempted-work accounting')
    _counter_map(ledger['counts'], complete=True)
    return ledger, journal


def _bindings(record, protocol, job, output):
    if (record.get('protocol_sha256') != protocol['sha256']
            or record.get('job_sha256') != record_hash(job)
            or record.get('output') != str(output)):
        raise ValueError('Recorded evidence differs from exact protocol/job/output binding')


def _source_refs(protocol, job):
    refs = list(protocol['input_files']) + list(protocol['spec']['preprocessing_files'])
    refs += list(protocol['spec']['implementation_files']) + [protocol['planner_source']]
    refs += [job[name] for name in ('actor', 'selection', 'controller') if name in job]
    expected = {}
    for reference in refs:
        if reference['path'] in expected and expected[reference['path']] != reference['sha256']:
            raise ValueError('Frozen references disagree on one source path')
        expected[reference['path']] = reference['sha256']
    return expected


def _reconstruct(root, backend):
    from helpers.maniguard_main_episode_source import reconstruct_runtime
    return reconstruct_runtime(root, backend)


def _archives(output, execution, protocol, job, runtime, root):
    expected = _source_refs(protocol, job)
    saved = {}
    for reference in execution['sources']:
        original = reference['original']
        if original in saved:
            raise ValueError('Duplicate archived original path')
        saved[original] = _reference(output, reference)
    if job['resource_contract']['profile'].startswith('aac_'):
        controller = _signed(saved[job['controller']['path']])
        config_reference = controller['aac_config']
        expected[config_reference['path']] = config_reference['sha256']
    if any(name not in saved or _sha(saved[name]) != digest for name, digest in expected.items()):
        raise ValueError('Frozen source/model/config archive is missing or changed')
    reader_name = 'experiments/helpers/maniguard_main_results.py'
    if expected.get(reader_name) != _sha(Path(__file__)):
        raise ValueError('Running main audit reader is not the frozen reader implementation')
    loaded = runtime.get('loaded_modules')
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError('Loaded runtime implementation bindings are missing')
    required_loaded = {'maniguard_main_episode_runtime.py', 'maniguard_main_episode_meter.py',
                       'maniguard_main_physics.py', 'maniguard_runtime.py', 'maniguard_client.py'}
    required_loaded.update({'maniguard_phase2_runtime.py', 'maniguard_phase2_adapter.py',
                            'maniguard_phase2_model.py', 'maniguard_phase2_features.py'}
                           if execution['backend'] == 'phase2' else ())
    required_loaded.update({'maniguard_aac_runtime.py', 'maniguard_aac_sampling.py',
                            'maniguard_phase2_aac.py'} if execution['backend'] == 'aac' else ())
    if not {'experiments/helpers/' + name for name in required_loaded} <= set(loaded):
        raise ValueError('Required runtime implementation was not recorded as loaded')
    for original, digest in loaded.items():
        if original not in expected or expected[original] != digest or _sha(saved[original]) != digest:
            raise ValueError('Loaded implementation differs from frozen archived source: ' + original)
    generated = execution['generated']
    evaluator = _reference(output, generated)
    provenance = _read(_reference(output, {'path': generated['provenance_path'],
                                          'sha256': generated['provenance_sha256']}))
    rebuilt = _reconstruct(root, execution['backend'])
    if isinstance(rebuilt, tuple):
        rebuilt = rebuilt[0]
    if isinstance(rebuilt, Path):
        rebuilt = rebuilt.read_text()
    if isinstance(rebuilt, str):
        rebuilt = rebuilt.encode()
    if evaluator.read_bytes() != rebuilt or provenance.get('runtime_sha256') != _sha(evaluator):
        raise ValueError('Generated evaluator does not reconstruct from pinned source')
    sources = _read(saved['configs/maniguard/phase2/sources.json'])
    if provenance.get('revision') != sources['maniguard']['revision']:
        raise ValueError('Generated evaluator uses another upstream revision')
    if runtime.get('synthetic') is not False:
        raise ValueError('Synthetic or unclassified runtime evidence cannot become a physical result')
    return saved


def _physics(execution, runtime, request, protocol, archived, root, counts):
    from helpers.maniguard_main_physics import preflight_driver
    reference = request['runtime_contract']
    if reference not in protocol['spec']['implementation_files']:
        raise ValueError('Runtime contract is not frozen with the main protocol')
    path = archived[reference['path']]
    if _sha(path) != reference['sha256']:
        raise ValueError('Runtime contract archive differs from parent request')
    contract = _signed(path, 'maniguard-main-runtime-contract-v1')
    qualified = preflight_driver(contract['backend'], root=root, protocol=protocol).describe()
    if execution.get('physics_enforcement') != qualified:
        raise ValueError('Execution driver differs from independently qualified physics driver')
    final = runtime.get('physics_enforcement', {})
    dynamic = {'installed', 'attached', 'pre_events', 'post_events', 'pending_tick', 'failed'}
    if any(final.get(key) != value for key, value in qualified.items() if key not in dynamic):
        raise ValueError('Final physics driver identity/scope changed during execution')
    if (final.get('installed') is not True or final.get('attached') is not True
            or final.get('failed') is not False or final.get('pending_tick') is not False
            or type(final.get('pre_events')) is not int or type(final.get('post_events')) is not int
            or final['pre_events'] != final['post_events']
            or final['post_events'] != counts.get('physics_steps')):
        raise ValueError('Final native pre/post physics accounting is incomplete or contradictory')


def _controller(output, execution, job, archived):
    profile = job['resource_contract']['profile']
    expected_mode = {'phase2_learned': 'learned', 'phase2_identity': 'fixed',
                     'phase2_fixed': 'fixed', 'phase2_contextual': 'contextual',
                     'aac_joint20': 'aac', 'aac_fixed8_joint20': 'fixed8',
                     'benchmark_original': 'benchmark_original'}[profile]
    choices = None
    if job['run_kind'] == 'fixed':
        controller = _signed(archived[job['controller']['path']])
        if controller['sha256'] != job['controller_sha256']:
            raise ValueError('Archived scientific controller differs from compiled job')
        if profile in ('phase2_identity', 'phase2_fixed'):
            choices = controller['choices']
    expected = {key: job[key] for key in ('family', 'scene', 'purpose', 'seed', 'sampling_seed',
                                         'episode_id', 'action_horizon', 'stochastic')}
    expected.update(mode=expected_mode, fixed_choices=choices, policy=job['checkpoint'])
    if execution.get('resolved') != expected:
        raise ValueError('Actual resolved execution differs from frozen scientific job')
    backend = 'phase2' if profile in PHASE2_PROFILES else (
        'aac' if profile.startswith('aac_') else 'benchmark_original')
    if execution['backend'] != backend:
        raise ValueError('Execution backend differs from controller resource contract')
    model = execution.get('execution_model')
    if backend != 'phase2':
        if model is not None:
            raise ValueError('Non-actor comparator cannot use an execution model')
        if backend == 'benchmark_original':
            from dataclasses import asdict

            from helpers.maniguard_adaptation import AdapterConfig
            carrier = _read(output / 'job.json')
            if (any(carrier.get(key) != value for key, value in expected.items())
                    or carrier.get('phase') != 'main_original' or carrier.get('actor_sha256') is not None
                    or carrier.get('model') is not None or carrier.get('random_policy') is not False
                    or carrier.get('max_steps_override') is not None
                    or carrier.get('adapter') != asdict(AdapterConfig(kind='identity', horizon=8))):
                raise ValueError('Native benchmark carrier differs from original eight-action execution')
            return carrier
        return None
    actor_path = _reference(output, model)
    if actor_path != output / 'actor.npz':
        raise ValueError('Phase-two replay requires its explicitly archived actor.npz')
    genome, _, _ = load_model(actor_path)
    digest = genome_hash(genome)
    if model.get('genome_sha256') != digest:
        raise ValueError('Execution model genome digest mismatch')
    if profile == 'phase2_learned':
        if (model.get('role') != 'selected_actor' or digest != job['actor_sha256']
                or _sha(actor_path) != job['actor']['sha256']):
            raise ValueError('Selected actor bytes/genome differ from actual execution model')
    elif (model.get('role') != 'identity_carrier' or job['actor_sha256'] is not None
          or digest != genome_hash(identity_genome())):
        raise ValueError('Reference carrier must be explicit canonical identity, not scientific actor')
    carrier = _read(output / 'job.json')
    for key, value in expected.items():
        if carrier.get(key) != value:
            raise ValueError('Runtime carrier job differs from resolved execution: ' + key)
    if (carrier.get('actor_sha256') != digest or carrier.get('phase') != 'phase2'
            or carrier.get('max_steps_override') is not None or carrier.get('random_policy') is not False
            or Path(carrier.get('model', '')).resolve() != actor_path):
        raise ValueError('Runtime carrier bypasses exact model/full-episode declaration')
    return carrier


def _handshake(output, runtime, job, archived, trace, root):
    metadata = runtime.get('handshake', {})
    pin = job['checkpoint']
    expected = {'serve_config': pin['config'], 'checkpoint_revision': pin['revision'],
                'episode_seed_reset': 'study_episode_id', 'rng_isolation': 'per_episode',
                'checkpoint': str((root / 'data/maniguard/policies' / job['family'] / pin['step']).resolve())}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Actual policy handshake differs from exact checkpoint/RNG contract')
    records = [event.get('metadata') for event in trace if event.get('type') == 'policy_metadata']
    if records != [metadata]:
        raise ValueError('Policy handshake is not the one recorded by the executing runtime')
    source_hashes = metadata.get('server_source_hashes')
    server_sources = {'experiments/scripts/maniguard_serve.py',
                      'experiments/helpers/maniguard_policy_provenance.py',
                      'experiments/helpers/maniguard_policy.py',
                      'experiments/helpers/maniguard_parallel_policy.py'}
    if not isinstance(source_hashes, dict) or set(source_hashes) != server_sources:
        raise ValueError('Policy server implementation identity is missing')
    for name, digest in source_hashes.items():
        if name not in archived or _sha(archived[name]) != digest:
            raise ValueError('Policy server source differs from frozen archive')
    provenance_path = _reference(output, runtime['policy_provenance'])
    inventory = validate_provenance_record(_read(provenance_path), pin, family=job['family'])
    attestation = metadata.get('checkpoint_provenance', {})
    expected = {'receipt_sha256': _sha(provenance_path), 'verified_files': len(inventory),
                'verified_bytes': sum(item['bytes'] for item in inventory.values()),
                'verification': 'all_declared_files_sha256_and_available_upstream_digests'}
    if any(attestation.get(key) != value for key, value in expected.items()):
        raise ValueError('Policy startup checkpoint verification differs from archived provenance')


def _outcome(output, job, trace, counts):
    results = _events(output / 'episode/results.jsonl')
    if len(results) != 1:
        raise ValueError('Expected one authoritative native benchmark result')
    result = results[0]
    steps = result.get('steps')
    native_seed = zlib.crc32(f"{job['seed']}:{job['scene']}".encode()) & 0x7FFFFFFF
    if (result.get('status') != 'completed' or result.get('scene_name') != job['scene']
            or type(steps) is not int or not 1 <= steps <= job['action_horizon']
            or type(result.get('success')) is not bool
            or (not result['success'] and steps != job['action_horizon'])
            or result.get('episode_seed') != native_seed):
        raise ValueError('Native result scene/seed/horizon/termination differs from the exact job')
    if [event.get('result') for event in trace if event.get('type') == 'result'] != [result]:
        raise ValueError('Trace terminal differs from native benchmark result')
    monitor = _read(_inside(output, 'episode/' + job['scene'] + '_ltl.json'))
    if (monitor.get('error') is not None or type(monitor.get('violated')) is not bool
            or result.get('ltl_monitored') is not True
            or monitor['violated'] != result.get('ltl_violated')
            or monitor.get('total_steps_monitored') != steps + 1):
        raise ValueError('Raw physical monitor is incomplete or contradictory')
    _steps(monitor['log'], range(steps + 1), 'native monitor')
    observed = [event for event in trace if event.get('type') == 'observed']
    _steps(observed, range(steps + 1), 'observed')
    for event, native in zip(observed, monitor['log'], strict=True):
        copy = event.get('monitor') or {}
        if copy.get('valid') is not True or copy.get('error') is not None or copy.get('observation') != native:
            raise ValueError('Observed monitor telemetry is stale or differs from raw sidecar')
    if counts.get('policy_control_steps') != steps:
        raise ValueError('Native action count differs from audited attempted-work accounting')
    goals = [event for event in trace if event.get('type') == 'goal']
    extra = [event for event in trace if event.get('type') == 'evaluation_only' and 'step' in event]
    outcome = derive_outcome(job['family'], result, monitor['log'], goals, extra or None)
    if not outcome['valid']:
        raise ValueError('Supplemental completion invalid: ' + '; '.join(outcome['invalid_reasons']))
    return result, outcome


def _action_journal(output, trace, steps, operations):
    events = _events(output / 'main_actions.jsonl')
    if [event.get('index') for event in events] != list(range(len(events))) or len(events) != 2 * steps:
        raise ValueError('Main action journal is missing, duplicated or incomplete')
    intents = []
    executed = [event for event in trace if event.get('type') == 'executed']
    _steps(executed, range(steps), 'executed')
    seen = set()
    for step in range(steps):
        before, after = events[2 * step:2 * step + 2]
        operation = before.get('operation_id')
        if (before.get('type') != 'intent' or after.get('type') != 'returned'
                or before.get('step') != step or after.get('step') != step
                or operation not in operations or operation in seen
                or after.get('operation_id') != operation):
            raise ValueError('Action intent/return does not bind a unique metered operation')
        seen.add(operation)
        intent = operations[operation]
        if (intent['category'] != 'slot'
                or intent['return_counts'].get('policy_control_steps') != 1
                or intent['attempt_counts'].get('policy_control_steps', 0) != 0):
            raise ValueError('Action is not bound to one returned slot control-step operation')
        for name in ('raw', 'adapted', 'environment_input', 'action_space_low', 'action_space_high'):
            _array(before.get(name), (8,), 'action ' + name)
        if np.any(np.asarray(before['action_space_low']) > before['action_space_high']):
            raise ValueError('Environment action limits are reversed')
        expected = np.clip(before['adapted'], before['action_space_low'], before['action_space_high'])
        if (not np.array_equal(expected, before['environment_input'])
                or not np.array_equal(executed[step]['action'], expected)):
            raise ValueError('Executed action differs from recorded adaptation/environment clipping')
        intents.append(before)
    return intents


def _phase2_replay(output, carrier, trace, result, encoder, intents):
    _audit_causality(carrier, result, trace)
    decisions, _ = _audit_decisions(carrier, trace, output, result['steps'], encoder)
    genome, _, _ = load_model(output / 'actor.npz')
    adapter = Phase2Adapter(genome, seed=carrier['sampling_seed'], mode=carrier['mode'],
                            fixed_choices=carrier.get('fixed_choices'))
    chunk, cursor, last_query, proposal_pending = None, 0, None, None
    awaiting_chunk = False
    queries = 0
    for event in trace:
        kind, step = event.get('type'), event.get('step')
        if kind == 'decision':
            adapter.decide(np.asarray(event['features']), step)
        elif kind == 'query':
            if awaiting_chunk:
                raise ValueError('Policy query precedes the previous response')
            if chunk is not None and cursor < len(chunk) and adapter.replan_reason() is None:
                raise ValueError('Policy requery precedes selected execution horizon')
            adapter.on_query()
            last_query, queries = step, queries + 1
            awaiting_chunk = True
        elif kind == 'chunk':
            if step != last_query or not awaiting_chunk:
                raise ValueError('Policy chunk lacks its query')
            awaiting_chunk = False
            chunk = np.asarray(event['actions'], dtype=np.float32)[:16]
            if chunk.ndim != 2 or chunk.shape[1] != 8 or not len(chunk) or not np.isfinite(chunk).all():
                raise ValueError('Invalid finite policy chunk')
            cursor = 0
        elif kind == 'proposal':
            if chunk is None or cursor >= len(chunk) or event.get('chunk_index') != cursor:
                raise ValueError('Action proposal is outside its unique policy chunk')
            native = chunk[cursor].copy()
            native[-1] = np.sign(native[-1]) if abs(native[-1]) > .01 else -1.0
            if not np.array_equal(event['original'], native):
                raise ValueError('Proposed action differs from native chunk/binarization')
            frame = Frame(**{name: np.asarray(value) if isinstance(value, list) else value
                             for name, value in event['frame'].items()})
            adapted, _ = adapter.act(native, frame)
            if (not np.array_equal(event['adapted'], adapted)
                    or not np.array_equal(intents[step]['adapted'], adapted)
                    or not np.array_equal(intents[step]['raw'], chunk[cursor])):
                raise ValueError('Actual adaptation differs from independently replayed controller')
            proposal_pending = step
        elif kind == 'executed':
            if proposal_pending != step:
                raise ValueError('Action execution lacks a unique replayed proposal')
            adapter.applied(np.asarray(event['action']))
            cursor += 1
            proposal_pending = None
    if awaiting_chunk or proposal_pending is not None:
        raise ValueError('Terminal episode has an unresolved query/action')
    return {'policy_queries': queries, 'adapter_decisions': len(decisions),
            'encoder_image_calls': len(decisions), 'encoder_images': 2 * len(decisions),
            'encoder_image_attempts': len(decisions), 'encoder_text_calls': 1,
            'encoder_text_attempts': 1, 'encoder_load_attempts': 1, 'actor_rpc_attempts': 0}


def _original_replay(carrier, trace, result, intents):
    _audit_causality(carrier, result, trace)
    if any(event.get('type') in ('decision', 'phase2_features', 'phase2_actor') for event in trace):
        raise ValueError('Native benchmark unexpectedly used a supplementary controller/encoder')
    chunk, cursor, query_step, awaiting, queries = None, 0, None, False, 0
    for event in trace:
        kind, step = event.get('type'), event.get('step')
        if kind == 'query':
            if awaiting or (chunk is not None and cursor != min(8, len(chunk))):
                raise ValueError('Native query does not follow complete eight-action/short prefix')
            query_step, awaiting, queries = step, True, queries + 1
        elif kind == 'chunk':
            if not awaiting or step != query_step:
                raise ValueError('Native policy chunk lacks its unique query')
            chunk = np.asarray(event['actions'], dtype=np.float32)
            if chunk.ndim != 2 or chunk.shape[1] != 8 or not len(chunk) or not np.isfinite(chunk[:8]).all():
                raise ValueError('Invalid native policy prefix')
            cursor, awaiting = 0, False
        elif kind == 'proposal':
            if chunk is None or cursor >= min(8, len(chunk)) or event.get('chunk_index') != cursor:
                raise ValueError('Native proposal lies outside its eight-action prefix')
            frame = event.get('frame', {})
            if frame.get('prerequisite') is not None or frame.get('target_engaged') is not None:
                raise ValueError('Native comparator unexpectedly accessed task predicates')
            raw = chunk[cursor]
            converted = raw.copy()
            converted[-1] = np.sign(converted[-1]) if abs(converted[-1]) > .01 else -1.0
            if (not np.array_equal(event['original'], converted)
                    or not np.array_equal(event['adapted'], converted)
                    or not np.array_equal(intents[step]['raw'], raw)
                    or not np.array_equal(intents[step]['adapted'], converted)):
                raise ValueError('Native benchmark action was changed beyond gripper conversion/clipping')
        elif kind == 'executed':
            cursor += 1
    if awaiting:
        raise ValueError('Native benchmark ended with an unresolved policy query')
    return {name: (queries if name == 'policy_queries' else 0) for name in WORK_COUNTS
            if name not in ('physics_steps', 'policy_control_steps')}


def _aac_replay(output, protocol, job, archived, counts, intents):
    from helpers.maniguard_aac_runtime import replay_episode
    from helpers.maniguard_phase2_aac import load_config
    manifest = _signed(output / 'aac-runtime.json', 'maniguard-aac-runtime-v1')
    controller = _signed(archived[job['controller']['path']])
    config = load_config(archived[controller['aac_config']['path']])
    schedule = 'aac' if job['resource_contract']['profile'] == 'aac_joint20' else 'fixed8'
    if (manifest.get('job') != job or manifest.get('protocol_sha256') != protocol['sha256']
            or manifest.get('settings') != config.as_dict() or manifest.get('schedule') != schedule):
        raise ValueError('AAC actual sampler/controller differs from frozen scientific job')
    replay = replay_episode(output)
    if not replay['valid']:
        raise ValueError('AAC replay failed: ' + '; '.join(replay['invalid_reasons']))
    for name, count in replay['counts'].items():
        if counts.get(name) != count:
            raise ValueError('Main and AAC resource accounting disagree: ' + name)
    native_seed = zlib.crc32(f"{job['seed']}:{job['scene']}".encode()) & 0x7FFFFFFF
    for group in output.glob('aac-group-*'):
        if _read(group / 'group.json').get('native_episode_seed') != native_seed:
            raise ValueError('AAC streams used another native episode seed')
    action_events = [event for event in _events(output / 'aac-trace.jsonl')
                     if event.get('type') == 'action_intent']
    if len(action_events) != len(intents):
        raise ValueError('Main/AAC action intent inventories disagree')
    for main, native in zip(intents, action_events, strict=True):
        for left, right in (('raw', 'raw'), ('adapted', 'converted'),
                            ('environment_input', 'clipped'), ('action_space_low', 'low'),
                            ('action_space_high', 'high')):
            if not np.array_equal(main[left], native[right]):
                raise ValueError('Main/AAC action boundaries disagree: ' + left)


def read_main_episode(output, protocol, job, *, process_receipt=None, root=ROOT, config_dir=CONFIG):
    """Audit one explicit main job; invalid evidence remains invalid with paid work retained."""
    output, root = Path(output).resolve(), Path(root).resolve()
    row = {'schema': SCHEMA, 'episode_id': job.get('episode_id'), 'requested_job': deepcopy(job),
           'output': str(output), 'valid': False, 'invalid_reasons': [], 'outcome': None,
           'benchmark_result': None, 'execution_binding_verified': False, 'costs': {},
           'cost_breakdown': {name: {} for name in CATEGORIES}, 'cost_issues': [],
           'missing_cost_fields': [], 'unknown_cost_fields': [], 'full_episode': False,
           'audit_reader_sha256': _sha(Path(__file__)), 'protocol_sha256': protocol.get('sha256'),
           'outcome_reader_sha256': _sha(Path(__file__).with_name('maniguard_phase2_outcomes.py'))}
    ledger, journal = None, None
    try:
        ledger, journal = _costs(output, job, row)
    except AUDIT_ERRORS as error:
        row['cost_issues'].append(str(error))
    try:
        early_process = _signed(output / 'main_process.json', 'maniguard-main-process-v1')
        _bindings(early_process, protocol, job, output)
        elapsed = early_process.get('process', {}).get('process_elapsed_seconds')
        if _number(elapsed):
            row['costs']['elapsed_seconds'] = elapsed
    except AUDIT_ERRORS:
        pass  # The mandatory process audit below still rejects missing/invalid evidence.
    try:
        authorization = protocol_api.authorize_job(protocol, job, root=root, config_dir=config_dir)
        if authorization != {'authorized': True, 'protocol_sha256': protocol['sha256'],
                              'episode_id': job['episode_id'], 'job_sha256': record_hash(job)}:
            raise ValueError('Exact main job was not authorized by frozen protocol')
        recorded = _signed(output / 'main_job.json', 'maniguard-main-job-v1')
        execution = _signed(output / 'main_execution.json', 'maniguard-main-execution-v1')
        runtime = _signed(output / 'main_runtime_evidence.json', 'maniguard-main-runtime-evidence-v1')
        process = _signed(output / 'main_process.json', 'maniguard-main-process-v1')
        for record in (recorded, execution, runtime, process):
            _bindings(record, protocol, job, output)
        if recorded.get('job') != job:
            raise ValueError('Recorded compiled job differs from requested frozen job')
        if process_receipt is not None and process != process_receipt:
            raise ValueError('Supplied process receipt differs from durable parent receipt')
        request = _signed(_reference(output, execution['request']), 'maniguard-main-episode-request-v1')
        if request.get('job') != job or Path(request.get('output', '')).resolve() != output:
            raise ValueError('Archived parent request belongs to another job/output')
        if _signed(output / 'main_protocol.json') != protocol:
            raise ValueError('Archived protocol differs from the authorized protocol')
        actual_process = process['process']
        if actual_process.get('command') != request.get('command') or not request.get('command'):
            raise ValueError('Actual process command differs from immutable main request')
        if (type(actual_process.get('returncode')) is not int or actual_process.get('returncode') != 0 or actual_process.get('timed_out') is not False
                or actual_process.get('interrupted') is not False
                or actual_process.get('process_group_remaining') is not False
                or any(actual_process.get(name) for name in ('error', 'cleanup_error', 'orphaned_descendants'))):
            raise ValueError('Parent process did not exit normally with its descendants')
        elapsed = actual_process.get('process_elapsed_seconds')
        if not _number(elapsed):
            raise ValueError('Parent process elapsed time is missing or malformed')
        if ledger is not None and ledger['elapsed_seconds'] > elapsed + 1e-3:
            raise ValueError('Child elapsed time exceeds parent process lifetime')
        row['costs']['elapsed_seconds'] = elapsed
        archived = _archives(output, execution, protocol, job, runtime, root)
        carrier = _controller(output, execution, job, archived)
        trace = _events(output / ('aac-trace.jsonl' if execution['backend'] == 'aac' else 'trace.jsonl'))
        _handshake(output, runtime, job, archived, trace, root)
        if ledger is None or journal is None:
            raise ValueError('Complete independently replayed operation accounting is required')
        _physics(execution, runtime, request, protocol, archived, root, row['costs'])
        result, outcome = _outcome(output, job, trace, row['costs'])
        intents = _action_journal(output, trace, result['steps'], journal['operations'])
        if execution['backend'] == 'phase2':
            encoder = _read(archived['configs/maniguard/phase2/encoder.json'])
            counters = _phase2_replay(output, carrier, trace, result, encoder, intents)
            for name, expected in counters.items():
                if row['costs'].get(name) != expected:
                    raise ValueError('Metered count differs from execution replay: ' + name)
        elif execution['backend'] == 'aac':
            _aac_replay(output, protocol, job, archived, row['costs'], intents)
        else:
            counters = _original_replay(carrier, trace, result, intents)
            if any(row['costs'].get(name) != count for name, count in counters.items()):
                raise ValueError('Native benchmark resource counts differ from original-only execution')
        row.update(deepcopy(recorded['job']))
        row.update(execution_binding_verified=True, full_episode=True, short_check=False,
                   benchmark_result=result, outcome=outcome,
                   physics_accounting_scope=runtime.get('physics_enforcement', {}).get('accounting_scope'),
                   provenance_limitations=['Loaded file hashes do not attest in-memory Python code.',
                       'Policy checkpoint bytes are attested by the pinned startup server.',
                       'Physics scope excludes app startup before native callback registration.'])
    except AUDIT_ERRORS as error:
        row['invalid_reasons'].append(str(error))
    if ledger is None or journal is None:
        row['unknown_cost_fields'] = list(WORK_COUNTS)
        try:
            known, categories = _journal_lower_bounds(output / 'main_operations.jsonl')
            for name, count in known.items():
                if count > row['costs'].get(name, 0):
                    row['costs'][name] = count
            for category, counters in categories.items():
                for name, count in counters.items():
                    if count > row['cost_breakdown'][category].get(name, 0):
                        row['cost_breakdown'][category][name] = count
        except AUDIT_ERRORS:
            pass
    row['missing_cost_fields'] = [name for name in (*WORK_COUNTS, 'elapsed_seconds')
                                  if name not in row['costs']]
    if row['missing_cost_fields']:
        row['cost_issues'].append('Missing explicit costs: ' + ', '.join(row['missing_cost_fields']))
    row['invalid_reasons'].extend(row['cost_issues'])
    row['valid'] = not row['invalid_reasons'] and row['execution_binding_verified']
    if not row['valid']:
        row['outcome'] = None
    return row
