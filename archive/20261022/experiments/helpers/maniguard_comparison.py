"""Compare conditions within declared scene/seed cases without hiding missing work."""

from collections import Counter

from helpers.maniguard_batch_report import OUTCOMES, summarize_batch

PAIR_COSTS = ('policy_control_steps', 'physics_steps', 'policy_queries')


def outcome(row):
    return next(name for name in OUTCOMES if row['result'][name])


def case_member(row):
    if row is None:
        return {'status': 'not_planned'}
    result = {key: row[key] for key in ('index', 'status', 'output')}
    if row['status'] == 'completed':
        result['outcome'] = outcome(row)
        result['costs'] = {key: row['result']['ledger'].get(key) for key in PAIR_COSTS}
    return result


def compare_batch(directory, control_label='identity'):
    report = summarize_batch(directory)
    groups = report['groups']
    if len({group['family'] for group in groups}) != 1:
        raise ValueError('Compare one policy family at a time')
    labels = [group['label'] for group in groups]
    if control_label not in labels:
        raise ValueError('Requested control condition is not in the batch')
    if len(labels) < 2:
        raise ValueError('Comparison requires a control and another condition')
    indexed = {label: {} for label in labels}
    for row in report['episodes']:
        key = (row['family'], row['scene'], row['seed'])
        if key in indexed[row['label']]:
            raise ValueError('Duplicate condition/scene/seed; distinguish repeated trials explicitly')
        indexed[row['label']][key] = row
    controls = indexed[control_label]
    comparisons = []
    for label in labels:
        if label == control_label:
            continue
        candidates = indexed[label]
        keys = sorted(controls.keys() | candidates.keys())
        cases, completed = [], []
        planned_pairs = 0
        for key in keys:
            control, candidate = controls.get(key), candidates.get(key)
            planned_pairs += int(control is not None and candidate is not None)
            cases.append({'family': key[0], 'scene': key[1], 'seed': key[2],
                          'control': case_member(control), 'candidate': case_member(candidate)})
            if (control is not None and candidate is not None
                    and control['status'] == candidate['status'] == 'completed'):
                completed.append((control, candidate))
        transitions = Counter((outcome(control), outcome(candidate))
                              for control, candidate in completed)
        pair_costs = None
        if completed and all(key in row['result']['ledger'] for pair in completed
                             for row in pair for key in PAIR_COSTS):
            pair_costs = {
                side: {key: sum(pair[index]['result']['ledger'][key] for pair in completed)
                       for key in PAIR_COSTS}
                for index, side in enumerate(('control', 'candidate'))
            }
        comparisons.append({
            'control_label': control_label, 'candidate_label': label,
            'case_slots': len(keys), 'planned_pairs': planned_pairs,
            'completed_pairs': len(completed), 'all_cases_complete': len(completed) == len(keys),
            'cases': cases,
            'outcome_transitions': {
                before: {after: transitions[(before, after)] for after in OUTCOMES}
                for before in OUTCOMES
            },
            'paired_outcome_counts': {
                side: {name: sum(pair[index]['result'][name] for pair in completed)
                       for name in OUTCOMES}
                for index, side in enumerate(('control', 'candidate'))
            },
            'safe_completion_gains': sum(not a['result']['safe_completion']
                                         and b['result']['safe_completion'] for a, b in completed),
            'safe_completion_losses': sum(a['result']['safe_completion']
                                          and not b['result']['safe_completion'] for a, b in completed),
            'violation_count_difference': sum(int(b['result']['violation'])
                                              - int(a['result']['violation']) for a, b in completed),
            'completed_pair_cost_totals': pair_costs,
        })
    return {
        'batch_report': report, 'comparisons': comparisons,
        'comparison_scope': ('Outcome transitions use only cases where both episodes completed and '
                             'validated. Every planned or unmatched case remains visible. Partial '
                             'comparisons are not a balanced condition ranking.'),
        'cost_scope': ('Whole-batch recorded_cost_totals count each episode once, including failed '
                       'and partial work. Pair totals cover only the completed pairs in that '
                       'comparison, and are null if a required cost is missing. Controls can appear '
                       'in several comparisons; never sum pair totals as whole-study costs.'),
        'interpretation': ('Shared scene/root seeds do not imply identical initial images or '
                           'physical trajectories. These are descriptive outcome comparisons, not '
                           'causal counterfactuals or an evolutionary-advantage claim.'),
    }
