#!/usr/bin/env python3
"""Recompute draft accounting and illustrative power sensitivity; runs no experiment."""
import argparse
import json
from pathlib import Path
from statistics import NormalDist

import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT/'configs/evo/stage5_readiness_draft.yaml').read_text())
    assert cfg['status'] == 'draft_for_supervisor_review' and not cfg['execution_enabled']
    bank, search, ev = cfg['bank'], cfg['search'], cfg['evaluation']
    s = len(search['search_seeds'])
    populations = search['population_sizes']
    g, k = search['generations'], ev['horizon_model_blocks']
    n_rules = len(search['penalty_modes'])
    n_arms = n_rules*len(search['noise_levels'])
    n_runs = len(populations)*s*n_arms
    n_picks = n_runs*ev['selected_policy_count_per_run']
    fitness = sum(populations)*g*s*n_rules*sum(search['fitness_samples'].values())*search['fitness_roots_per_generation']*k
    selection = len(populations)*s*n_rules*(g+1)*sum(search['selection_samples'].values())*bank['selection_episodes']*k
    audit = (n_picks+s)*bank['evaluation_episodes']*k*sum(ev['audit_samples'].values())
    real = (n_picks+2)*bank['evaluation_episodes']*ev['horizon_sim_steps']
    roots = sum(bank[key] for key in ['fitness_episodes', 'selection_episodes', 'evaluation_episodes'])
    max_collection = roots*bank['max_attempts_per_required_root']*bank['max_steps_per_source_episode']
    std_normal = NormalDist()
    delta = cfg['analysis']['confirmatory_effect_for_planning']
    scenarios = []
    # NOT a fitted crossed variance model: the old single-winner variance is assigned
    # to the root component for sensitivity only. The interaction component is unknown.
    root_var = 0.34576612903225806
    for alpha in [.05, .025]:
        critical = std_normal.inv_cdf(1-alpha/2)
        for seed_sd in [0., .05, .10, .15, .20]:
            se = (root_var/bank['evaluation_episodes']+seed_sd**2/s)**.5
            power = 1-std_normal.cdf(critical-delta/se)+std_normal.cdf(-critical-delta/se)
            scenarios.append(dict(alpha=alpha, assumed_paired_seed_sd=seed_sd,
                assumed_root_variance=root_var, assumed_interaction_variance=0.,
                standard_error=se, approximate_two_sided_power=power))
    out = dict(status='draft accounting and sensitivity only; no new data or experiments',
        config='configs/evo/stage5_readiness_draft.yaml',
        search_runs=n_runs, selected_policy_slots=n_picks, new_source_roots=roots,
        fitness_predictor_rows=fitness, selection_predictor_rows=selection,
        final_audit_predictor_rows=audit, total_predictor_rows=fitness+selection+audit,
        final_real_steps=real, max_source_collection_steps=max_collection,
        max_real_steps_excluding_debug_and_power_pilots=real+max_collection,
        cached_duplicate_policy_savings_assumed=False,
        baseline_imagination_repeated_per_search_seed=True,
        source_collection_attempt_cap_is_per_role=True,
        power_scenarios=scenarios,
        power_warning='Illustrations only, not evidence of 80% power: seed/root interaction, the second contrast variance, Monte Carlo error and variance-component estimates remain unmeasured.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({key:value for key,value in out.items() if key != 'power_scenarios'}, indent=2))


if __name__ == '__main__':
    main()
