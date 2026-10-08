"""Declared crossed-seed/episode estimators for the four-arm readiness study.

These calculations neither authorize the protocol nor turn reused development data
into confirmation. Source roles and complete evaluation receipts are caller obligations.
"""
from itertools import product

import numpy as np

from helpers.evoStats import wilson


class CrossedBootstrap:
    """One shared set of independently drawn seed/episode weights for every contrast."""
    def __init__(self,n_seeds,n_episodes,*,replicates,seed):
        if min(n_seeds,n_episodes)<2 or replicates<2:
            raise ValueError('at least two independent seeds/episodes and replicates required')
        self.shape=(n_seeds,n_episodes)
        self.replicates,self.seed=int(replicates),int(seed)
        rng=np.random.default_rng(seed)
        self.seed_weights=rng.multinomial(n_seeds,np.full(n_seeds,1/n_seeds),size=replicates)/n_seeds
        self.episode_weights=rng.multinomial(n_episodes,np.full(n_episodes,1/n_episodes),size=replicates)/n_episodes

    def array(self,value):
        x=np.asarray(value,np.float64)
        if x.shape!=self.shape or not np.isfinite(x).all():
            raise ValueError('finite seed-by-independent-episode matrix required')
        return x

    def draws(self,value):
        x=self.array(value)
        return np.einsum('br,br->b',self.seed_weights@x,self.episode_weights)

    def interval(self,value,*,coverage=.95):
        if not 0<coverage<1:
            raise ValueError('coverage must lie strictly between zero and one')
        x=self.array(value)
        q=(1-coverage)/2
        draws=self.draws(x)
        lo,hi=np.quantile(draws,[q,1-q])
        degenerate=bool(np.allclose(draws,draws[0],rtol=0,atol=1e-14))
        return dict(point=float(x.mean()),lo=float(lo),hi=float(hi),coverage=coverage,
                    seed_means=x.mean(1).tolist(),degenerate_empirical_bootstrap=degenerate,
                    warning='Degenerate empirical resampling does not establish zero population risk or no possible effect.' if degenerate else None)

    def ratio(self,numerator,denominator,*,coverage=.95):
        if not 0<coverage<1:
            raise ValueError('coverage must lie strictly between zero and one')
        n,d=self.array(numerator),self.array(denominator)
        nd,dd=self.draws(n),self.draws(d)
        valid=bool(d.mean()>0 and np.all(dd>0))
        out=dict(defined=valid,point=None,lo=None,hi=None,coverage=coverage,
                 nonpositive_bootstrap_denominators=int((dd<=0).sum()))
        if valid:
            q=(1-coverage)/2
            lo,hi=np.quantile(nd/dd,[q,1-q])
            out.update(point=float(n.mean()/d.mean()),lo=float(lo),hi=float(hi))
        return out


def _validate_cell(cell,shape):
    real=np.asarray(cell['real'])
    if real.shape!=shape or not np.isin(real,[0,1]).all():
        raise ValueError('complete binary seed-by-episode real outcomes required')
    progress=np.asarray(cell['progress'],np.float64)
    if progress.shape!=shape or not np.isfinite(progress).all():
        raise ValueError('complete finite real progress required')
    if set(cell['imagined'])!={0.,1.}:
        raise ValueError('both k=0 and k=1 audits required for every policy')
    imagined={float(k):np.asarray(v,np.float64) for k,v in cell['imagined'].items()}
    if any(v.shape!=shape or not np.isfinite(v).all() or ((v<0)|(v>1)).any() for v in imagined.values()):
        raise ValueError('complete imagined probabilities in [0,1] required')
    return dict(real=real.astype(np.float64),progress=progress,imagined=imagined)


def analyze_readiness(cells,baseline,*,search_seeds,episode_keys,population_sizes,checkpoints,
                      low_pressure=(16,1),high_pressure=(256,16),replicates=10000,
                      bootstrap_seed=20261120,minimum_progress_ratio=.90):
    """Cells keyed (noise_k, penalty_mode, population, generation), each S x R.

    Every cell supplies binary real failures, real full-horizon progress and both
    imagined audits. The real baseline is shared across seeds; its noisy audits are
    seed-specific. All intervals share exactly the same crossed resampling weights.
    """
    s,r=len(search_seeds),len(episode_keys)
    if len(set(search_seeds))!=s or len(set(episode_keys))!=r:
        raise ValueError('unique search seeds and independent source-episode keys required')
    if len(set(population_sizes))!=len(population_sizes) or len(set(checkpoints))!=len(checkpoints):
        raise ValueError('unique populations and checkpoints required')
    if any(g<=0 for g in checkpoints) or any(p<2 for p in population_sizes):
        raise ValueError('nonbaseline checkpoints and valid populations required')
    if not np.isfinite(minimum_progress_ratio) or minimum_progress_ratio<=0:
        raise ValueError('positive progress retention threshold required')
    expected=set(product([0.,1.],['zero','rosarl_style'],population_sizes,checkpoints))
    if set(cells)!=expected:
        raise ValueError('complete declared four-arm Cartesian grid required; no dropped cells')
    if low_pressure not in set(product(population_sizes,checkpoints)) or high_pressure not in set(product(population_sizes,checkpoints)):
        raise ValueError('declared pressure endpoints are missing')
    cells={key:_validate_cell(value,(s,r)) for key,value in cells.items()}
    baseline=_validate_cell(baseline,(s,r))
    if not all(np.array_equal(baseline[name],np.broadcast_to(baseline[name][0],(s,r))) for name in ['real','progress']):
        raise ValueError('one shared real baseline is required across search seeds')
    boot=CrossedBootstrap(s,r,replicates=replicates,seed=bootstrap_seed)
    def get(k,mode,pressure):
        return cells[(k,mode,*pressure)]
    def gap(cell,k):
        return cell['real']-cell['imagined'][k]
    zero_low=get(0.,'zero',low_pressure)
    zero_high=get(0.,'zero',high_pressure)
    fix_low=get(1.,'rosarl_style',low_pressure)
    fix_high=get(1.,'rosarl_style',high_pressure)
    amp_zero=gap(zero_high,0.)-gap(zero_low,0.)
    amp_fix=gap(fix_high,1.)-gap(fix_low,1.)
    first=boot.interval(amp_zero,coverage=.975)
    second=boot.interval(amp_fix-amp_zero,coverage=.975)
    primary=dict(k0_zero_gap_amplification=first,
        combined_method_minus_k0_zero_amplification=second,
        family_coverage=.95,individual_coverage=.975,multiplicity='Bonferroni two declared contrasts',
        amplification_interval_above_zero=first['lo']>0,
        combined_amplification_difference_interval_below_zero=second['hi']<0)
    real_difference=boot.interval(fix_high['real']-zero_high['real'])
    retention=boot.ratio(fix_high['progress'],zero_high['progress'])
    practical=dict(comparison='high-pressure k1 ROSARL-style minus high-pressure k0 zero penalty',
        real_failure_difference=real_difference,ratio_of_mean_real_progress=retention,
        minimum_progress_ratio=minimum_progress_ratio,
        real_reduction_supported=real_difference['hi']<0,
        progress_retention_supported=bool(retention['defined'] and retention['lo']>=minimum_progress_ratio),
        confidence_note='descriptive paired 95% intervals; distinct from co-primary inference',
        safe_controller_or_calibration_claim=False)
    practical['both_practical_conditions_supported']=practical['real_reduction_supported'] and practical['progress_retention_supported']
    def fixed_policy_rates(real):
        return [dict(search_seed=int(seed),violations=int(row.sum()),independent_episodes=r,
            point=float(row.mean()),wilson_95=list(wilson(int(row.sum()),r)),
            approximate_zero_failure_upper_95=3/r if row.sum()==0 else None)
            for seed,row in zip(search_seeds,real)]
    curves=[]
    for (k,mode,pop,g),cell in sorted(cells.items()):
        own=gap(cell,k)
        curves.append(dict(noise_k=k,penalty_mode=mode,population=pop,generation=g,
            offspring_evaluations=pop*g,real_failure=boot.interval(cell['real']),
            fixed_policy_real_rates_by_seed=fixed_policy_rates(cell['real']),
            imagined_failure=boot.interval(cell['imagined'][k]),gap_own_noise=boot.interval(own),
            gap_common_k0=boot.interval(gap(cell,0.)),real_progress=boot.interval(cell['progress']),
            real_failure_change_from_baseline=boot.interval(cell['real']-baseline['real']),
            own_gap_change_from_baseline=boot.interval(own-gap(baseline,k)),
            common_k0_gap_change_from_baseline=boot.interval(gap(cell,0.)-gap(baseline,0.)),
            progress_ratio_to_baseline=boot.ratio(cell['progress'],baseline['progress'])))
    factorial=[]
    pairs=[('penalty_at_k0',(0.,'rosarl_style'),(0.,'zero')),
           ('penalty_at_k1',(1.,'rosarl_style'),(1.,'zero')),
           ('noise_at_zero_penalty',(1.,'zero'),(0.,'zero')),
           ('noise_at_rosarl',(1.,'rosarl_style'),(0.,'rosarl_style'))]
    for label,a,b in pairs:
        high_a,high_b=get(*a,high_pressure),get(*b,high_pressure)
        low_a,low_b=get(*a,low_pressure),get(*b,low_pressure)
        delta_amp=(gap(high_a,a[0])-gap(low_a,a[0]))-(gap(high_b,b[0])-gap(low_b,b[0]))
        factorial.append(dict(contrast=label,pressure='high',
            real_failure_difference=boot.interval(high_a['real']-high_b['real']),
            own_noise_gap_difference=boot.interval(gap(high_a,a[0])-gap(high_b,b[0])),
            common_k0_gap_difference=boot.interval(gap(high_a,0.)-gap(high_b,0.)),
            amplification_difference=boot.interval(delta_amp),
            real_progress_ratio=boot.ratio(high_a['progress'],high_b['progress'])))
    baseline_summary=dict(real_failure=boot.interval(baseline['real']),
        real_progress=boot.interval(baseline['progress']),
        fixed_policy_real_rate=fixed_policy_rates(baseline['real'])[0],
        imagined_failure={str(k):boot.interval(baseline['imagined'][k]) for k in [0.,1.]},
        gap={str(k):boot.interval(gap(baseline,k)) for k in [0.,1.]})
    return dict(primary=primary,practical_benefit=practical,curves=curves,baseline=baseline_summary,
        descriptive_factorial_contrasts=factorial,
        sampling=dict(search_seeds=list(search_seeds),independent_episode_count=r,
            seed_count=s,bootstrap_replicates=replicates,bootstrap_seed=bootstrap_seed,
            resampling='independent seed and source-episode axes; identical draws across every contrast',
            conditional_on='frozen assets, fitness/selection banks and source distribution',
            total_pairs_are_not_independent_episodes=s*r),
        pressure=dict(low=list(low_pressure),high=list(high_pressure)),
        outcome_policy='report every interval; no posthoc replacement of a failed primary contrast')
