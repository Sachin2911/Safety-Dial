# Compact E2 representative-test figure

Use `e2-representative-bounds.png` in the report and retain the vector PDF as a supplement. Native size is 7 by 4.9 inches; all text is at least 10.5 pt except no text below 10 pt is permitted by the render check. No full-resolution original chart is changed.

**Caption.** Recorded decisions on 4,096 representative-test labelled cases from 128 source roots. Left: each arm's achieved acceptance rate. Right: saved false-safe-acceptance (FSA) lower/upper identification bounds for its accepted set. All four FSA point estimates are undefined because some accepted futures have unresolved outcomes. Colored rectangular bands have no midpoint, confidence-interval interpretation or sampling uncertainty claim. The no-update, adapted and readout-correction thresholds were fixed from development; their achieved test acceptance rates differ. Fixed margin is the recorded 80 px fallback and accepts only 3.2% of plans. These are different fixed accepted sets, not matched test acceptance. Stress-bank results are separate.

| Arm | Accepted / 4,096 | Known false-safe | Accepted unresolved | Acceptance | Saved FSA bounds |
|---|---:|---:|---:|---:|---:|
| No update | 3,139 | 756 | 49 | 76.636% | 24.084% to 25.645% |
| Adapted | 2,950 | 620 | 50 | 72.021% | 21.017% to 22.712% |
| Readout correction | 2,905 | 795 | 34 | 70.923% | 27.367% to 28.537% |
| Fixed margin (80 px) | 132 | 4 | 12 | 3.223% | 3.030% to 12.121% |

The saved conservative outer identification interval for relative reduction, adapted versus no update on these fixed accepted sets, is 5.6977% to 18.0470%; its upper bound is below the specified 25% target. Shared unresolved labels need not jointly attain the endpoints. This is not a sampling confidence interval, a population inference, evidence of no improvement, or a replacement for the original undefined-point gate. The saved gate remains unchanged.

**Provenance.** Values are copied from `docs/mainPlan/results/continuation-v4/repair-summary-charts/chart_data.json`, specifically its `test` entry. Exact unrounded values and original source pointers remain in `plotted_data.json`. Only percentage conversion and display rounding are used. No new statistical estimation, threshold selection, bootstrap, model, physics or network call is performed. The manifest hashes this source, the original chart directory and this generator before and after rendering. The fresh bundle is local report preparation, not a publication receipt.
