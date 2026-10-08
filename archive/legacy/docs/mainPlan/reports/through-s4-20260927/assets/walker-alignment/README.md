# Compact saved Walker S1 alignment view

This report-only figure uses the saved `traces.csv` and `alignment.json` from
`docs/mainPlan/results/continuation-v2/walker_s1_alignment`. No HDF5 file,
model, simulator, GPU job or network service was opened or queried.

The two columns show set A episodes 0 and 2, final 24 rows only. These are
explicitly chosen illustrative episodes from the prior post hoc audit: a
recorded time-limit ending and a recorded health-terminal ending. This is
neither a prevalence estimate nor a claim of preregistered selection. The prior
audit itself happened after pretraining began.

Height and pitch belong to pre-action state s[t]. Velocity, cost and ending
flags belong to the transition from t to t+1. The dashed velocity reconstruction
uses the next saved position and stops before the final row, whose successor is
absent. The final recorded velocity remains shown, but cannot be reconstructed
from these saved positions. A healthy last saved state does not contradict a
health-terminal transition. Its terminal successor cannot be checked here.

Limits and cost markers reproduce the saved rules and fields: strict height
(0.8, 2.0 m), pitch (-1, 1 rad), and signed velocity > 2.3415 m/s. This is not
an absolute-speed rule. Recorded ending flags are retained, not inferred from
the plotted pre-action states.

The original CSV and JSON agree exactly for every field in all 48 selected
rows. All original alignment-bundle file hashes were checked before and after.
`selected_rows.json` retains the exact selected JSON windows; `manifest.json`
records source and generator identities plus output hashes. The original
artifacts were not modified. No new scientific metric or gate was computed.

The figure is 7 by 7.4 inches with native PDF text and labels at least 9 pt.
Keep near its native width in the final report. PNG resolution is 1540 by 1628.

Reproduce to a new output directory:

```sh
uv run python runs/continuation-audit/report-walker-alignment-20260927/generator.py \
  --root /workspace/Safety-Dial --output /tmp/walker-alignment-review
```
