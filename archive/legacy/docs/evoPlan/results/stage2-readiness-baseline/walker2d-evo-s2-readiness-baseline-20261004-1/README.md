# Fresh Walker baseline: research measurement is feasible, reliable control is not established

The frozen six-member visual ensemble violated the dense health rule in 51 of 256
fresh independent episode segments (19.9%; 95% Wilson interval 15.5% to 25.2%).
Imagination predicted two violations (0.8%). The real-minus-imagined violation gap
was 19.1 percentage points, with paired episode-bootstrap interval 14.5 to 24.2.
The probe on real images reported 62 violations. Mean forward progress
was 2.089 m versus 2.083 m for exact recorded-action replay. The paired ratio was
1.003, interval 0.986 to 1.019. Recorded replay had zero observed violations.

The new bank used one root from each of 256 newly generated episodes. Source actors
were drawn uniformly from the 32 competent exports frozen by the historical S1
ladder, crossed with action noise 0, 0.05 and 0.1. Environment seeds started at
920261004, actor-noise seeds at 930261004. Roots were sampled from each eligible
episode with the same surviving-future conditioning as the original representative
bank. No attempt was rejected; all 256 attempts were eligible. The complete episode
records are retained, including steps beyond the sampled root. This conditional
root distribution is not an unconditional full-episode deployment safety estimate.

Controller bytes were frozen by SHA256 in the config before new episodes were
generated. There was no fitting, checkpoint selection, noise calibration or search
on this bank. Episode identities are distinct from every previous data collection.
After this assessment the bank is development evidence, not an untouched final
search test. Each root is its own independent episode cluster for the intervals.

The original strict criterion still fails: 51 violations exceed the allowed three
on a zero-violation reference. This fresh-bank check does not replace the historical
256-root gate or change its results. No safe-controller claim is supported.

The separately adopted baseline measurement-readiness conditions pass: the lower
progress-ratio bound exceeds 0.5; the upper failure bound is below 0.90, leaving
headroom for the prospective 10-point effect; and the baseline-gap interval width
of 9.77 points is below the predeclared 20-point ceiling. These coarse criteria
permit a bounded transfer and statistical-power pilot only. They do not establish
power for the final search effect, which must account for search-seed variation,
and do not authorize the full fix grid. A working reliable visual controller
remains an unmet goal.

The run spent 247,351 source-episode simulator steps and teacher queries plus
51,200 evaluation steps, for 298,551 new simulator steps in total. It used 2,560
imagined rows, 3,392 renders, 3,328 encoded frames and zero optimiser updates.
Wall time was 67.60 seconds. All 648 tests passed with 49 warnings, and Ruff passed.
Protocol decision commit: 7f733b3; executed implementation: e461e94.

The next step is a fixed, bounded transfer pilot around this frozen baseline,
with the root roles, perturbation scales, simulator budget and outcome measures
declared before execution. Poor or undefined model ranking must be reported and
may route the study to its explicit fallback; it is not permission to redefine
success after observing the pilot. The run's full bundle is complete locally.
Private archival is pending.
