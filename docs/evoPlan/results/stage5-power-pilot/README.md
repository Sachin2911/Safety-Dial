# Bounded development search/power pilot

Run `walker2d-evo-s5-power-pilot-20261004-1` is in progress under supervisor
`evo_power_pilot`. The [prospective protocol](../../protocols/stage5-power-pilot-20261004.md)
was committed as `9c043c1` before launch. It is a bounded engineering/variance
pilot on inspected development episodes, not the main Stage 5 experiment.

Four search seeds compare k=0 and k=1 at populations 16 and 256, through 16
generations, with checkpoint selection on separate development episodes. There
is no extra unsafe penalty or ROSARL comparison. All checkpoint choices freeze
before paired real assessment. No final confirmation bank is generated.

The archived zero-correction imagined control is bitwise identical. All 657 tests passed (49 warnings), including the three focused checks. The
experiment is still in progress. Validation overlapped an early part of this
development run, so overlapping times are not an uncontended throughput benchmark.
No scientific outcome is reported yet. Costs, partial outputs and failure status
will be preserved if execution stops before completion.
