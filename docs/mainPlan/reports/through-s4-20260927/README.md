# Through-S4 research report

The experiments are complete through S4. Push-T stopped at the unchanged E2 repair/retention gate. Walker LeWM training completed at 214,780 updates, and the health-only S4 comparison completed all 12 adapted models. S5 was not launched.

- [Full PDF report](pdf-v2/report.pdf)
- [Readable report source](report.md) and [structured source](report.json)
- [Recorded Push-T MP4/GIF examples](../../results/continuation-v4/repair-animations-v2/README.md)
- [Walker S4 plots and full saved tables](../../results/s4/walker2d-s4-recovery-20260927-1-figures/README.md)
- [Compact report figures](assets/)
- [Delivery status and remote references](delivery.json)

The final result is mixed: adaptation improves Walker's saved dial-wide ranking score but does not show a consistent boundary-over-random advantage on the main test, and ordinary prediction errors often increase. Smaller-budget stress improvements occur with lower acceptance and do not repeat at the larger budget. The PDF records counts, uncertainty, costs, negative gates and provenance limits.

The final PDF is the version in `pdf-v2/`; earlier local draft renders are retained separately and are not final deliverables. All media use recorded evidence. Publication is tracked separately from scientific completion in `delivery.json`.
