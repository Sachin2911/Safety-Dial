# Stage 6 development report

All results below use reused Stage 5 episodes and are exploratory. The model, probe, and starting controller remain frozen. Two paired search seeds do not support a generalization claim.

Starting-controller final failure rate: 6.25%.

| Arm | Imagined-selected real failure | Simulator-selected real failure | Progress / baseline | Baseline fallbacks |
|---|---:|---:|---:|---:|
| k0-zero | 17.19% | 6.25% | 1.000 | 2/2 |
| k0-rosarl_style | 17.19% | 6.25% | 1.000 | 2/2 |
| k0-fixed-0.5 | 17.19% | 6.25% | 1.000 | 2/2 |
| k0-fixed-2 | 17.19% | 6.25% | 1.000 | 2/2 |
| k0-fixed-8 | 17.19% | 6.25% | 1.000 | 2/2 |
| k1-zero | 17.19% | 6.25% | 1.000 | 2/2 |
| k1-rosarl_style | 14.84% | 6.25% | 1.000 | 2/2 |
| k1-fixed-0.5 | 20.31% | 6.25% | 1.000 | 2/2 |
| k1-fixed-2 | 14.84% | 6.25% | 1.000 | 2/2 |
| k1-fixed-8 | 14.84% | 6.25% | 1.000 | 2/2 |

The predeclared development rule selects fixed penalty **0.5**. This choice can inform a later locked confirmation protocol; confirmation sizes and fresh data have not been locked or collected.

Actual costs: 4,241,280 predictor rows; 134,400 simulator steps; 9.16 minutes; zero gradient updates. Source-collection costs are zero for this reused-bank run; historical source collection remains charged to Stage 5.

All 635 query archives and 110 source snapshots verified. Every simulator selection decision and the complete statistical analysis were independently reproduced from saved numeric outcomes.

The planned next steps are a fresh-bank Walker2d confirmation with a locked fixed penalty, then Hopper and independent world-model replications. No fresh or cross-environment result is claimed here.
