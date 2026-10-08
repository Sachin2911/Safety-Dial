# Stage 6 engineering report

All results below use reused Stage 5 episodes and are exploratory. The model, probe, and starting controller remain frozen. Two paired search seeds do not support a generalization claim.

Starting-controller final failure rate: 0.00%.

| Arm | Imagined-selected real failure | Simulator-selected real failure | Progress / baseline | Baseline fallbacks |
|---|---:|---:|---:|---:|
| k0-zero | 0.00% | 0.00% | 1.000 | 2/2 |
| k0-rosarl_style | 0.00% | 0.00% | 1.000 | 2/2 |
| k0-fixed-2 | 0.00% | 0.00% | 1.000 | 2/2 |
| k1-zero | 6.25% | 0.00% | 1.000 | 2/2 |
| k1-rosarl_style | 6.25% | 0.00% | 1.000 | 2/2 |
| k1-fixed-2 | 6.25% | 0.00% | 1.000 | 2/2 |

This engineering smoke run validates execution only. It does not choose the confirmation penalty.

Actual costs: 22,480 predictor rows; 3,600 simulator steps; 0.27 minutes; zero gradient updates. Source-collection costs are zero for this reused-bank run; historical source collection remains charged to Stage 5.

All 117 query archives and 109 source snapshots verified. Every simulator selection decision and the complete statistical analysis were independently reproduced from saved numeric outcomes.

The planned next steps are a fresh-bank Walker2d confirmation with a locked fixed penalty, then Hopper and independent world-model replications. No fresh or cross-environment result is claimed here.
