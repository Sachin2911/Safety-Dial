# Walker S4: compact saved paired comparisons

Figure caption: Each dot is the saved boundary-minus-random false-safe acceptance (FSA) difference for one acquisition seed and additional-branch budget N. Horizontal segments are the exact saved 95% bootstrap intervals clustered by source episode. Negative differences favor boundary. Test and stress are separate panels; seeds are not pooled. Each model uses its own margin calibrated on development and held fixed on test/stress. Achieved acceptance rates differ between arms, so these are not comparisons at exactly equal test or stress acceptance. The figure copies saved estimates and intervals; it performs no new bootstrap or population-level inference. Undefined full-data estimates have no dot or interval, even if some bootstrap replicates were usable. N counts additional branches; allocated charged simulator steps also include source generation and common-seed experience. Repeated allocations must not be summed as a lifetime cost.

The PNG is 7 inches wide at 200 dpi; the PDF is vector output. All figure text is at least 10.5 pt. `plotted_data.json` retains each complete saved paired record, including exact accepted counts, censor support, achieved acceptance and bootstrap support. Nonfinite saved values become JSON null without being replaced by zero. `manifest.json` binds original inputs, qualified output identities, and inert generator bytes.

## Test

| N | Seed | Charged steps | Random AR | Boundary AR | Random FSA | Boundary FSA | Difference (pp) | 95% interval (pp) | Usable bootstrap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 0 | 325,638 | 96.875% | 97.266% | 43.548% | 43.976% | 0.428 | [-0.155, 1.147] | 1000/1000 |
| 128 | 1 | 325,638 | 98.047% | 95.703% | 43.825% | 43.265% | -0.559 | [-1.583, 0.326] | 1000/1000 |
| 128 | 2 | 325,638 | 97.266% | 97.070% | 44.378% | 44.266% | -0.112 | [-0.959, 0.676] | 1000/1000 |
| 512 | 0 | 364,038 | 96.680% | 95.312% | 44.040% | 43.238% | -0.803 | [-1.737, 0.000] | 1000/1000 |
| 512 | 1 | 364,038 | 96.094% | 96.289% | 42.886% | 43.408% | 0.522 | [-0.091, 1.198] | 1000/1000 |
| 512 | 2 | 364,038 | 95.312% | 98.438% | 43.443% | 44.841% | 1.399 | [0.472, 2.598] | 1000/1000 |

## Stress

| N | Seed | Charged steps | Random AR | Boundary AR | Random FSA | Boundary FSA | Difference (pp) | 95% interval (pp) | Usable bootstrap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 0 | 325,638 | 91.992% | 87.891% | 71.550% | 70.000% | -1.550 | [-2.787, -0.193] | 1000/1000 |
| 128 | 1 | 325,638 | 91.797% | 87.305% | 71.277% | 69.799% | -1.478 | [-2.436, -0.525] | 1000/1000 |
| 128 | 2 | 325,638 | 95.117% | 92.383% | 72.074% | 71.036% | -1.038 | [-1.805, -0.316] | 1000/1000 |
| 512 | 0 | 364,038 | 89.258% | 87.695% | 70.022% | 70.379% | 0.357 | [-0.911, 1.737] | 1000/1000 |
| 512 | 1 | 364,038 | 90.625% | 89.648% | 70.905% | 70.806% | -0.099 | [-1.077, 0.866] | 1000/1000 |
| 512 | 2 | 364,038 | 93.750% | 95.312% | 71.875% | 71.926% | 0.051 | [-0.866, 0.808] | 1000/1000 |
