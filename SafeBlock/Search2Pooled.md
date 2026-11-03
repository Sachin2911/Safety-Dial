# Search 2 pooled

Protocol: [Search2.md](Search2.md). Repeats present [0, 1, 2], complete [0, 1, 2].

## Per repeat

| repeat | controller | sustained | success | violated |
|---:|---|---|---|---|
| 0 | Native `ap_rank` | 3/16 [0.07, 0.43] | 9/16 [0.33, 0.77] | 10/16 [0.39, 0.81] |
| 0 | Untrained selector | 6/16 [0.18, 0.61] | 12/16 [0.51, 0.90] | 4/16 [0.10, 0.49] |
| 0 | CMA-ES nominee | 4/16 [0.10, 0.49] | 5/16 [0.14, 0.56] | 11/16 [0.44, 0.86] |
| 1 | Native `ap_rank` | 2/16 [0.04, 0.36] | 10/16 [0.39, 0.81] | 9/16 [0.33, 0.77] |
| 1 | Untrained selector | 6/16 [0.18, 0.61] | 10/16 [0.39, 0.81] | 9/16 [0.33, 0.77] |
| 1 | Random-search nominee | 6/16 [0.18, 0.61] | 9/16 [0.33, 0.77] | 6/16 [0.18, 0.61] |
| 1 | CMA-ES nominee | 5/16 [0.14, 0.56] | 7/16 [0.23, 0.67] | 10/16 [0.39, 0.81] |
| 2 | Native `ap_rank` | 4/16 [0.10, 0.49] | 8/16 [0.28, 0.72] | 9/16 [0.33, 0.77] |
| 2 | Untrained selector | 4/16 [0.10, 0.49] | 7/16 [0.23, 0.67] | 11/16 [0.44, 0.86] |
| 2 | CMA-ES nominee | 5/16 [0.14, 0.56] | 9/16 [0.33, 0.77] | 9/16 [0.33, 0.77] |

## Pooled final panel

48 starts, paired by start (random-search arm complete in repeats [1] only).

| controller | sustained | success | violated |
|---|---|---|---|
| Native `ap_rank` | 9/48 [0.10, 0.32] | 27/48 [0.42, 0.69] | 28/48 [0.44, 0.71] |
| Untrained selector | 16/48 [0.22, 0.47] | 29/48 [0.46, 0.73] | 24/48 [0.36, 0.64] |
| Random-search nominee | 6/16 [0.18, 0.61] | 9/16 [0.33, 0.77] | 6/16 [0.18, 0.61] |
| CMA-ES nominee | 14/48 [0.18, 0.43] | 21/48 [0.31, 0.58] | 30/48 [0.48, 0.75] |

| controller | sustained gained / lost | p | violated added / removed | p | per-repeat sustained difference |
|---|---|---:|---|---:|---|
| Untrained selector | 10 / 3 | 0.092 | 7 / 11 | 0.481 | +3, +4, +0 |
| Random-search nominee | 4 / 0 | 0.125 | 1 / 4 | 0.375 | +4 |
| CMA-ES nominee | 9 / 4 | 0.267 | 8 / 6 | 0.791 | +1, +3, +1 |

## Pooled verdict: **mixed** (CMA minus native: sustained +5, violated +2; random-search tie None)

## Nominees and selection

| repeat | nominee source | retrieval_distance | imagined_cost | mean_arm | closing_fraction | progress | imagined_min_q | imagined_failure_probability | imagined_disagreement |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | final_mean | -0.09 | -0.13 | +0.79 | -0.17 | +0.43 | +0.69 | +0.02 | +0.25 |
| 1 | final_mean | +1.62 | +0.60 | -0.86 | -0.02 | +0.88 | +0.19 | -0.23 | +1.01 |
| 2 | elite | -0.54 | +0.15 | -0.22 | -0.26 | +1.54 | +0.98 | -3.95 | +0.62 |

