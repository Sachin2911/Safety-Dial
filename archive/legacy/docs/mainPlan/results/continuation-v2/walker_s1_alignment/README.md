# Post hoc Walker S1 alignment audit

This checks retained data after pretraining began. It is not evidence of a pretraining-time review. No simulator or model was queried.

Rows contain pre-action state and health. Cost, velocity and terminal flags concern the transition caused by the action at that row. State-frame velocity uses the preceding transition, unavailable at episode starts.

Fixed selection: episodes 0, 1, 2, 3 from setA and probe; first/last 24 rows. Root/test data were not inspected. Full source-file SHA256 was streamed; only bounded slices were decoded. Selection is illustrative, not a prevalence estimate.

| Check | Checked unique rows | Mismatches | Unavailable |
|---|---:|---:|---:|
| healthy | 384 | 0 | 0 |
| cost | 384 | 0 | 0 |
| velocity | 376 | 0 | 8 |
| termination | 380 | 0 | 4 |

The last post-action state of each episode is absent. Probe termination is disabled by the collector; its copied render-context manifest misleadingly says enabled. This audit preserves and documents that limitation.

## Recorded traces

All selected rows and action vectors are retained in traces.csv and alignment.json. The following last six rows per episode make the timing distinction inspectable.

### setA.h5, episode 0

| Row | t | Height | Pitch | Healthy | Transition speed | Cost | Term | Trunc | Successor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 994 | 994 | 1.076690 | 0.534031 | 1 | 2.193735 | 0 | 0 | 0 | saved |
| 995 | 995 | 1.077187 | 0.549174 | 1 | 2.267128 | 0 | 0 | 0 | saved |
| 996 | 996 | 1.079682 | 0.564997 | 1 | 2.266191 | 0 | 0 | 0 | saved |
| 997 | 997 | 1.084700 | 0.580264 | 1 | 2.258049 | 0 | 0 | 0 | saved |
| 998 | 998 | 1.092368 | 0.594096 | 1 | 2.246575 | 0 | 0 | 0 | saved |
| 999 | 999 | 1.102750 | 0.605798 | 1 | 2.209229 | 0 | 0 | 1 | unavailable |

### setA.h5, episode 1

| Row | t | Height | Pitch | Healthy | Transition speed | Cost | Term | Trunc | Successor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1994 | 994 | 1.108410 | -0.159697 | 1 | 1.945179 | 0 | 0 | 0 | saved |
| 1995 | 995 | 1.113332 | -0.131055 | 1 | 1.958931 | 0 | 0 | 0 | saved |
| 1996 | 996 | 1.118245 | -0.101498 | 1 | 1.983961 | 0 | 0 | 0 | saved |
| 1997 | 997 | 1.123039 | -0.070667 | 1 | 2.012770 | 0 | 0 | 0 | saved |
| 1998 | 998 | 1.127713 | -0.038702 | 1 | 2.043758 | 0 | 0 | 0 | saved |
| 1999 | 999 | 1.132074 | -0.005597 | 1 | 2.073308 | 0 | 0 | 1 | unavailable |

### setA.h5, episode 2

| Row | t | Height | Pitch | Healthy | Transition speed | Cost | Term | Trunc | Successor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 2433 | 433 | 0.912779 | 0.646002 | 1 | 2.980050 | 1 | 0 | 0 | saved |
| 2434 | 434 | 0.896050 | 0.641703 | 1 | 3.059160 | 1 | 0 | 0 | saved |
| 2435 | 435 | 0.877720 | 0.638780 | 1 | 3.175257 | 1 | 0 | 0 | saved |
| 2436 | 436 | 0.858318 | 0.638454 | 1 | 3.261041 | 1 | 0 | 0 | saved |
| 2437 | 437 | 0.837925 | 0.640840 | 1 | 3.326005 | 1 | 0 | 0 | saved |
| 2438 | 438 | 0.816584 | 0.645865 | 1 | 3.400062 | 1 | 1 | 0 | unavailable |

### setA.h5, episode 3

| Row | t | Height | Pitch | Healthy | Transition speed | Cost | Term | Trunc | Successor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 3433 | 994 | 1.092273 | 0.440670 | 1 | 2.286036 | 0 | 0 | 0 | saved |
| 3434 | 995 | 1.086804 | 0.472419 | 1 | 2.116250 | 0 | 0 | 0 | saved |
| 3435 | 996 | 1.082612 | 0.494626 | 1 | 2.044258 | 0 | 0 | 0 | saved |
| 3436 | 997 | 1.079496 | 0.511522 | 1 | 2.058295 | 0 | 0 | 0 | saved |
| 3437 | 998 | 1.077031 | 0.525653 | 1 | 2.116844 | 0 | 0 | 0 | saved |
| 3438 | 999 | 1.075516 | 0.540329 | 1 | 2.167703 | 0 | 0 | 1 | unavailable |

### probe.h5, episode 0

| Row | t | Height | Pitch | Healthy | Transition speed | Cost | Term | Trunc | Successor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 81 | 81 | 0.184163 | -3.933226 | 0 | 0.129546 | 0 | 0 | 0 | saved |
| 82 | 82 | 0.188978 | -3.948173 | 0 | 0.157446 | 0 | 0 | 0 | saved |
| 83 | 83 | 0.188110 | -3.957984 | 0 | 0.155176 | 0 | 0 | 0 | saved |
| 84 | 84 | 0.184510 | -3.961349 | 0 | 0.095624 | 0 | 0 | 0 | saved |
| 85 | 85 | 0.182993 | -3.963774 | 0 | 0.054771 | 0 | 0 | 0 | saved |
| 86 | 86 | 0.182692 | -3.967864 | 0 | -0.013627 | 0 | 0 | 0 | unavailable |

### probe.h5, episode 1

| Row | t | Height | Pitch | Healthy | Transition speed | Cost | Term | Trunc | Successor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 347 | 260 | 0.054595 | -1.547443 | 0 | 0.000008 | 0 | 0 | 0 | saved |
| 348 | 261 | 0.054597 | -1.547447 | 0 | 0.000005 | 0 | 0 | 0 | saved |
| 349 | 262 | 0.054598 | -1.547450 | 0 | 0.000003 | 0 | 0 | 0 | saved |
| 350 | 263 | 0.054599 | -1.547452 | 0 | 0.000002 | 0 | 0 | 0 | saved |
| 351 | 264 | 0.054600 | -1.547454 | 0 | 0.000001 | 0 | 0 | 0 | saved |
| 352 | 265 | 0.054600 | -1.547455 | 0 | 0.000001 | 0 | 0 | 0 | unavailable |

### probe.h5, episode 2

| Row | t | Height | Pitch | Healthy | Transition speed | Cost | Term | Trunc | Successor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 425 | 72 | 0.235109 | -3.502623 | 0 | -0.296071 | 0 | 0 | 0 | saved |
| 426 | 73 | 0.232871 | -3.474736 | 0 | -0.078568 | 0 | 0 | 0 | saved |
| 427 | 74 | 0.232649 | -3.461082 | 0 | 0.039970 | 0 | 0 | 0 | saved |
| 428 | 75 | 0.233464 | -3.452749 | 0 | 0.131618 | 0 | 0 | 0 | saved |
| 429 | 76 | 0.234777 | -3.445359 | 0 | 0.183936 | 0 | 0 | 0 | saved |
| 430 | 77 | 0.236353 | -3.438820 | 0 | 0.199318 | 0 | 0 | 0 | unavailable |

### probe.h5, episode 3

| Row | t | Height | Pitch | Healthy | Transition speed | Cost | Term | Trunc | Successor |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 618 | 187 | 0.080329 | 1.709790 | 0 | 0.085758 | 0 | 0 | 0 | saved |
| 619 | 188 | 0.082558 | 1.707545 | 0 | 0.096882 | 0 | 0 | 0 | saved |
| 620 | 189 | 0.084545 | 1.703872 | 0 | 0.112875 | 0 | 0 | 0 | saved |
| 621 | 190 | 0.086021 | 1.700206 | 0 | 0.108737 | 0 | 0 | 0 | saved |
| 622 | 191 | 0.086781 | 1.697184 | 0 | 0.100478 | 0 | 0 | 0 | saved |
| 623 | 192 | 0.086718 | 1.695348 | 0 | 0.099476 | 0 | 0 | 0 | unavailable |
