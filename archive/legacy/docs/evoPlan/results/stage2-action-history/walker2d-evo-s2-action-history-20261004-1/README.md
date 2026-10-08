# Previous action memory improves imitation but leaves four development failures

A controlled action-history comparison produced these development outcomes:

| Arm | Health violations | PPO starts | PPO-Lagrangian starts | Progress in metres | Validation MSE |
|---|---:|---:|---:|---:|---:|
| zero_history | 6/24 | 3/8 | 3/16 | 2.073 | 0.08770 |
| action_history | 4/24 | 3/8 | 1/16 | 2.091 | 0.05776 |

Both arms use a 444-256-128-60 network with 154,556 parameters. The treatment
receives the preceding ten executed actions, normalised by the frozen world-model
action scaler. The control receives zeros in those same 60 inputs. Neither receives
teacher identity or privileged state. The remaining 384 inputs are the unchanged
visual latent and its block difference. Initial past actions come from root history;
later real and imagined decisions use their own executed actions.

Protocol and code were committed as 63ca137 before training. Both arms use exactly
the original mixed 45,986 fit examples and 23,640 common validation examples,
three seeds, 100 epochs, and 13,500 optimiser updates. Checkpoint selection uses
family-balanced offline validation MSE only. Preceding action reconstruction
checks every cached target against its source file, preventing temporal misalignment.

The paired violation change was -8.3 percentage points with a 95% source-episode
bootstrap interval [-29.2, 12.5]; progress changed by +0.018 m with interval
[-0.077, 0.133]. Neither interval excludes zero. The four treatment failures match
the earlier mixed controller's count, not necessarily the same failed starts.
The stronger offline fit does not establish a viable closed-loop controller.
Both arms were imagined safe on all 24 starts. No Gate 0 ran.

All 630 tests passed with 49 warnings and no skips or failures; Ruff passed.
Four new tests cover past-only cache reconstruction, trainable/stateless policy
equivalence, exact real physics and grouping, and imagined action feedback,
including multiple candidates and noisy samples.

New scientific costs were 4,800 simulator steps, 480 imagined rows, 27,000 optimiser
updates, 616 renders including 64 fingerprint frames, and 552 encoded frames.
Existing cached images and demonstration targets were reused; no collection ran.
The run took 44.93 seconds. Config, hashes, source, trajectories
and selected weights are saved in the run bundle. Archival is pending.
