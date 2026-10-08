# Walker2d animations from the trained LeWM (walker2d-lewm-a-recovery-20260927-1)

Checkpoint at 214,780 optimizer updates (`weights.pt` sha256 `35aa484963dd149e...`), read out with the frozen S3 probe `runs/walker2d-probes-recovery-20260927-1/walker_mlp.pt`. Rebuild with `uv run python experiments/scripts/walker_lewm_animations.py` (about four minutes on the 5090).

**These are illustrations, not new statistics.** Each clip is one saved branch. The counts beside each clip say how common its outcome is in the saved S4 rows; the S4 report holds the rates, intervals and costs.

## How to read a clip

LeWM predicts latents and has no image decoder, so imagination is shown two ways.

- **Left image:** the MuJoCo render of the stored simulator state, one video frame per 0.008 s environment step, played at 0.2x speed.
- **Middle image:** the nearest real frame, by L2 distance in the 192-d latent, to the encoded real frame at the last block endpoint. It shows what retrieval looks like when the latent is right.
- **Right image:** the nearest real frame to the latent the model imagined for that endpoint. The gallery is 24,147 frames from 164 probe-split episodes, disjoint from the test roots.
- **Traces:** torso height and pitch. Black is the truth at every step, blue dots are the probe on real endpoint frames, red squares are the probe on imagined latents. Green bands are the healthy range.
- **Clearance panel:** the health rule's scaled clearance. Below zero is unsafe. The model accepts a tape when its imagined minimum stays at or above the margin.

The model sees three real frames (0.16 s of history) and the action tape, then imagines ten blocks of 0.08 s. Images update once per block because imagination exists only at block endpoints.

## Clips

### Test bank (512 saved rows, margin +0.00)

Outcomes in the saved rows: 279 safe and accepted, 226 unsafe but accepted, 4 unsafe and rejected, 3 safe but rejected.

| Clip | Outcome | Count | Branch | True min | Imagined min | First violation (true / imagined block) |
|---|---|---|---|---|---|---|
| [`test_safe_accepted`](test_safe_accepted.mp4) ([gif](test_safe_accepted.gif), [poster](test_safe_accepted.png)) | Truly safe, and the model accepts | 279 | 47 of root `test-e89-t671` | +0.61 | +0.66 | - / - |
| [`test_fall_missed`](test_fall_missed.mp4) ([gif](test_fall_missed.gif), [poster](test_fall_missed.png)) | Truly unsafe, but the model accepts (false safe) | 226 | 155 of root `test-e93-t182` | -1.92 | +0.66 | 5 / - |
| [`test_fall_caught`](test_fall_caught.mp4) ([gif](test_fall_caught.gif), [poster](test_fall_caught.png)) | Truly unsafe, and the model rejects | 4 | 335 of root `test-e273-t479` | -0.67 | -0.05 | 8 / 8 |
| [`test_false_alarm`](test_false_alarm.mp4) ([gif](test_false_alarm.gif), [poster](test_false_alarm.png)) | Truly safe, but the model rejects | 3 | 437 of root `test-e265-t142` | +0.18 | -0.01 | - / 1 |

Of the 4 rejected unsafe rows, 3 have the imagined violation within 2 blocks of the true one; the clip is drawn from those (0 of them end with the torso below the healthy height).

### Stress bank (512 saved rows, margin +0.00)

Outcomes in the saved rows: 136 safe and accepted, 360 unsafe but accepted, 15 unsafe and rejected, 1 safe but rejected.

| Clip | Outcome | Count | Branch | True min | Imagined min | First violation (true / imagined block) |
|---|---|---|---|---|---|---|
| [`stress_safe_accepted`](stress_safe_accepted.mp4) ([gif](stress_safe_accepted.gif), [poster](stress_safe_accepted.png)) | Truly safe, and the model accepts | 136 | 82 of root `stress-e293-t258` | +0.58 | +0.50 | - / - |
| [`stress_fall_missed`](stress_fall_missed.mp4) ([gif](stress_fall_missed.gif), [poster](stress_fall_missed.png)) | Truly unsafe, but the model accepts (false safe) | 360 | 95 of root `stress-e21-t300` | -2.00 | +0.63 | 7 / - |
| [`stress_fall_caught`](stress_fall_caught.mp4) ([gif](stress_fall_caught.gif), [poster](stress_fall_caught.png)) | Truly unsafe, and the model rejects | 15 | 487 of root `stress-e121-t939` | -2.16 | -0.54 | 9 / 9 |
| [`stress_false_alarm`](stress_false_alarm.mp4) ([gif](stress_false_alarm.gif), [poster](stress_false_alarm.png)) | Truly safe, but the model rejects | 1 | 364 of root `stress-e85-t529` | +0.27 | -1.43 | - / 8 |

Of the 15 rejected unsafe rows, 14 have the imagined violation within 2 blocks of the true one; the clip is drawn from those (13 of them end with the torso below the healthy height).

### Long horizon

[`long_horizon`](long_horizon.mp4) ([gif](long_horizon.gif), [poster](long_horizon.png)): test-role episode 13 of `roots.h5` from step 100, imagined 30 blocks (2.4 s) ahead with its recorded actions. Height error is 0.014 m at 0.8 s and 0.013 m at 2.4 s; pitch error is 0.007 and 0.028 rad. Everything past 0.8 s lies outside the horizon evaluated in S3 and S4 and is shaded grey.

Selection: among test-role episodes in roots.h5 with at least 420 steps, the one with the median forward displacement over the window; root at step 100 (78 eligible episodes; this one moves 4.95 m forward in the window). It is ordinary walking under the policy that produced the training data, which is the easy case.

## How examples were chosen

From the saved S4 no-update rows only, by a fixed rule, never by eye:

1. Keep the rows of the outcome category at the saved matched margin.
2. For rejected unsafe rows, prefer those whose imagined violation arrives within 2 blocks of the true one. A rejection that fires many blocks early from a borderline root reading is a correct decision, not an imagined fall.
3. For the two unsafe outcomes, prefer rows where the torso really dropped below the healthy height, so the clip shows a fall and not a lean.
4. Take the row with the median true clearance, ties broken by branch index.

## Caveats

- **Speed is not shown.** S3 qualified the health rule only; the speed readout from real frames did not track the truth well enough.
- **Missed falls are the common failure.** At this margin the model accepts almost every tape, so most unsafe tapes are accepted. The rejected examples are rare and should not be read as typical.
- **Retrieval is an illustration of a latent, not a decoded image.** The right-hand image is a real frame from another episode that happens to lie nearest in latent space. The floor pattern and exact limb pose can differ.
- **Probe readings of fallen poses are unreliable.** The blue dots scatter once the robot is on the ground, which is outside the range where the probe was accurate.
- **Stored frames lag the stored state by one physics substep.** The bank rendered each endpoint straight after a simulation step, when MuJoCo's kinematics are 0.002 s behind the state. Every model input here uses those stored frames, so the numbers equal the saved S4 rows. The displayed frames are re-rendered from the stored states and differ from the stored endpoint frames in at most 1.588% of pixels.
- MP4 is lossy and GIF is resized and palette encoded. Initial and final holds repeat a frame; they are not new observations.

Full provenance, hashes and per-block numbers are in [`animations.json`](animations.json).
