# Push-T dataset exploration

Source file: `/workspace/Safety-Dial/data/processed/pusht_expert_train.h5`

## Layout

- Flat timestep arrays (not one group per episode).
- `2336736` steps, `18685` episodes.
- Episode i: `ep_offset[i] : ep_offset[i] + ep_len[i]`.

## Keys

| key | shape | dtype | notes |
|-----|-------|-------|-------|
| action | (2336736, 2) | float32 | appears normalised / relative, not raw 512px targets |
| state | (2336736, 7) | float32 | see column guess below |
| proprio | (2336736, 4) | float32 | looks like pusher xy + 2 extras |
| pixels | (2336736, 224, 224, 3) | uint8 | do not load whole array |
| ep_len | (18685,) | int32 | |
| ep_offset | (18685,) | int64 | |
| episode_idx | (2336736,) | int64 | per-step episode id |
| step_idx | (2336736,) | int64 | per-step index within episode |

## State columns (heuristic)

| col | label | min | max |
|-----|-------|-----|-----|
| 0 | pusher_x | -153.0066 | 666.1378 |
| 1 | pusher_y | -90.5952 | 723.6395 |
| 2 | block_x | -32.8419 | 507.5937 |
| 3 | block_y | -1.8579 | 538.7749 |
| 4 | block_theta | 0.0000 | 6.2832 |
| 5 | extra_5 | -558.1359 | 762.6093 |
| 6 | extra_6 | -546.0454 | 669.1609 |

## Episode length

- min / mean / max: 49 / 125.1 / 246

## Arena note

Pusher/block xy are roughly arena-scale but can leave `[0, 512]` (negatives and values > 512 appear). Confirm against `gym-pusht` live.
