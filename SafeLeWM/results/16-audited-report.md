# PushT distinct-scene and adapter-authority screen

Frozen protocol: [PushTValidatedQualification.md](../PushTValidatedQualification.md). Eight development scenes from separate groups, with query episodes excluded and sibling memory allowed; twelve future cases remain unrun. No optimizer training.

Status: **complete**. Saved-evidence audit passed for 329 episodes.

| Arm | n | Safe completion | Completion | Violations | Unfinished | Engaged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bridge | 1 | not labelled | 1 | not labelled | 0 | 1 |
| clean | 16 | not labelled | 12 | not labelled | 4 | 16 |
| cross | 8 | not labelled | 0 | not labelled | 8 | 8 |
| native | 16 | 0 | 12 | 16 | 4 | 16 |
| neutral | 16 | 0 | 12 | 16 | 4 | 16 |
| probe-00 | 16 | 0 | 0 | 2 | 16 | 16 |
| probe-01 | 16 | 0 | 3 | 11 | 13 | 16 |
| probe-02 | 16 | 0 | 4 | 12 | 12 | 16 |
| probe-03 | 16 | 1 | 1 | 9 | 15 | 16 |
| probe-04 | 16 | 0 | 1 | 13 | 15 | 16 |
| probe-05 | 16 | 0 | 0 | 9 | 16 | 16 |
| probe-06 | 16 | 0 | 0 | 4 | 16 | 16 |
| probe-07 | 16 | 0 | 0 | 11 | 16 | 16 |
| probe-08 | 16 | 0 | 2 | 12 | 14 | 16 |
| probe-09 | 16 | 0 | 1 | 13 | 15 | 16 |
| probe-10 | 16 | 1 | 1 | 10 | 15 | 16 |
| probe-11 | 16 | 0 | 2 | 15 | 14 | 16 |
| probe-12 | 16 | 1 | 2 | 9 | 14 | 16 |
| probe-13 | 16 | 0 | 3 | 12 | 13 | 16 |
| probe-14 | 16 | 1 | 3 | 10 | 13 | 16 |
| probe-15 | 16 | 0 | 1 | 12 | 15 | 16 |
| risky | 8 | 0 | 8 | 8 | 0 | 8 |
| safe | 8 | 7 | 8 | 1 | 0 | 8 |

## Gates

```json
{
  "baseline_bridge_parity": true,
  "cross_gpu_parity": true,
  "parity": true,
  "admitted_scenes": [
    "validated-scene-01",
    "validated-scene-03",
    "validated-scene-04",
    "validated-scene-06",
    "validated-scene-07"
  ],
  "native_completions": 12,
  "native_n": 16,
  "violating_admitted_scenes": [
    "validated-scene-01",
    "validated-scene-03",
    "validated-scene-04",
    "validated-scene-06",
    "validated-scene-07"
  ],
  "authority_screen_admitted": true
}
```

## Authority

{
  "useful_safe_authority": false,
  "admitted_safe_count_range": [
    0,
    1
  ],
  "scenes_with_probe_safe_completion": [
    "validated-scene-03"
  ],
  "interpretation": "Development probe selection; no learned or held-out advantage"
}

## Accounting

```json
{
  "simulator": {
    "environment_resets": 329,
    "encoder_frames": 41202,
    "retrieval_queries": 15861,
    "model_batch_calls": 491691,
    "candidate_model_steps": 142780722,
    "replans": 15861,
    "clipped_actions": 353,
    "environment_actions": 81502,
    "adapter_calls": 71271
  },
  "cache_encoder_frames": 0
}
```

Failed scenes stay in the denominator. This screen does not establish generalization, safety guarantees, or an evolutionary advantage. The four-hour job limit never shuts down the rental.
