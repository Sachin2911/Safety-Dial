# Todo 2, item 1: fetch and index

Host: RunPod pod `f1o65q3iygv372` (A40), 23 October 2026, about 11:50 UTC. Image
`python3` 3.11.10 with `huggingface_hub`, `opencv-python-headless`, `numpy`, `pandas`,
`matplotlib` installed by `pip`.

## Panels

`progress/panels.json` was written first, from the panel section of
`progress/02-after-replication.md`, before any episode was looked at. Development 0, 3,
6, 9, 15, 17; held-out 2, 5, 8, 14, 16, 23, 7, 18, 12; pool 1, 4, 10, 11, 13, 19, 20, 21,
22, 24, 25. The three lists partition tasks 0 to 25. The file is frozen.

## Download

`python3 -I scripts/todo2/fetch_episodes.py --dest /workspace/episodes` (tmux `fetch`,
log `/workspace/fetch.log`), `snapshot_download` of
`Sachioster/maniguard-jar-replication-20261022` at revision
`9300cebfb64fe6f9baedd51a973dcbd067a56490` restricted to `jar/seed0_run2`,
`jar/seed0_run3`, `jar/seed1`, `jar/seed2`: `index.json` plus `row.json`,
`base_ltl.json`, `base_main.mp4`, `base_wrist.mp4` per task folder. 420 files in 41 s,
1.0 GB on disk (0.78 GB of video). The token came from `HF_TOKEN` in the environment
(the tmux shell sources `/etc/rp_environment`).

## Checks (all passed, exit 0)

| check | result |
|---|---|
| index entries per run | 26 in each of the 4 runs |
| `row.json` | 104 |
| `base_ltl.json` | 104; every log has steps + 1 entries, last entry step = steps |
| videos | 208, all 256 x 256 |
| main video frames = steps + 1 | true for all 104 |
| index `steps` = row `steps` | true for all 104 |

Outcome counts from the index (match the replication note):

| run | safe success | unsafe success | safe unsuccess | unsafe unsuccess | steps min to max |
|---|---|---|---|---|---|
| seed0_run2 | 7 | 0 | 9 | 10 | 584 to 2000 |
| seed0_run3 | 7 | 5 | 7 | 7 | 684 to 2000 |
| seed1 | 6 | 0 | 10 | 10 | 778 to 2000 |
| seed2 | 4 | 2 | 11 (one `not_engaged`) | 9 | 503 to 2000 |

Sidecar format: `formula`, three `constraints` (`close_before_lift`, `jar_not_dropped`,
`jar_upright`), `violated`, `violation_step`, `violation_count`,
`total_steps_monitored`, and `log`, one entry per step with the four atomic
propositions, automaton `state`, `accepting` and `doomed`.
