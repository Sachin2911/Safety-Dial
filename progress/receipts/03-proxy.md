# Todo 3, item 2: the supervisor proxy checked on the host

Instance I, 23 October 2026, times UTC. `scripts/todo3/supervisor_proxy.py` as committed in
`4dcd75f`; **no default changed** (patience 10 queries, stall 0.01 rad, warmup 10, 1 release
chunk, 3 move chunks, at most 3 retreats, execute horizon 8).

## Local test with the openpi venv's python

`third_party/openpi/.venv/bin/python -I scripts/todo3/test_proxy_local.py` (fake upstream, no
GPU, websockets 15.0.1, msgpack 1.1.0, numpy 1.26.4):

```
off mode: 60 replies byte-identical to direct (metadata too)
retry mode: first retreat at query 22; 12 replaced replies; release chunk holds pose with gripper +1; move reaches home; pass-through resumes
log: episode line with retreats at queries [22, 35, 48] (cap 3), 60 queries
seed change inside a connection starts a fresh episode in the log
ALL CHECKS PASSED
```

One change to the test file only: its `websockets.sync.client.connect(..., legacy=True)` calls
raised `TypeError: create_connection() got an unexpected keyword argument 'legacy'` under
websockets 15.0.1 (the venv's pinned version, `03-env-openpi-freeze.txt`); the keyword was
removed. The proxy itself is untouched.

## Policy servers and warm-up

`GPU=0 PORT=8000` and `GPU=1 PORT=8001 bash scripts/todo1b/serve_jar_lane.sh` (unchanged), tmux
`serve0`/`serve1`, started 19:43:45; checkpoint restored in 7.5 s on each; both advertise
`serve_config: pi05-base_datagen_v1_jar_joint_2cam_lora` and the checkpoint path.
`scripts/todo1/probe_server.py` on each port (tmux `probe0`/`probe1`, 19:45:50 to about 19:47):
the compile query dropped after 42 to 43 s (keepalive, as on B, C and D), the reconnect answered
in 0.2 s, three probe queries at 196 to 208 ms returned `(16, 8)` float64 chunks. **5 model
queries per server before any episode**, 8,492 MiB resident per GPU after warm-up.

## Proxies (tmux `proxy-<mode>-<port>`, started 19:46:01)

| listen | mode | upstream | JSONL log |
|---|---|---|---|
| 9000 | off | 8000 | `/workspace/logs/proxy-off-9000.jsonl` |
| 9001 | off | 8001 | `/workspace/logs/proxy-off-9001.jsonl` |
| 9100 | retry | 8000 | `/workspace/logs/proxy-retry-9100.jsonl` |
| 9101 | retry | 8001 | `/workspace/logs/proxy-retry-9101.jsonl` |

Each log opens with a `listening` line carrying the parameters above. The checklist asks for
9000 off here; the other three were started at the same time so that items 4 and 6 use proxies
that have been idle since start. A metadata-only connection (no model query) through 9000
received the same 124 bytes the server sends directly; the proxy logged one `disconnect`.

## Tooling added for the later items (no evaluator or benchmark change)

`scripts/todo3/make_plan.py` (item 5 plan), `run_episode_lane.sh` (one episode on a lane with a
chosen port, optional todo 2 recorder; the evaluator arguments of `scripts/todo1b/run_smoke_lane.sh`),
`run_gate_lane.py` (item 6 lane runner over the frozen plan), `score_gate.py` (item 7),
`smoke_frames.py` (item 4 frame sheets).
