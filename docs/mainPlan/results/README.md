# Results of the main study (running log)

Each experiment folder holds small committed results with a manifest naming the Hugging
Face revisions used. Model checkpoints and banks live in the private Hugging Face
repositories under the `Sachioster` namespace (see `infrastructure.md`).

| Folder | Experiment | Status (26 September 2026) |
|---|---|---|
| `e0/` | Assets, replay determinism, geometry, timing | **Passed.** 149 branches from 50 roots (15 mid-contact) replay bitwise identically through contact; substep mirror exact; endpoint interpolation errs by up to 52 px / 14 deg in contact |
| `e1/` | Probes, banks, four-source decomposition | Probes trained (block pose MLP: centre p50 4.1 px, angle p50 1.6 deg on held-out episodes); dev/test/stress banks built with frozen layouts; decomposition running |
| `e2/` | Repairability | Queued after E1 (`runs/logs/pusht_chain.sh`) |
| `e3/`, `e4/` | Acquisition and transfer | Queued after E2 |
| `e5/` | Closed loop | Script ready (`pusht_e5_closedloop.py`), run only if E3 holds |
| `s0/` | Walker rules, observability, policies | Observability passed at frameskip 10 (speed R2 0.99); PPO and PPO-Lagrangian training in OmniSafe |
| `s1/` to `s4/` | Walker data, LeWM-A, gate, study | Queued after the policies finish (`runs/logs/walker_chain.sh`); S2 smoke run passed |
| `s5/` | Pretraining vs adaptation | Set-B builder ready (`walker_s5_setb.py`) |

Entry points live in `experiments/scripts/pusht_e*.py` and `walker_s*.py`; helpers in
`experiments/helpers/` (new files only; historical files are untouched). Logs of the runs
on the instance are under `runs/logs/` (not committed).
