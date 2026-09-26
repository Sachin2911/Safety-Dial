# Results of the main study (running log)

Each experiment folder holds small committed results with a manifest naming the Hugging
Face revisions used. Model checkpoints and banks live in the private Hugging Face
repositories under the `Sachioster` namespace (see `infrastructure.md`).

| Folder | Experiment | Status (26 September 2026) |
|---|---|---|
| `e0/` | Assets, replay determinism, geometry, timing | **Passed.** 149 branches from 50 roots (15 mid-contact) replay bitwise identically through contact; substep mirror exact; endpoint interpolation errs by up to 52 px / 14 deg in contact |
| `e1/` | Probes, banks, four-source decomposition | **Gate passed.** Test bank (3,958 branch-layout rows, 31% unsafe): at m=0 imagined FSA 0.204 [0.16, 0.25], real readout 0.076, endpoint 0.009; of 591 imagined false-safes, 427 are attributed to imagination, 156 to readout, 8 to temporal sampling. Privileged coordinate MLP is worse (0.279). Imagined pose error grows 12 to 32 px over the horizon; rotation in contact is over-predicted 1.5x |
| `e2/` | Repairability | **Gate not met.** Recipe chosen on dev (predictor-only, teacher-forced, lr 1e-4, 3,000 steps; 16-point grid). Test bank: FSA at matched acceptance 0.204 (no update) to 0.192 (adapted), paired root-bootstrap difference -0.012 [-0.028, +0.004]; p95 optimistic clearance error unchanged (44 px). Retention improved (latent MSE 0.0087 to 0.0072). Readout-only correction worse (0.249). Fixed 23 px margin reaches 0.149 at acceptance 0.49. BatchNorm running statistics must stay frozen during adaptation (found and fixed during the run) |
| `e3/`, `e4/` | Acquisition and transfer | One-seed run in progress as the bounded check of whether more or boundary-targeted experience changes the E2 picture |
| `e5/` | Closed loop | Script ready (`pusht_e5_closedloop.py`), run only if E3 holds |
| `s0/` | Walker rules, observability, policies | Observability passed at frameskip 10 (speed R2 0.99); PPO and PPO-Lagrangian training in OmniSafe |
| `s1/` to `s4/` | Walker data, LeWM-A, gate, study | Queued after the policies finish (`runs/logs/walker_chain.sh`); S2 smoke run passed |
| `s5/` | Pretraining vs adaptation | Set-B builder ready (`walker_s5_setb.py`) |

Entry points live in `experiments/scripts/pusht_e*.py` and `walker_s*.py`; helpers in
`experiments/helpers/` (new files only; historical files are untouched). Logs of the runs
on the instance are under `runs/logs/` (not committed).
