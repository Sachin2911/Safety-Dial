# Task 5 execution adapter: build receipt

Frozen on 2026-10-24T17:17:52.169051+00:00 before any task 5 simulator start.
Plan SHA-256: `8df22726223ad3d18545b9622b076cf57d2af136bdf298b99783e60b53504628`. The plan lists exact controller settings,
38 planned jobs, the 40-start ceiling and all critical source hashes.

The reusable controller is `experiments/helpers/execution_adapter.py`. The live
hook reads measured arm position and velocity before each 20 Hz action. Native
mode preserves command bytes and dtype; all modes preserve the gripper. Adaptive
history resets per episode. This is our explicitly specified APEX-inspired update,
not verified author code or a reproduction of the paper's guarantees.

`patch_adapter05.py` accepts only the pinned evaluator or this exact existing
patch. It inserts the adapter before native clipping, and journals warm-up and
returned controls. Original evaluator SHA-256: `0991d5fdf5f65b4d70f113fa57bbff33190f4d65d5eb32683e87b44968ccae29`.
Patched evaluator SHA-256: `47e3e98209ef919b86cefaf63274f0fca4392751296b720d26777c351f72e453`. The reversible diff is retained
under the local source audit and will be copied into remote evidence.

Validation on the workstation: `.venv/bin/python -m pytest -q experiments/tests`
passed **17 tests** (14 adapter checks and 3 existing tracking checks). Ruff passed
for the new Python files; the launch shell script passed `bash -n`. The tests cover
exact native bytes and RNG preservation, bounds, gripper, causal update timing,
reset, a declared toy plant, patch placement/reversibility, live-hook replay on
known joint states, corrupted feedback detection, task/lane balance and budget
preservation across a queue restart. These are engineering checks, not Jar results.

The scored queue requires both engineering recordings to validate and an explicit
launch decision citing task 4. It rejects source drift and the wrong rental, locks
against concurrent queues and retains start counts across restarts. An incomplete
attempt stops admission pending diagnosis. Source snapshots, journals and partial
attempts are preserved. No optimizer search is part of this allocation.
