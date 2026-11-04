# LeWM Experiment 1

This is the working area for the new LeWM experiment. The Vast compute pipeline
is ready; the experiment's code, protocol and configuration still need to be
specified. Existing research designs and results remain in `SafeLeWM/`.

## Start here

Read the root [agent instructions](../AGENTS.md) and the
[compute guide](computeConfigs/README.md). Use the existing controller for normal
operations. From the repository root:

```bash
python3 LeWM_Experiment1/computeConfigs/compute.py --help
python3 LeWM_Experiment1/computeConfigs/compute.py status
```

`status` reports the locally recorded session, including a destroyed session.
On a fresh checkout without local state, it reports no managed rental.

For an authorized session, choose limits before renting. These are illustrative
values, not a new allocation:

```bash
cd LeWM_Experiment1/computeConfigs
python3 compute.py search --hours 0.5 --max-hourly 1 --download-gb 0 --upload-gb 0.01 --budget-usd 1
python3 compute.py up --hours 0.5 --max-hourly 1 --download-gb 0 --upload-gb 0.01 --budget-usd 1
python3 compute.py ssh -- nvidia-smi
```

Use `run` with the experiment source, JSON config, prospective protocol, seeds and
runtime limit, then `logs`, `status` and `sync`. Use `recover` after an uncertain
launch; preserve ignored `computeConfigs/.state/` so the controller retains the
instance identity. `stop` ends a job; `destroy` removes the rental only after
idle-job and backup checks. Rental shutdown requires the user's direction.

## Environment and outputs

The [Vast template](computeConfigs/config/vast-template.json) already supplies
workspace, CPU-thread, headless-rendering and HF-cache settings. The controller
loads credentials from the root `.env` or process environment, with existing CLI
credentials as documented in the compute guide. An authenticated `gh` login with
package pull access can supply the image credential; a separate `GHCR_READ_TOKEN`
is optional. The controller passes `HF_TOKEN` to the upload worker over SSH stdin.
It does not copy `.env` or forward every variable to the experiment subprocess.
W&B integration and private checkpoint downloads inside the job are not wired up.

Write outputs beneath `LEWM_RUN_DIR`. Completed artifacts upload to the private
dataset [Sachioster/lewm-experiment1](https://huggingface.co/datasets/Sachioster/lewm-experiment1),
under `runs/<run-id>/`. Use the `run_io` helpers described in the compute guide for
periodic uploads and immutable checkpoints. A final download verifies stored bytes
before destruction is allowed.

## Layout and validation

- `computeConfigs/compute.py`: everyday command.
- `computeConfigs/config/`: template, selection policy and storage destination.
- `computeConfigs/image/`: Docker build, dependency lock and runtime validation.
- `computeConfigs/scripts/`: instance, runner and upload implementation.
- `tests/compute/`: local orchestration tests using fake providers and processes.

The first startup check took 182.37 seconds to validated readiness on an RTX 4090.
It executed zero benchmark episodes and the test instance was destroyed. See the
compute guide for the timing breakdown, tested scope and dependency versions.
