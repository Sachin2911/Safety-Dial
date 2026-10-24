# Task 5 execution adapter: host

Replacement instance 57544241, machine 54514 (New Zealand),
created 2026-10-24T17:37:26.591023+00:00. This is the physical machine used for the baseline
replication, but task 5 reruns its own native controls and does not pool historical
outcomes. Two RTX 3090 GPUs, each 24,576 MiB, and driver 580.173.02 were verified
through SSH. The offer allocates 20 effective CPU cores and about 128 GB RAM.
The 120 GB filesystem had 117 GB free immediately after bootstrap.

Compute plus selected disk is $0.575556/hour. Listed download is $0.00260417/GB
and upload $0.00520833/GB. The established template is
`62414e8fca6befba085051de7ed36190`. Workstation credentials were not transferred;
setup uses public pinned sources and results return over SSH for private upload.

The first instance, 57539828 on machine 153150, never reached SSH and was
destroyed with provider absence verified. It incurred no simulator starts or model
queries. Its conservative compute/storage estimate is $0.3278 through destruction;
transfer charges are separate. See `05-adapter-failed-provisioning.json`.

The replacement retains the original allocation deadlines. Admissions close at
21:26:49 UTC; the remote watchdog was verified active with shutdown at
22:56:49 UTC. Its configured lifetime is shortened for the time already spent.
The new rate over the remaining allocation plus the failed-host estimate remains
below $4.50 in compute/storage; transfer is recorded separately.

Pinned environment preparation began 17:38:49 UTC, with `MAX_JOBS=8` to bound
compilation memory. Host render-library checks passed at 17:39:06 UTC, then public
source clones, OpenPI setup and checkpoint download began concurrently. The
engineering launcher waits for the finish marker, checks source hashes and pins,
applies the audited hook, starts both servers and runs only the two frozen
engineering jobs. Source plan `05-adapter-plan-v3.json` is authoritative for
execution. Pilot admission still requires task 4 evidence and both engineering
recordings to validate.

`sync_adapter05.py` copies all available evidence to ignored local storage during
execution. Final installation, simulation and preservation outcomes are recorded
in separate run and preservation receipts.
