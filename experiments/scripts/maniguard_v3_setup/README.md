# Prepare the fresh ManiGuard v3 rental

These scripts install the pinned public simulator and policy environments and
hash-verify the public model, benchmark and assets. They do not instantiate a
simulator scene, start a policy server, issue model queries or launch an experiment.
The independently installed rental guards retain their original absolute deadline.

Run `prepare-only.sh` only on the newly authorized rented host, from
`/workspace/setup-maniguard-v3-20261030/`. The future goal prompt supplies separate
implementation, software-check, source-freeze and experiment-launch instructions.
The root project environment is not synchronized into either rented-host environment.
