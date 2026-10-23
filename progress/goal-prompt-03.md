Work through progress/03-retry-gate.md in order, following AGENTS.md. This is a Vast
template instance with two GPUs. scripts/stage2/prepare_vast_host.sh, which replays
the pinned ManiGuard install, may already be running in tmux session "prepare"
(started from the workstation): if so, wait for PREPARE DONE in
third_party/logs/prepare-*.log and do not start it again; if it is not running, run it
to PREPARE DONE. The task is one experiment:
does a retry-on-stall supervisor, built as a proxy between the evaluator and the policy
server with the exact defaults in the checklist, raise safe completion on the six
development Jar tasks without raising violations. Keep the benchmark, the checkpoint,
the eval config and the scoring untouched; the proxy in off mode must be a byte-identical
pass-through and item 3 proves it before anything else runs. Only the development
tasks in progress/panels.json get episodes; never touch a held-out task. Freeze the
plan file before the first job and never edit it. Watch the smoke video and write what
you see in plain sentences. Use HF_TOKEN, GITHUB_TOKEN and CONTAINER_API_KEY from the
environment and never print them. Run long steps in tmux. After each checklist item
write its receipt under progress/receipts/, tick the box, commit with the prefix
"todo3:", pull with rebase and push to main. Do not edit todo 1 or todo 2 receipts,
panels.json or theOdessy.tex. When item 8 is done, the last action is to stop this
instance from inside it with scripts/todo1c/stop_instance.py --receipt
progress/receipts/03-preserve.md after its guards pass; the instance is unreachable
after that, so nothing may remain only on local disk.
