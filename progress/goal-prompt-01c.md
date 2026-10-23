Work through progress/01c-seed0-rerun.md in order, following AGENTS.md. This instance
is the same machine that ran seeds 1 and 2 (instance C) and has two RTX 3090s. Replay
the install recorded in progress/receipts/01b-install.md and 01-sources.json at the
same pinned revisions, then run seed 0 of the Jar replication on both GPUs at once:
two independent repeats of the same seed, one per lane, each with its own policy server
port, output leaves jar_transport_joint_legacy256/seed0-rerun1 and seed0-rerun2, using
the paper-protocol config in progress/receipts/01-jar-eval.yaml and the official
evaluator unchanged apart from the port. Save a video of every rollout and upload both
repeats to the private Hugging Face dataset under jar/seed0-rerun1/ and
jar/seed0-rerun2/ with the same per-task layout as jar/seed0/. Use HF_TOKEN and
GITHUB_TOKEN from the environment and never print them. Run long steps in tmux. After
each checklist item write its receipt under progress/receipts/, tick the box, commit
with the prefix "todo1c:", pull with rebase and push to main. Do not edit earlier
receipts or checklists. Nothing beyond this checklist runs here. When item 6 is done,
push and stop.
