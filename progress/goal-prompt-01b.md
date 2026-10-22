Work through progress/01b-lanes.md in order, following AGENTS.md. This instance has
two RTX 3090s. Replay the install recorded in progress/receipts/01-install.md and
01-sources.json at the same pinned revisions, then run seeds 1 and 2 of the Jar
replication concurrently, one seed per GPU, each with its own policy server port,
using the paper-protocol config in progress/receipts/01-jar-eval.yaml and the
official evaluator unchanged apart from port and task subset. Save a video of every
rollout and upload both seeds to the private Hugging Face dataset with the layout
jar/seed<s>/task_<nnnn>/. Use HF_TOKEN and GITHUB_TOKEN from the environment and
never print them. Run long steps in tmux. After each checklist item write its receipt
under progress/receipts/, tick the box, commit with the prefix "todo1b:", pull with
rebase and push to main. Another agent on instance B is pushing seed 0 to the same
branch; do not touch its files. Before launching the seed runs, check whether B's
seed 0 shows zero safe successes after six or more episodes, and if so stop and wait.
When item 7 is done, push and stop.
