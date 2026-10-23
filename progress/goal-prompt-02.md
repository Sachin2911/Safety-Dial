Work through progress/02-after-replication.md in order, following AGENTS.md. This is
a description task, not an experiment: the 104 Jar episodes of record are already in
the private Hugging Face dataset Sachioster/maniguard-jar-replication-20261022 at
revision 9300cebf, and your job is to say what happened in them, when, how repeatably,
and what differs between a success and a failure of the same task, then map that
onto every family of intervention listed in the checklist and rank where the evidence
says to look first. Do not run any intervention. This is a RunPod pod, not the Vast
image: items 1 to 6 need only the image's python3 plus a few pip packages; item 7
first checks that OptiX and EGL exist in the container and, only if they do, replays
the pinned ManiGuard install from progress/receipts/01b-install.md with Miniforge
under /workspace and runs exactly one episode to validate the per-step recorder. Write the panels file first and never change it. Look at every
frame sheet and pair strip you make and write what you see in plain sentences. Use
HF_TOKEN and GITHUB_TOKEN from the environment and never print them. Run long steps
in tmux. After each checklist item write its receipt under progress/receipts/, tick
the box, commit with the prefix "todo2:", pull with rebase and push to main. Do not
edit todo 1 receipts or theOdessy.tex. When item 9 is done, the last action is to
stop this pod from inside it with scripts/todo2/stop_pod.py --receipt
progress/receipts/02-preserve.md, after its guards pass; the pod is unreachable after
that, so nothing may remain only on local disk.
