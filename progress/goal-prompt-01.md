Work through progress/01-replication.md in order, following AGENTS.md. The aim is to
reproduce the ManiGuard paper's Jar in-distribution result (21/4/27/26 over 78
episodes) using the official benchmark runner on this instance, with the paper's
256-pixel legacy rendering, not the 640 high-fidelity default. Pin every source
revision before installing and record it. Save a video of every rollout, and at the
end upload all videos with their results rows to a private Hugging Face dataset laid
out by seed and task with a README index, so each episode can be watched. Use the HF_TOKEN and GITHUB_TOKEN already
in the environment for downloads and pushes, and never print them. Run long steps in
tmux. After each checklist item, write its receipt under progress/receipts/, tick the
box in 01-replication.md, commit with the prefix "todo1:" and push to main. Obey the
stop rule after seed one: if there are 0 or 1 safe successes, stop and wait for
review instead of running more seeds. When all eight items are done, push and stop.
Do not start progress/02-after-replication.md.
