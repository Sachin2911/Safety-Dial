# Archived research guidance

This is a preserved historical workspace. The current research direction is in
[oracle.md](../../oracle.md), and current engineering guidance is in
[the root AGENTS.md](../../AGENTS.md). Archived plans do not activate new experiments.

## Engineering

Keep original code, notebooks, configs and recorded results unchanged. They are covered
by `../manifest.json`. Put new implementation in the root `experiments/` workspace.

Run historical Python commands from the repository root through
`bash archive/run.sh python ...`; this selects the shared root environment and correct
legacy working directory. Do not create a nested virtual environment or launch old
managed jobs. The original environment specification remains here for provenance.

The full original guidance, including study-specific safety and replay controls, is in
[provenance/AGENTS.md.txt](provenance/AGENTS.md.txt). Its old current-direction sections are
historical records. Consult [the archive index](../README.md) before reusing results.
