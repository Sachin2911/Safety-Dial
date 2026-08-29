# SafetyDial: Evolutionary Multi-Objective Planning in JEPA

> **Note:** There is some project info

## Setup

```bash
git clone https://github.com/Sachin2911/Safety-Dial.git
cd Safety-Dial
bash scripts/setup.sh
```

To download the data sources

```bash
uv run python scripts/download_data.py                 # all: source + Push-T + Cube
uv run python scripts/download_data.py --config-name pusht
uv run python scripts/download_data.py --config-name cube
uv run python scripts/download_data.py weights_only=true   # skip ~60 GB datasets
uv run python scripts/download_data.py clone_source=false  # skip third_party/le-wm
```


# Documents Produced
> **Note:** All documentation can be found in the `docs` subfolder.
- [x] Ideation Doc
- [x] Annotated Bibliography
- [x] Literature Review
- [x] Research Proposal


# Project Trajectory
## Safe AI via Evolutionary Algorithms

What can we learn from evolution to design AI agents that are more aligned to human values?

This project explores the potential of black-box optimization techniques such as genetic algorithms, evolutionary strategies and neuroevolution to learn safe reinforcement learning (RL) policies without relying on hand-crafted reward penalties and cost functions.

Traditional RL approaches often require manually tuning safety penalties in the reward function, which can lead to unsafe behavior if penalties are too weak or overly conservative policies if they are too strong. Alternatively, constraint-based methods require carefully designed cost functions, which can be just as challenging to specify.

In this project, we investigate whether evolutionary algorithms can effectively balance maximizing rewards while minimizing the probability of safety violations, enabling agents to learn safe behaviors autonomously. The results could contribute to developing safer AI systems in real-world applications, such as robotic navigation in human environments and ethical Large Language Models.

TLDR: Can we replace the hand tuned penalties and cost funcs used in Safe RL with evolutionary algorithms. Such that we let the safe behavior emerge from selection pressure rather than manual engineering.