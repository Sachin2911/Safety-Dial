"""Render already-audited result figures on the idle L4 rental."""

import hashlib
import json
import os
from pathlib import Path
import socket

os.environ.setdefault("MPLCONFIGDIR", "/tmp/safelewm-closeout-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/closeout-ppo-ap-20261102"


def main():
    assert socket.gethostname() == "8cf3a706cdd1"
    ppo = json.loads((OUT / "ppo.json").read_text())
    ap = json.loads((OUT / "ap.json").read_text())
    assert ap["passed"] and ap["episodes"] == 256
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})
    colors = ["#7c8890", "#246b8b", "#c17a32", "#6f5b97"]
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    x = np.arange(3)
    for index, (arm, label) in enumerate([
        ("untrained", "Untrained"), ("main", "CMA-ES"),
        ("random", "Random search"), ("ppo", "PPO"),
    ]):
        groups = [ppo["groups"][f"{host}-{repeat}-{arm}"]
                  for host, repeat in [("4090", "R0"), ("4090", "R1"), ("l4", "R2")]]
        values = [100 * g["safe_completion"] / g["n"] for g in groups]
        pos = x + (index - 1.5) * 0.18
        ax.bar(pos, values, width=0.17, color=colors[index], label=label)
        for at, value, group in zip(pos, values, groups, strict=True):
            ax.text(at, value + 0.4, str(group["safe_completion"]), ha="center", fontsize=9)
    ax.axhline(100 * 7 / 72, color="#333333", linestyle="--", linewidth=1,
               label="Native (7/72 on each host)")
    ax.set_xticks(x, ["R0: RTX 4090", "R1: RTX 4090", "R2: L4"])
    ax.set_ylim(0, 30)
    ax.set_ylabel("Safe completion (%)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
    ax.set_title("Matched budget: 40,000 training actions per optimizer and repeat")
    for suffix in ["pdf", "png"]:
        fig.savefig(OUT / f"14-matched-ppo.{suffix}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.2, 2.5))
    names = ["Published AP-CEM\nexternal reference", "Authors' evaluator\nour RTX 4090 run",
             "Our wrapper\nmatched inputs/runtime"]
    counts = [89, ap["summary"]["authors"]["completed"], ap["summary"]["wrapper"]["completed"]]
    ax.bar(np.arange(3), np.asarray(counts) * 100 / 128,
           color=["#aab5bd", "#246b8b", "#4d8d71"], width=0.6)
    for i, count in enumerate(counts):
        ax.text(i, 100 * count / 128 + 2, f"{count}/128", ha="center")
    ax.set_xticks(np.arange(3), names)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Task completion (%)")
    ax.set_title("PushT: 128 fixed queries, 16,688 memory episodes, no hazards")
    for suffix in ["pdf", "png"]:
        fig.savefig(OUT / f"15-pusht-ap-reproduction.{suffix}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    provenance = dict(
        ppo_sha256=hashlib.sha256((OUT / "ppo.json").read_bytes()).hexdigest(),
        ap_sha256=hashlib.sha256((OUT / "ap.json").read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        additional_simulator_actions=0,
    )
    (OUT / "figure-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
