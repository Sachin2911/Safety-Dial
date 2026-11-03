"""Render completed, audited PushT authority evidence on its rented host."""

import hashlib
import json
import os
from pathlib import Path
import socket

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pusht-validated-closeout-mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs/pusht-validated-20261102"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert socket.gethostname() == "c45db82630e6"
    audit = json.loads((BASE / "run-v1/audit.json").read_text())
    assert audit["passed"]
    analysis = json.loads((BASE / "run-v1/analysis.json").read_text())
    out = BASE / "closeout"
    out.mkdir(exist_ok=False)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    probes = [f"probe-{i:02d}" for i in range(16)]
    keys = ["neutral", *probes]
    fig, axes = plt.subplots(2, 1, figsize=(9.3, 4.4), sharex=True)
    for ax, field, title in [
        (axes[0], "groups", "All eight development scenes"),
        (axes[1], "admitted_groups", "Five admitted scenes"),
    ]:
        groups = analysis[field]
        values = [groups[k]["safe_completion"] for k in keys]
        den = groups["neutral"]["n"]
        ax.bar(np.arange(len(keys)), values, color=["#7c8890"] + ["#246b8b"] * 16)
        for i, value in enumerate(values):
            ax.text(i, value + 0.1, str(value), ha="center", fontsize=8)
        ax.set_ylim(0, max(4, max(values) + 1.4))
        ax.set_yticks(range(0, int(ax.get_ylim()[1]) + 1))
        ax.set_ylabel(f"Safe completion / {den}")
        ax.set_title(title)
    axes[1].set_xticks(np.arange(len(keys)), ["Neutral"] + [str(i) for i in range(16)])
    axes[1].set_xlabel("Pre-drawn parameter probe (no training)")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(out / f"16-pusht-validated.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    admitted = analysis["admitted_groups"]
    best = max(probes, key=lambda k: admitted[k]["safe_completion"])
    summary = dict(
        best_probe_by_admitted_safe=best,
        best_admitted=admitted[best],
        best_all=analysis["groups"][best],
        authority=analysis["authority"],
        native_all=analysis["groups"]["native"],
        neutral_admitted=admitted["neutral"],
        resources=analysis["resources"],
        audit_episodes=audit["episodes"],
    )
    (out / "closeout-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    provenance = dict(
        analysis_sha256=sha(BASE / "run-v1/analysis.json"),
        audit_sha256=sha(BASE / "run-v1/audit.json"),
        source_sha256=sha(Path(__file__)),
        additional_simulator_actions=0,
    )
    (out / "figure-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    files = {
        p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
        for p in sorted(out.iterdir())
        if p.is_file()
    }
    (out / "manifest.json").write_text(json.dumps(dict(files=files), indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
