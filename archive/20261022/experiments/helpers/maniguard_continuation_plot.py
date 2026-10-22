"""Plot factual gripper inputs and measured joints from a verified screen audit."""

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

MODES = ("native_only", "closure_only", "clock_handoff", "privileged_lid_handoff")
LABELS = ("Native", "Closure", "Clock handoff", "Privileged handoff")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plot_screen(run, audit_path, output):
    """Consume an existing full physical audit, checking its immutable trace bindings."""
    run, audit_path, output = Path(run), Path(audit_path), Path(output)
    audit = json.loads(audit_path.read_text())
    state_path = run / "screen.json"
    state = json.loads(state_path.read_text())
    if (state.get("screen_complete") is not True or len(state["attempts"]) != 4
            or any(a["status"] != "completed" for a in state["attempts"])
            or audit["screen_ledger_sha256"] != digest(state_path)
            or audit["protocol_sha256"] != state["protocol_sha256"]
            or len(audit["episodes"]) != 4
            or {r["mode"] for r in audit["episodes"]} != set(MODES)):
        raise ValueError("Use a complete four-start screen and its already verified audit")
    if output.exists():
        raise FileExistsError("Preserve earlier figure artifacts; select a fresh directory")
    reports = {r["mode"]: r for r in audit["episodes"]}
    streams = {}
    for mode in MODES:
        report = reports[mode]
        trace = run / "episodes" / report["attempt"] / "trace.jsonl"
        if digest(trace) != report["trace_sha256"]:
            raise ValueError("The plotted trace differs from its full physical audit")
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        commands = [r for r in rows if r["type"] == "executed"]
        queries = [r for r in rows if r["type"] == "latent_features"]
        steps = report["commands"]["whole_episode"]["controls"]
        if ([r["step"] for r in commands] != list(range(steps))
                or [r["step"] for r in queries] != list(range(0, steps, 8))
                or any(r["action"][7] not in (-1, 1) for r in commands)):
            raise ValueError("Require every actual gripper input and query measurement")
        positions = np.asarray([r["frame"]["fingers"] for r in queries], dtype=float)
        if positions.shape != (len(queries), 2) or not np.isfinite(positions).all():
            raise ValueError("Require two finite measured joint positions per query")
        streams[mode] = (commands, queries, positions)
    output.mkdir(parents=True)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(4, 2, figsize=(7, 4.7), sharex=True, layout="constrained")
    low = min(0.0, min(v[2].min() for v in streams.values())) - .002
    high = max(.04, max(v[2].max() for v in streams.values())) + .002
    for index, (mode, label) in enumerate(zip(MODES, LABELS)):
        commands, queries, positions = streams[mode]
        left, right = axes[index]
        left.step([r["step"] for r in commands], [r["action"][7] for r in commands],
                  where="post", color="#176A92", linewidth=.7)
        left.set(yticks=[-1, 1], yticklabels=["Close", "Open"], ylim=(-1.35, 1.35))
        left.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=28)
        times = [r["step"] for r in queries]
        right.fill_between(times, positions.min(axis=1), positions.max(axis=1),
                           color="#176A92", alpha=.2, linewidth=0)
        right.plot(times, positions.mean(axis=1), ".-", color="#176A92",
                   markersize=1.4, linewidth=.7)
        right.set(ylim=(low, high), yticks=[0, .02, .04])
        switch = reports[mode]["switch_query_step"]
        for axis in (left, right):
            axis.set(xlim=(0, 2000))
            axis.grid(axis="x", color=".9", linewidth=.5)
            axis.spines[["top", "right"]].set_visible(False)
            if switch is not None:
                axis.axvspan(switch, 2000, color="#946239", alpha=.08, linewidth=0)
                axis.axvline(switch, color="#946239", linestyle="--", linewidth=.8)
        if switch is not None:
            left.text(switch + 25, -.95, str(switch), color="#754620", fontsize=7)
    axes[0, 0].set_title("Executed gripper input")
    axes[0, 1].set_title("Measured finger joints (m), at queries")
    axes[-1, 0].set_xlabel("Executed control index")
    axes[-1, 1].set_xlabel("Observation control index")
    pdf, png = output / "gripper-trajectories.pdf", output / "gripper-trajectories.png"
    figure.savefig(pdf, metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(png, dpi=220)
    plt.close(figure)
    result = {
        "schema": "maniguard-continuation-gripper-figure-v1",
        "audit_sha256": digest(audit_path), "screen_ledger_sha256": digest(state_path),
        "trace_sha256": {mode: reports[mode]["trace_sha256"] for mode in MODES},
        "source_sha256": digest(__file__),
        "versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "files": {p.name: digest(p) for p in (pdf, png)},
        "new_physical_controls": 0, "new_policy_queries": 0, "new_encoder_calls": 0,
        "scope": "Four individual seen-task/seed development trials. Inputs do not prove "
                 "achieved closure. Points are two-joint means at query instants; bands "
                 "span the two joints, not uncertainty or fingertip aperture. Dashed lines "
                 "mark actual switches; shaded regions show native continuation. "
                 "At-switch position samples precede the first native action. No causal, "
                 "evolutionary or fresh-generalization effect is inferred by plotting.",
    }
    (output / "provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
