#!/usr/bin/env python3
"""Post hoc, saved-only S1 alignment evidence. No simulator, model or network calls.

Fixed selection: episodes 0..3 of setA and probe, first/last 24 recorded rows.
The root/test bank is not inspected. Whole H5 bytes are streamed for SHA256, but
only these small windows (plus one adjacent row) are decoded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 -- register saved Blosc2 filters
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DT, SPEED_LIMIT = 0.008, 2.3415
VELOCITY_ATOL, VELOCITY_RTOL = 1e-10, 1e-10
EPISODES, WIDTH = tuple(range(4)), 24
FIELDS = ("qpos", "qvel", "action", "x_velocity", "cost", "healthy", "terminated",
          "truncated", "episode_idx", "step_idx")


def require(value, message):
    if not value:
        raise ValueError(message)


def identity(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_record(path):
    before = identity(path)
    digest = sha256(path)
    require(identity(path) == before, f"Source changed: {path}")
    return {"path": str(Path(path).relative_to(ROOT)), "sha256": digest, "stat_identity": before}


def healthy(qpos):
    q = np.asarray(qpos)
    return (q[..., 1] > 0.8) & (q[..., 1] < 2.0) & (q[..., 2] > -1.0) & (q[..., 2] < 1.0)


def window_rows(columns, *, episode, offset, start, stop, length, termination_on,
                previous_velocity=None):
    """Check only selected rows; an extra successor can support the final selected row."""
    n = stop - start
    require(n > 0 and len(columns["qpos"]) == n + int(stop < length), "Bad bounded window")
    require(all(np.isfinite(v).all() for v in columns.values()), "Nonfinite saved window")
    require(all(np.isin(columns[k], (0, 1)).all() for k in
                ("cost", "healthy", "terminated", "truncated")), "Nonbinary saved flags")
    require(np.all(columns["episode_idx"] == episode), "Window crosses source episodes")
    require(np.array_equal(columns["step_idx"], np.arange(start, start + len(columns["qpos"]))),
            "Noncontiguous episode step identities")
    rows = []
    for j in range(n):
        q = columns["qpos"][j]
        speed = float(columns["x_velocity"][j])
        has_next = start + j + 1 < length
        expected_speed = float((columns["qpos"][j + 1, 0] - q[0]) / DT) if has_next else None
        expected_termination = (int(not healthy(columns["qpos"][j + 1])) if has_next else None) if termination_on else 0
        prior_speed = float(columns["x_velocity"][j - 1]) if j else previous_velocity
        rows.append({
            "episode": episode, "row": offset + start + j, "step": start + j,
            "x": float(q[0]), "height": float(q[1]), "pitch": float(q[2]),
            "action": columns["action"][j].tolist(), "healthy": int(columns["healthy"][j]),
            "expected_healthy": int(healthy(q)), "x_velocity": speed,
            "frame_velocity_from_preceding_transition": prior_speed,
            "expected_x_velocity": expected_speed,
            "velocity_abs_error": abs(speed - expected_speed) if has_next else None,
            "cost": int(columns["cost"][j]), "expected_cost": int(speed > SPEED_LIMIT),
            "terminated": int(columns["terminated"][j]),
            "truncated": int(columns["truncated"][j]),
            "expected_terminated": expected_termination,
            "successor_available": has_next,
            "healthy_matches": bool(columns["healthy"][j] == healthy(q)),
            "cost_matches": bool(columns["cost"][j] == (speed > SPEED_LIMIT)),
            "velocity_matches": bool(np.isclose(speed, expected_speed, atol=VELOCITY_ATOL,
                                                  rtol=VELOCITY_RTOL)) if has_next else None,
            "termination_matches": bool(columns["terminated"][j] == expected_termination)
            if expected_termination is not None else None,
        })
    return rows


def summarize(rows):
    result = {"rows": len(rows)}
    for field in ("healthy", "cost", "velocity", "termination"):
        values = [r[f"{field}_matches"] for r in rows]
        result[field] = {"checked": sum(x is not None for x in values),
                         "mismatches": sum(x is False for x in values),
                         "unavailable": sum(x is None for x in values)}
    result.update({"healthy_rows": sum(r["healthy"] for r in rows),
                   "cost_positive_rows": sum(r["cost"] for r in rows),
                   "terminated_rows": sum(r["terminated"] for r in rows),
                   "truncated_rows": sum(r["truncated"] for r in rows),
                   "missing_episode_successors": sum(not r["successor_available"] for r in rows),
                   "maximum_velocity_abs_error": max((r["velocity_abs_error"] for r in rows
                                                       if r["velocity_abs_error"] is not None), default=None)})
    return result


def array_digest(columns):
    h = hashlib.sha256()
    for name in sorted(columns):
        a = np.ascontiguousarray(columns[name])
        h.update(json.dumps([name, str(a.dtype), list(a.shape)]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def plot_tails(windows, output, dataset):
    tails = [w for w in windows if w["dataset"] == dataset and w["window"] == "tail"]
    fig, axes = plt.subplots(3, len(tails), figsize=(15, 8), squeeze=False, sharex="col")
    for col, window in enumerate(tails):
        rows = window["rows"]
        x = [r["step"] for r in rows]
        axes[0, col].set_title(f"Episode {window['episode']}; saved tail")
        for ax, key, bounds in ((axes[0, col], "height", (0.8, 2.0)),
                                (axes[1, col], "pitch", (-1.0, 1.0))):
            ax.plot(x, [r[key] for r in rows], "o-", markersize=3, label=f"Pre-action {key}")
            for boundary in bounds:
                ax.axhline(boundary, color="grey", linestyle=":", linewidth=1)
            bad = [r for r in rows if not r["healthy"]]
            ax.scatter([r["step"] for r in bad], [r[key] for r in bad], color="#d0473b", s=22,
                       zorder=3, label="Unhealthy state")
        ax = axes[2, col]
        ax.plot(x, [r["x_velocity"] for r in rows], "o-", markersize=3, label="Transition velocity")
        available = [r for r in rows if r["successor_available"]]
        ax.plot([r["step"] for r in available], [r["expected_x_velocity"] for r in available],
                color="#e69f00", linestyle="--", label="From next saved position")
        ax.axhline(SPEED_LIMIT, color="grey", linestyle=":", label="Signed speed limit")
        bad = [r for r in rows if r["cost"]]
        ax.scatter([r["step"] for r in bad], [r["x_velocity"] for r in bad], color="#d0473b", s=22,
                   zorder=3, label="Transition cost = 1")
        ending = rows[-1]
        ax.set_xlabel("Action / saved row index within episode\n"
                      f"Last row: term={ending['terminated']}, trunc={ending['truncated']}; successor unavailable", fontsize=8)
        for a in axes[:, col]:
            a.grid(alpha=0.2)
    for ax, label in zip(axes[:, 0], ("Height (m)", "Pitch (rad)", "Velocity (m/s)"), strict=True):
        ax.set_ylabel(label)
    handles, labels = [], []
    for ax in axes[:, 0]:
        hs, ls = ax.get_legend_handles_labels()
        for h, label in zip(hs, ls, strict=True):
            if label not in labels:
                handles.append(h)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(f"Post hoc S1 alignment: {dataset}\nFixed episodes 0..3, final 24 recorded rows; illustrative, not prevalence", fontsize=14)
    fig.tight_layout(rect=(0, 0.095, 1, 0.92))
    fig.savefig(output / f"{dataset}_tail_traces.png", dpi=150)
    fig.savefig(output / f"{dataset}_tail_traces.pdf")
    plt.close(fig)


def write_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/study/walker2d/continuation-20260926-2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provenance-index", type=Path, required=True)
    args = parser.parse_args(argv)
    data, output, index = args.data_dir.resolve(), args.output_dir.resolve(), args.provenance_index.resolve()
    parent = ROOT / "docs/mainPlan/results/continuation-v2"
    require(output.parent == parent and index.parent == parent and not output.exists() and not index.exists(),
            "Require fresh continuation-v2 output directory and provenance index")
    manifest = json.loads((data / "manifest.json").read_text())
    run_id = manifest["run_id"]
    result_dir = ROOT / "docs/mainPlan/results/s1" / run_id
    small_paths = [data / name for name in ("manifest.json", "config.yaml", "splits.json", "upload_receipt.json", "hf_upload.json")]
    small_paths += [result_dir / "collection.json", result_dir / "policy_ladder.json",
                    ROOT / "docs/mainPlan/results/s0/observability.json", Path(__file__),
                    ROOT / "experiments/scripts/walker_s0_observability.py",
                    ROOT / "experiments/scripts/walker_s1_collect.py",
                    ROOT / "experiments/helpers/locoCollect.py", ROOT / "experiments/helpers/locoEnv.py"]
    sources = {str(p.relative_to(ROOT)): source_record(p) for p in small_paths}
    windows, dataset_records = [], {}
    for name, termination_on in (("setA.h5", True), ("probe.h5", False)):
        path = data / name
        record = source_record(path)
        require(record["sha256"] == manifest["data"]["sha256"][name], f"Manifest hash differs: {name}")
        sources[record["path"]] = record
        with h5py.File(path, "r") as f:
            require(str(f.attrs["run_id"]) == run_id, "H5 run identity mismatch")
            env = json.loads(f.attrs["env_manifest"])
            require(env["velocity_threshold"] == SPEED_LIMIT and env["healthy_z_range"] == [0.8, 2.0]
                    and env["healthy_angle_range"] == [-1.0, 1.0]
                    and json.loads(env["model_scalars_json"])["dt"] == DT, "Unexpected recorded rules")
            dataset_records[name] = {"n_steps": int(f.attrs["n_steps"]), "n_episodes": int(f.attrs["n_episodes"]),
                                    "render_fingerprint": str(f.attrs["render_fingerprint"]),
                                    "stored_env_terminate_when_unhealthy": env["terminate_when_unhealthy"],
                                    "collector_declared_termination_on": termination_on,
                                    "termination_mode_provenance": "Existing S1 collector mode, not inferred from copied render manifest"}
            for episode in EPISODES:
                offset, length = int(f["ep_offset"][episode]), int(f["ep_len"][episode])
                for kind, start, stop in (("head", 0, min(WIDTH, length)),
                                          ("tail", max(0, length - WIDTH), length)):
                    columns = {k: f[k][offset + start:offset + stop + int(stop < length)] for k in FIELDS}
                    previous = float(f["x_velocity"][offset + start - 1]) if start else None
                    rows = window_rows(columns, episode=episode, offset=offset, start=start, stop=stop,
                                       length=length, termination_on=termination_on, previous_velocity=previous)
                    windows.append({"dataset": name, "window": kind, "episode": episode,
                                    "episode_offset": offset, "episode_length": length,
                                    "row_range_half_open": [offset + start, offset + stop],
                                    "decoded_slice_sha256": array_digest(columns), "summary": summarize(rows), "rows": rows})
    unique = {(w["dataset"], r["row"]): r for w in windows for r in w["rows"]}
    summary = summarize(list(unique.values()))
    passed = all(summary[field]["mismatches"] == 0 for field in ("healthy", "cost", "velocity", "termination"))
    receipt = json.loads((data / "upload_receipt.json").read_text())
    collection = json.loads((result_dir / "collection.json").read_text())
    require(collection["run_id"] == run_id and collection["hf_revision"] == receipt["revision"], "S1 receipt/result mismatch")
    report = {"status": "passed" if passed else "failed", "observed_utc": datetime.now(timezone.utc).isoformat(),
              "timing": "post_hoc_saved_data_audit_after_pretraining_started", "run_id": run_id,
              "selection_policy": {"datasets": ["setA.h5", "probe.h5"], "episodes": list(EPISODES),
                                   "windows": "first and last 24 rows, plus one available successor and predecessor",
                                   "outcome_selection": False, "final_test_roots_inspected": False},
              "rules": {"dt_s": DT, "signed_speed_limit": SPEED_LIMIT,
                        "health": "0.8 < height < 2.0 and -1.0 < pitch < 1.0 (strict bounds)",
                        "cost": "transition x_velocity > 2.3415, signed, strict",
                        "velocity_tolerance": {"atol": VELOCITY_ATOL, "rtol": VELOCITY_RTOL}},
              "summary_unique_rows": summary, "datasets": dataset_records, "windows": windows,
              "source_files": sources, "recorded_private_data_receipt": receipt,
              "limitations": ["Bounded deterministic sample, not a full dataset validation or prevalence estimate.",
                              "Post hoc evidence cannot establish that the planned alignment review occurred before training.",
                              "The last state after each episode's final action was not retained; velocity and benchmark health termination cannot be independently checked there.",
                              "Probe H5 env_manifest was copied from the termination-on rendering context; collector source declares termination off. Both facts are retained explicitly.",
                              "No action-to-state physics, rewards, camera pixels or model predictions were recomputed. No simulation, model, GPU or network calls.",
                              "Original source commit is recorded dirty; current source hashes are review references, not proof of the exact historical working tree."]}
    output.mkdir()
    write_json(output / "alignment.json", report)
    flat = [{"dataset": w["dataset"], "window": w["window"], **r} for w in windows for r in w["rows"]]
    with (output / "traces.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows({**r, "action": json.dumps(r["action"])} for r in flat)
    for dataset in ("setA.h5", "probe.h5"):
        plot_tails(windows, output, dataset)
    text = ["# Post hoc Walker S1 alignment audit", "", "This checks retained data after pretraining began. It is not evidence of a pretraining-time review. No simulator or model was queried.", "",
            "Rows contain pre-action state and health. Cost, velocity and terminal flags concern the transition caused by the action at that row. State-frame velocity uses the preceding transition, unavailable at episode starts.", "",
            "Fixed selection: episodes 0, 1, 2, 3 from setA and probe; first/last 24 rows. Root/test data were not inspected. Full source-file SHA256 was streamed; only bounded slices were decoded. Selection is illustrative, not a prevalence estimate.", "",
            "| Check | Checked unique rows | Mismatches | Unavailable |", "|---|---:|---:|---:|"]
    for field in ("healthy", "cost", "velocity", "termination"):
        c = summary[field]
        text.append(f"| {field} | {c['checked']} | {c['mismatches']} | {c['unavailable']} |")
    text += ["", "The last post-action state of each episode is absent. Probe termination is disabled by the collector; its copied render-context manifest misleadingly says enabled. This audit preserves and documents that limitation.", "", "## Recorded traces", "", "All selected rows and action vectors are retained in traces.csv and alignment.json. The following last six rows per episode make the timing distinction inspectable."]
    for w in windows:
        if w["window"] != "tail":
            continue
        text += ["", f"### {w['dataset']}, episode {w['episode']}", "", "| Row | t | Height | Pitch | Healthy | Transition speed | Cost | Term | Trunc | Successor |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in w["rows"][-6:]:
            text.append(f"| {r['row']} | {r['step']} | {r['height']:.6f} | {r['pitch']:.6f} | {r['healthy']} | {r['x_velocity']:.6f} | {r['cost']} | {r['terminated']} | {r['truncated']} | {'saved' if r['successor_available'] else 'unavailable'} |")
    (output / "README.md").write_text("\n".join(text) + "\n")
    s0 = json.loads((ROOT / "docs/mainPlan/results/s0/observability.json").read_text())
    provenance = {"kind": "post_hoc_existing_evidence_index", "created_utc": report["observed_utc"],
                  "s0": {"report": "docs/mainPlan/results/s0/observability.json", "recorded_frameskips": s0["frameskips"],
                         "recorded_render_fingerprint": s0["render_fingerprint"],
                         "current_source_reference": "experiments/scripts/walker_s0_observability.py",
                         "unknown_original_provenance": ["Exact executed working-tree bytes and original commit", "CLI/seed configuration", "Episode seeds and train/validation episode identities", "Saved CNN weights and original rendered frames"]},
                  "s1": {"run_id": run_id, "original_manifest": str((data / "manifest.json").relative_to(ROOT)),
                         "recorded_repo": manifest["repo"], "recorded_packages": manifest["packages"],
                         "saved_config": str((data / "config.yaml").relative_to(ROOT)),
                         "saved_splits": str((data / "splits.json").relative_to(ROOT)),
                         "data_sha256_recorded_in_original_manifest": manifest["data"]["sha256"],
                         "data_sha256_reverified_in_this_audit": {name: sources[str((data / name).relative_to(ROOT))]["sha256"] for name in ("setA.h5", "probe.h5")},
                         "recorded_upload_receipt": receipt,
                         "unknown_original_provenance": ["Exact historical dirty working-tree source bytes are not established by its recorded commit alone"],
                         "probe_manifest_caveat": report["limitations"][3]},
                  "alignment_report": str((output / "alignment.json").relative_to(ROOT)),
                  "alignment_report_sha256": sha256(output / "alignment.json"), "source_files": sources,
                  "remote_verification": "None in this saved-only audit; the original upload receipt is indexed, not independently reverified"}
    require(all(identity(ROOT / name) == record["stat_identity"] for name, record in sources.items()), "An audit input changed")
    require(all(sha256(p) == sources[str(p.relative_to(ROOT))]["sha256"] for p in small_paths), "Small source content changed")
    write_json(index, provenance)
    write_json(output / "manifest.json", {"kind": "saved_only_post_hoc_alignment", "status": report["status"],
                "input_files": sources, "upstream_revisions": receipt, "source_inputs_unchanged": True,
                "outputs_sha256": {p.name: sha256(p) for p in sorted(output.iterdir())},
                "provenance_index": str(index.relative_to(ROOT)), "provenance_index_sha256": sha256(index)})
    print(json.dumps({"status": report["status"], "summary": summary, "output": str(output),
                      "alignment_sha256": sha256(output / "alignment.json"), "index_sha256": sha256(index)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
