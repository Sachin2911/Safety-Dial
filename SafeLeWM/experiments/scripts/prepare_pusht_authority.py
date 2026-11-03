"""Frozen scene selection and retrieval preparation on the rented host."""

import argparse
import json
from pathlib import Path
import socket
import sys
import time

import hdf5plugin  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from helpers import manip as M  # noqa: E402
from helpers.pusht_authority import distinct_start  # noqa: E402
import manip_pipeline as P  # noqa: E402

BASE = M.ROOT / "outputs/pusht-authority-20261102"
M.OUT = BASE


def select_panel():
    assert socket.gethostname() == "c45db82630e6"
    target = BASE / "pusht/panel.json"
    assert not target.exists()
    old = json.loads((M.ROOT / "configs/manip-screen/pusht-panel.json").read_text())
    excluded_groups = {
        c["episode"] // 101 for k in ["stage_a", "stage_b", "development"] for c in old[k]
    }
    candidates, rejected, selected_starts = [], [], []
    with P.open_data("pusht") as data:
        lengths, offsets = P.episode_index(data)
        starts = np.asarray(data["state"][offsets], dtype=np.float64)
        assert len(lengths) == 185 * 101
        for group in range(185):
            first = starts[group * 101]
            assert np.allclose(starts[group * 101 : (group + 1) * 101], first, atol=1e-4, rtol=0), (
                group
            )
        memory = [e for e in range(80 * 101) if lengths[e] > 5][:8000]
        assert len(memory) == 8000
        for group in range(81, 185):
            if group in excluded_groups:
                rejected.append(dict(group=group, reason="prior_screen_group"))
                continue
            first = starts[group * 101]
            if any(not distinct_start(first, other) for other in selected_starts):
                rejected.append(dict(group=group, reason="near_duplicate_selected_start"))
                continue
            chosen = None
            for episode in range(group * 101, (group + 1) * 101):
                if lengths[episode] < 141:
                    continue
                start = int(offsets[episode])
                if P.initially_successful("pusht", data, start, start + 140):
                    continue
                states = np.asarray(data["state"][start : start + 141], dtype=np.float64)
                layout = M.pusht_layout(states, "on_path", M.TASKS["pusht"])
                if layout is None:
                    continue
                chosen = dict(
                    id=f"pusht-scene-{len(candidates):02d}",
                    group=group,
                    episode=episode,
                    start_row=start,
                    goal_row=start + 140,
                    horizon=140,
                    start_state=states[0].tolist(),
                    **layout,
                )
                break
            if chosen is None:
                rejected.append(dict(group=group, reason="no_geometry_eligible_episode"))
                continue
            candidates.append(chosen)
            selected_starts.append(first)
            if len(candidates) == 8:
                break
        excluded = [e for c in candidates for e in range(c["group"] * 101, (c["group"] + 1) * 101)]
        assert not set(memory).intersection(excluded)
        receipt = json.loads((BASE / "data-receipt.json").read_text())
        panel = dict(
            task="pusht",
            cases=candidates,
            memory_episodes=memory,
            excluded_episodes=excluded,
            excluded_prior_groups=sorted(excluded_groups),
            rejected=rejected,
            groups_verified=185,
            group_size=101,
            dataset_sha256=receipt["dataset"]["sha256"],
            created=time.time(),
            status="ready" if len(candidates) == 8 else "insufficient_distinct_scenes",
        )
        M.save(target, panel)
    print(
        json.dumps(
            dict(
                status=panel["status"],
                scenes=len(candidates),
                groups=[c["group"] for c in candidates],
            )
        ),
        flush=True,
    )
    if len(candidates) != 8:
        raise RuntimeError("Panel shortfall; no recipe relaxation or episode execution")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["panel", "cache", "cache-finish"])
    parser.add_argument("--shard", type=int, default=0)
    args = parser.parse_args()
    M.configure_runtime()
    if args.mode == "panel":
        select_panel()
    elif args.mode == "cache":
        P.cache_stage("pusht", args.shard, 8)
    else:
        P.cache_finish("pusht")


if __name__ == "__main__":
    main()
