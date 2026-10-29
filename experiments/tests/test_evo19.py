import glob
import math
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/helpers"))
sys.path.insert(0, str(ROOT / "experiments/scripts"))

import evo19_check  # noqa: E402
import regrasp15_check  # noqa: E402

TASK15 = sorted(glob.glob(str(ROOT / "outputs/regrasp15/remote/episodes/job1*project-0.10*")))


def config(**kw):
    base = {"mode": "project", "speed_limit_mps": 0.1, "near_radius_m": 0.22,
            "retreat_weight": 1.0}
    base.update(kw)
    return evo19_check.EvoConfig(**base)


def test_native_config_rejects_limits():
    with pytest.raises(ValueError):
        evo19_check.EvoConfig(mode="native", speed_limit_mps=0.1)
    with pytest.raises(ValueError):
        config(retreat_weight=1.5)
    assert evo19_check.EvoConfig().mode == "native"


@pytest.mark.skipif(not TASK15, reason="task 15 recordings are not mirrored locally")
def test_unit_retreat_weight_reproduces_task15_projection():
    old = regrasp15_check.CheckConfig(mode="project", speed_limit_mps=0.1)
    new = config()
    checked = projected = 0
    for episode in TASK15:
        rows = [json.loads(x) for x in Path(episode, "adapter.jsonl").read_text().splitlines()]
        yaw = next(r["yaw"] for r in rows if r["event"] == "base")
        for row in rows:
            if row["event"] != "decision" or not row["gate_open"]:
                continue
            args = (np.asarray(row["native"], np.float32), np.asarray(row["q"]),
                    np.asarray(row["eef"]), np.asarray(row["jar"]), yaw)
            a = regrasp15_check.features(*args, old.near_radius_m)
            b = evo19_check.features(*args, new)
            assert a["max_near_speed_mps"] == b["max_near_speed_mps"]
            if row["action"] == "projected":
                ra, sa, _ = regrasp15_check.project(*args, old)
                rb, sb, _ = evo19_check.project(*args, new)
                assert sa == sb == row["scale"]
                assert ra.astype(np.float32).tobytes() == rb.astype(np.float32).tobytes()
                assert rb.astype(np.float32).tobytes() == np.asarray(
                    row["executed"], np.float32).tobytes()
                projected += 1
            checked += 1
    assert checked > 1000 and projected > 100


@pytest.mark.skipif(not TASK15, reason="task 15 recordings are not mirrored locally")
def test_retreat_weight_only_relaxes_the_check():
    loose, strict = config(retreat_weight=0.0), config()
    for episode in TASK15[:4]:
        rows = [json.loads(x) for x in Path(episode, "adapter.jsonl").read_text().splitlines()]
        yaw = next(r["yaw"] for r in rows if r["event"] == "base")
        for row in rows:
            if row["event"] == "decision" and row["gate_open"]:
                args = (np.asarray(row["native"]), np.asarray(row["q"]), np.asarray(row["eef"]),
                        np.asarray(row["jar"]), yaw)
                a = evo19_check.features(*args, loose)["max_near_speed_mps"]
                b = evo19_check.features(*args, strict)["max_near_speed_mps"]
                assert a <= b


def test_shield_settings_and_retrace():
    import pytest

    from evo19_check import EvoConfig, retrace, tilt_deg, triggered

    shield = EvoConfig(mode="shield", gate="always", tilt_limit_deg=9.0, retreat_m=0.07)
    assert shield.settings()["retreat_m"] == 0.07
    assert "tilt_limit_deg" not in EvoConfig(mode="project", speed_limit_mps=0.1).settings()
    for bad in ({"gate": "closure"}, {"retreat_m": 0.0}, {"speed_limit_mps": 0.1}):
        with pytest.raises(ValueError):
            EvoConfig(**{**shield.settings(), **bad})
    with pytest.raises(ValueError):
        EvoConfig(mode="project", speed_limit_mps=0.1, tilt_limit_deg=9.0)
    assert abs(tilt_deg([0, 0, 0.3, 0.954])) < 1e-9  # pure yaw is upright
    assert abs(tilt_deg([math.sin(0.1), 0, 0, math.cos(0.1)]) - math.degrees(0.2)) < 1e-9
    assert triggered(9.0, 1, shield) and not triggered(9.0, 0, shield)
    assert not triggered(8.9, 2, shield)
    q = np.arange(40, dtype=np.float64)[:, None].repeat(7, 1)
    eef = np.zeros((40, 3))
    eef[:, 0] = np.arange(40) * 0.01  # 1 cm per control
    native = np.ones((16, 8))
    native[:, 7] = np.linspace(-1, 1, 16)
    chunk, target = retrace(native, q, eef, 30, 0.065)
    assert target == 23
    assert chunk[0, 0] == 29 and chunk[7, 0] == 23 and chunk[15, 0] == 23
    assert (chunk[:, 7] == -1).all()
    chunk, target = retrace(native, q, eef, 3, 0.07)
    assert target == 0 and chunk[15, 0] == 0


def test_shield_contact_window():
    from evo19_check import EvoConfig, window_contacts

    base = {"mode": "shield", "gate": "always", "tilt_limit_deg": 9.0, "retreat_m": 0.07}
    query_only, windowed = EvoConfig(**base), EvoConfig(**base, contact_window=8)
    assert "contact_window" not in query_only.settings()
    assert windowed.settings()["contact_window"] == 8
    contacts = [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1]
    assert window_contacts(contacts, 9, query_only) == 0
    assert window_contacts(contacts, 10, query_only) == 1
    assert window_contacts(contacts, 9, windowed) == 2  # controls 2..9
    assert window_contacts(contacts, 8, windowed) == 2  # controls 1..8
    assert window_contacts(contacts, 1, windowed) == 0
    with pytest.raises(ValueError):
        EvoConfig(**base, contact_window=9)
