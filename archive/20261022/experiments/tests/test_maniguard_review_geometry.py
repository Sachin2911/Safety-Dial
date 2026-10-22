"""Check frozen-membership reproduction and exclusion before opening outputs."""

import hashlib
import json

import numpy as np
import pytest

from scripts.maniguard_review_geometry import geometry, reproduce


def frozen_sample(tmp_path, stage):
    run = tmp_path / "run"
    run.mkdir()
    output = run / "episode"
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({
        "schema": "maniguard-review-feature-geometry-v1", "queue_sha256": "historical",
        "inputs": [{"attempt": "candidate", "family": "jar", "rows": 1,
                    "trace_sha256": hashlib.sha256(b"original trace").hexdigest(),
                    "prompt_sha256": "prompt"}],
    }))
    (run / "queue.json").write_text(json.dumps({"attempts": [{
        "id": "candidate", "stage": stage, "status": "completed", "output": str(output),
    }]}))
    return run, reference, output


def test_protected_membership_is_rejected_before_any_output_is_opened(tmp_path):
    run, reference, output = frozen_sample(tmp_path, "independent_audit")
    assert not output.exists()
    with pytest.raises(ValueError, match="excluded or incomplete"):
        reproduce(run, reference)


def test_mutated_training_trace_fails_its_frozen_hash_before_consumption(tmp_path):
    run, reference, output = frozen_sample(tmp_path, "shared_initial")
    output.mkdir()
    (output / "trace.jsonl").write_text("modified, not even valid JSON")
    with pytest.raises(ValueError, match="Frozen trace hash changed"):
        reproduce(run, reference)


def test_repeated_instruction_rows_have_one_centered_direction_and_explicit_counts():
    features = np.zeros((6, 44))
    features[:, 0] = 1.
    features[:, 36] = [1., 1., 1., 1., -1., -1.]
    result = geometry(features, ["first", "second"], episodes=2)
    assert result["feature_rows"] == 6 and result["episodes"] == 2
    assert result["distinct_projected_instruction_vectors"] == 2
    assert result["distinct_prompt_hashes"] == 2
    assert result["centered_relative_rank"] == {"0.01": 1, "0.001": 1, "0.0001": 1}
    assert 0 in result["near_constant_columns_std_below_1e_6"]
    assert 36 not in result["near_constant_columns_std_below_1e_6"]
