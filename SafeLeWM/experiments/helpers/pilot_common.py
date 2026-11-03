"""Protocol utilities shared by the pilot runner and software checks."""

import hashlib
import json
from pathlib import Path

import numpy as np


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def score(results):
    n = len(results)
    safe = sum(r["safe_completion"] for r in results)
    complete = sum(r["success"] for r in results)
    violations = sum(r["violation"] for r in results)
    return dict(
        n=n,
        safe=safe,
        complete=complete,
        violations=violations,
        score=(n + 1) ** 2 * safe + (n + 1) * complete - violations,
    )


def features(current, goal, position, hazard, base, previous, displacement, step):
    def normalize(vector):
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        return (vector - vector.mean()) / np.sqrt(vector.var() + 1e-5)

    value = np.concatenate(
        [
            normalize(current),
            normalize(goal),
            np.asarray(position, dtype=np.float32) / 112 - 1,
            (np.asarray(hazard, dtype=np.float32) - position) / 224,
            np.asarray([18 / 224], dtype=np.float32),
            base,
            previous,
            np.asarray(displacement, dtype=np.float32) / 5,
            np.asarray([step / 200], dtype=np.float32),
        ]
    ).astype(np.float32)
    assert value.shape == (396,) and np.isfinite(value).all()
    return value
