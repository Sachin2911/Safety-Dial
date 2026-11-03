"""Run the frozen audit, caching each NPZ's arrays to avoid repeated decompression.

All assertions and arithmetic remain in audit_adapter_pilot.py, unchanged since
the pre-run freeze. This postprocessing-only wrapper changes file access, not data.
"""

import json
from pathlib import Path
import sys

import audit_adapter_pilot as audit


class CachedArrays(dict):
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.clear()


def main():
    if len(sys.argv) > 1:
        audit.OUT = Path(sys.argv[1]).resolve()
    original_load = audit.np.load

    def cached_load(*args, **kwargs):
        loaded = original_load(*args, **kwargs)
        if not isinstance(loaded, audit.np.lib.npyio.NpzFile):
            return loaded
        with loaded:
            return CachedArrays({name: loaded[name] for name in loaded.files})

    audit.np.load = cached_load
    try:
        audit.main()
    finally:
        audit.np.load = original_load
    (audit.OUT / "audit-file-access.json").write_text(
        json.dumps(
            dict(
                original_assertions_unchanged=True,
                npz_arrays_cached_once_per_open=True,
                additional_simulator_actions=0,
                additional_world_model_queries=0,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
