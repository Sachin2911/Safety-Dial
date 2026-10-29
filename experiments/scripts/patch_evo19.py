"""Apply the task 19 adapter hook to ManiGuard 6bf4c61's evaluator.

The same five audited edits as task 15 (`patch_regrasp15.EDITS`), with the adapter
imported from `evo19_runtime` instead of `regrasp15_runtime`. The evaluator is unchanged
unless SAFETY_DIAL_REGRASP_CONFIG is set.
"""

import argparse
import difflib
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_regrasp15 import EDITS as REGRASP_EDITS  # noqa: E402
from patch_regrasp15 import ORIGINAL_SHA256  # noqa: E402

EDITS = tuple((old, new.replace("regrasp15_runtime", "evo19_runtime"))
              for old, new in REGRASP_EDITS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--diff", type=Path, required=True)
    args = parser.parse_args()
    path = args.checkout / "maniguard/eval/benchmark.py"
    original = path.read_text()
    if hashlib.sha256(original.encode()).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("Evaluator is not the pristine pinned revision")
    source = original
    for old, new in EDITS:
        if source.count(old) != 1:
            raise ValueError("Adapter hook anchor is not unique")
        source = source.replace(old, new)
    compile(source, "benchmark.py", "exec")
    args.diff.parent.mkdir(parents=True, exist_ok=True)
    args.diff.write_text("".join(difflib.unified_diff(
        original.splitlines(True), source.splitlines(True), "a/benchmark.py", "b/benchmark.py")))
    path.write_text(source)
    print("patched", hashlib.sha256(source.encode()).hexdigest())


if __name__ == "__main__":
    main()
