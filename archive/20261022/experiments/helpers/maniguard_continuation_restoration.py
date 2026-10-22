"""Layer recovered screen waves without allowing physical evidence to change."""

import hashlib
import shutil
from pathlib import Path

MUTABLE_METADATA = {
    "screen/screen.json",
    "screen/source-records/maniguard-continuation-screen-20261020-v1-coordinator.log",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge_recovered_waves(waves, destination):
    """Keep each downloaded wave intact and layer only two named mutable records."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    replacements = []
    for wave, record in enumerate(waves):
        base = Path(record["merged"])
        for source in sorted(base.rglob("*")):
            if source.is_symlink():
                raise ValueError("Recovered screen evidence cannot contain symlinks")
            if not source.is_file():
                continue
            relative = source.relative_to(base)
            if relative.parts[0] != "screen":
                raise ValueError("Use the separate screen archives, without original-run overlays")
            target = destination / relative
            if target.exists() and digest(target) != digest(source):
                if relative.as_posix() not in MUTABLE_METADATA:
                    raise ValueError("Immutable screen evidence differs: " + str(relative))
                replacements.append({"path": relative.as_posix(), "wave": wave,
                                     "earlier_sha256": digest(target),
                                     "later_sha256": digest(source)})
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return replacements
