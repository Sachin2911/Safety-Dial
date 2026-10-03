"""Stage configs, pinned inputs and run bundles for the evolution study (docs/evoPlan).

Every stage reads one Hydra config from `configs/evo/` (which pulls in `inputs.yaml`, the
pinned Hugging Face revisions), downloads its inputs at those revisions, writes its outputs
to an immutable `runs/<run_id>/` folder plus a small committed copy under
`docs/evoPlan/results/<stage>/<run_id>/`, and uploads the run folder to the private model
repository under kind "evo".

    cfg = load_stage_config("stage0")
    paths = fetch_inputs(cfg.inputs)           # {"model": Path, "probes": Path, ...}
    run = EvoRun.create(cfg, "s0")             # run_id, run_dir, results_dir, config.yaml
    run.finish(manifest, upload=True)          # manifest.json in both, HF kind "evo"

Never print token values: `HFStore` reads them from the environment.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "evo"
RESULTS_ROOT = REPO_ROOT / "docs" / "evoPlan" / "results"
RUNS_ROOT = REPO_ROOT / "runs"
HF_KEY = "walker2d"
HF_KIND = "evo"
_FULL_SHA = re.compile(r"[0-9a-f]{40}")


def load_stage_config(name: str, overrides: list[str] | tuple[str, ...] = ()):
    """Compose `configs/evo/<name>.yaml` (with its defaults, e.g. `inputs`) through Hydra."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=name, overrides=list(overrides))


def fetch_inputs(inputs, names=None, *, store=None) -> dict[str, Path]:
    """Download each pinned input at its full-commit revision; refuse anything else."""
    from helpers.hfStore import HFStore

    store = store or HFStore()
    out = {}
    for name in names or list(inputs.keys()):
        spec = inputs[name]
        if not _FULL_SHA.fullmatch(str(spec.revision)):
            raise ValueError(f"input {name} must pin a full commit, got {spec.revision!r}")
        local = store.download_run(spec.key, spec.path, revision=spec.revision)
        receipt = json.loads((local / "hf_download.json").read_text())
        if receipt["revision"] != spec.revision:
            raise ValueError(f"input {name} resolved to {receipt['revision']}, pinned {spec.revision}")
        out[name] = local
    return out


def input_revisions(paths: dict[str, Path]) -> dict:
    """The download receipts of the inputs a run consumed (repo, path, resolved commit)."""
    out = {}
    for name, local in paths.items():
        receipt = json.loads((Path(local) / "hf_download.json").read_text())
        out[name] = {k: receipt[k] for k in ("repo_id", "repo_type", "path", "revision")}
    return out


def next_run_id(stage: str, n: int | None = None) -> str:
    """`walker2d-evo-<stage>-<yyyymmdd>-<n>`, the first unused n unless one is given."""
    from helpers.runManifest import make_run_id

    if n is not None:
        return make_run_id("walker2d", "evo", stage, n=n)
    k = 1
    while (RUNS_ROOT / make_run_id("walker2d", "evo", stage, n=k)).exists():
        k += 1
    return make_run_id("walker2d", "evo", stage, n=k)


@dataclass
class EvoRun:
    run_id: str
    stage: str
    run_dir: Path
    results_dir: Path

    @classmethod
    def create(cls, cfg, stage: str, run_id: str | None = None, *, resume: bool = False) -> "EvoRun":
        """Make the run folders and save the resolved config. A resumed run must already
        exist with an identical config; a new run never reuses a folder."""
        from omegaconf import OmegaConf

        from helpers.runManifest import validate_run_id

        run_id = validate_run_id(run_id) if run_id else next_run_id(stage)
        run_dir = RUNS_ROOT / run_id
        results_dir = RESULTS_ROOT / f"stage{stage.lstrip('s')}" / run_id
        text = OmegaConf.to_yaml(cfg, resolve=True)
        if resume:
            saved = run_dir / "config.yaml"
            if not saved.is_file() or saved.read_text() != text:
                raise ValueError(f"cannot resume {run_id}: missing or different config.yaml")
            results_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir.mkdir(parents=True, exist_ok=False)
            results_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "config.yaml").write_text(text)
        (results_dir / "config.yaml").write_text(text)
        return cls(run_id, stage, run_dir, results_dir)

    def write_json(self, name: str, payload, *, results: bool = True) -> None:
        """Atomic JSON write to the run folder and, if `results`, the committed copy."""
        text = json.dumps(payload, indent=1, default=_jsonable) + "\n"
        for dest in (self.run_dir, self.results_dir) if results else (self.run_dir,):
            path = dest / name
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(text)
            tmp.replace(path)

    def finish(self, manifest: dict, *, upload: bool = True, readme: str | None = None) -> str | None:
        """Write the manifest everywhere, then upload the run folder (kind "evo")."""
        from helpers.hfStore import HFStore
        from helpers.runManifest import write_manifest

        if readme:
            (self.run_dir / "README.md").write_text(readme)
        for dest in (self.run_dir, self.results_dir):
            write_manifest(dest, manifest)
        if not upload:
            return None
        revision = HFStore().upload_run(HF_KEY, HF_KIND, self.run_dir, run_id=self.run_id)
        receipt = json.loads((self.run_dir / "hf_upload.json").read_text())
        (self.results_dir / "hf_upload.json").write_text(json.dumps(receipt, indent=2) + "\n")
        return revision


def _jsonable(v):
    import numpy as np

    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    return str(v)
