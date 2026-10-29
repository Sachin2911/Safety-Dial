"""Export the audited task-18 chart with embedded TrueType fonts for the manuscript."""

import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/scripts"))
media = importlib.import_module("media_lewm18")

source = ROOT / "outputs/lewm18-combined"
export = ROOT / "outputs/honours-gecco/task18-figure-export"
(export / "analysis").mkdir(parents=True, exist_ok=True)
shutil.copy2(source / "analysis/summary.json", export / "analysis/summary.json")
media.plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
media.figure(export)
np.testing.assert_array_equal(
    np.asarray(Image.open(source / "analysis/comparison.png")),
    np.asarray(Image.open(export / "analysis/comparison.png")),
)
target = Path(__file__).resolve().parent / "assets/lewm18-comparison.pdf"
shutil.copy2(export / "analysis/comparison.pdf", target)
receipt = {
    "change": "Manuscript PDF export uses embedded TrueType fonts instead of Type 3",
    "raster_chart_identical_to_published_figure": True,
    "source_summary_sha256": hashlib.sha256(
        (source / "analysis/summary.json").read_bytes()
    ).hexdigest(),
    "paper_figure_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
    "published_dataset_artifacts_unchanged": True,
}
(export / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
