"""Make the checkout's experiment modules available without installing the legacy project."""

import sys
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXPERIMENTS / "helpers"))
