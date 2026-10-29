"""The explicitly authorized second-host speed comparison, with a separate rental record."""

import os

os.environ["LEWM18_CONTROL_DIR"] = "outputs/lewm18-replacement"
os.environ["LEWM18_RENTAL_RECORD"] = "progress/rental-lewm18-replacement-20261026.json"
from manage_lewm18 import main

if __name__ == "__main__":
    raise SystemExit(main())
