"""Build a visibly labelled demonstration PDF using only invented aggregate counts."""

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT / "demo-data.json").read_text())
    if data.get("synthetic") is not True:
        raise ValueError("This builder accepts explicitly synthetic data only")
    categories = ("safe_complete", "unsafe_complete", "safe_unfinished", "unsafe_unfinished")
    rows = []
    for condition in data["conditions"]:
        counts = [condition[key] for key in categories]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("Counts must be nonnegative integers")
        if sum(counts) != condition["episodes"]:
            raise ValueError("Synthetic outcome categories must sum to the denominator")
        rows.append(
            condition["label"]
            + " & "
            + " & ".join(str(value) for value in counts)
            + r" \\"
        )
    table = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Setting & SC & UC & SU & UU\\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (ROOT / "synthetic-table.tex").write_text("\n".join(table) + "\n")
    plots = []
    for key, color in zip(categories, ("teal", "orange", "gray!45", "red!65"), strict=True):
        coordinates = " ".join(
            f"({index},{100 * condition[key] / condition['episodes']:.2f})"
            for index, condition in enumerate(data["conditions"])
        )
        plots.append(
            r"\addplot[fill=" + color + r",draw=white] coordinates {" + coordinates + "};"
        )
    (ROOT / "synthetic-chart.tex").write_text("\n".join(plots) + "\n")
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-halt-on-error", "-interaction=batchmode", "synthetic-supervisor-demo.tex"],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    print("Built synthetic-supervisor-demo.pdf from explicitly invented counts.")


if __name__ == "__main__":
    main()
