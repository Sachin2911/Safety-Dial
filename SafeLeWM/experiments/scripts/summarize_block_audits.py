"""Summarize completed Block audits using the frozen Search 2 reading.

Run on a rented host. One audit produces a repeat report; exactly three produce
the pooled report. Pairing always stays within the original repeat and host.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import socket
import time


ARMS = ("native", "untrained", "random", "cma")
METRICS = ("sustained", "benchmark_success", "violation", "unfinished", "controls",
           "interventions")
AUDITOR = "df250a3716cd3db778c7bcf7769a7b46c13b73b63d4328071681725c7022cc5c"
PHASES = dict(smoke=4, cma=512, select=64, final=64, random=512)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(final):
    return {
        arm: dict(n=len(rows), **{m: sum(r[m] for r in rows.values()) for m in METRICS})
        for arm, rows in final.items()
    }


def reading(counts, repeats):
    safe = counts["cma"]["sustained"] - counts["native"]["sustained"]
    violation = counts["cma"]["violation"] - counts["native"]["violation"]
    random_difference = counts["cma"]["sustained"] - counts["random"]["sustained"]
    verdict = (
        "transfers" if safe >= 3 * repeats and violation <= 0 else
        "no signal" if abs(safe) <= repeats and abs(violation) <= repeats else "mixed"
    )
    return dict(verdict=verdict, cma_minus_native_sustained=safe,
                cma_minus_native_violations=violation,
                cma_minus_random_sustained=random_difference,
                random_search_tie=abs(random_difference) <= repeats)


def paired_counts(final):
    out = {}
    for reference in ("native", "untrained", "random"):
        pairs = [(row["sustained"], final[reference][seed]["sustained"])
                 for seed, row in final["cma"].items()]
        out[reference] = dict(
            cma_only=sum(a == 1 and b == 0 for a, b in pairs),
            reference_only=sum(a == 0 and b == 1 for a, b in pairs),
            both=sum(a == b == 1 for a, b in pairs),
            neither=sum(a == b == 0 for a, b in pairs),
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audits", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert Path("/workspace/Safety-Dial").is_dir(), "Run on the rented research host"
    assert len(args.audits) in (1, 3), "No partial pooled report"
    assert not args.output.exists(), "Preserve earlier reports"
    reports, combined, resources = {}, {arm: {} for arm in ARMS}, Counter()
    sources = None
    hosts = set()
    for path in args.audits:
        audit = json.loads(path.read_text())
        repeat = audit["repeat"]
        assert repeat in (0, 1, 2) and repeat not in reports
        assert audit["passed"] and audit["auditor_sha256"] == AUDITOR
        assert audit["phase_counts"] == PHASES
        if sources is None:
            sources = audit["source_bindings"]
        assert sources == audit["source_bindings"], "Repeat implementation differs"
        final = audit["final"]
        assert set(final) == set(ARMS)
        seeds = set(final["native"])
        assert len(seeds) == 16
        for arm in ARMS:
            assert set(final[arm]) == seeds
            for seed, row in final[arm].items():
                assert row["benchmark_success"] + row["unfinished"] == 1
                assert row["sustained"] <= row["benchmark_success"]
                assert not (row["sustained"] and row["violation"])
                combined[arm][f"r{repeat}:{seed}"] = row
        counts = aggregate(final)
        reports[repeat] = dict(
            host=audit["host"], audit_sha256=sha(path), final=counts,
            reading=reading(counts, 1), paired_sustained=paired_counts(final),
            resources=audit["resources"], failed_attempts=len(audit["failed_attempts"]),
            limitations=audit["limitations"],
        )
        resources.update(audit["resources"])
        hosts.add(audit["host"])
    pooled = None
    if len(reports) == 3:
        assert set(reports) == {0, 1, 2} and len(hosts) == 3
        counts = aggregate(combined)
        pooled = dict(final=counts, reading=reading(counts, 3),
                      paired_sustained=paired_counts(combined))
    result = dict(
        checked_at=time.time(), analysis_host=socket.gethostname(),
        source_sha256=sha(Path(__file__)), repeats=reports, pooled=pooled,
        completed_run_resources=dict(resources),
        limitations=[
            "Frozen categorical readings are descriptive, not significance tests",
            "Independent repeats use different hosts; every arm comparison stays within host",
            "Failed-attempt raw supplements and discarded setup work must be accounted separately",
            "No additional rollouts, tuning, simulator replay or optimizer numerical replay",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), repeats=sorted(reports), pooled=pooled)))


if __name__ == "__main__":
    main()
