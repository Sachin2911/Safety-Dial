"""Prospectively specified paired contrasts blocked by physical host."""

import numpy as np

from helpers.unisafe09_statistics import exact_binomial, paired_binary


def matched_groups(rows, candidate, reference):
    groups = {}
    for host in sorted({str(r["physical_host"]) for r in rows}):
        a = sorted((r for r in rows if str(r["physical_host"]) == host
                    and r["condition"] == candidate), key=lambda r: r["seed"])
        b = sorted((r for r in rows if str(r["physical_host"]) == host
                    and r["condition"] == reference), key=lambda r: r["seed"])
        if not a or [r["seed"] for r in a] != [r["seed"] for r in b]:
            raise ValueError("Every host needs a complete matched seed panel")
        if len({r["seed"] for r in a}) != len(a):
            raise ValueError("Duplicated paired seed")
        groups[host] = (a, b)
    if not groups:
        raise ValueError("A nonempty paired panel is required")
    return groups


def blocked_binary(groups, key, alpha=0.05):
    hosts = len(groups)
    n = sum(len(a) for a, _ in groups.values())
    details = {}
    for host, (a, b) in groups.items():
        # paired_binary splits this host error over its win and loss marginals.
        details[host] = paired_binary([r[key] for r in a], [r[key] for r in b], alpha/hosts)
    return {
        "n": n, "hosts": hosts,
        "difference": sum(d["n"]/n*d["difference"] for d in details.values()),
        "interval": [sum(d["n"]/n*d["interval"][i] for d in details.values()) for i in (0, 1)],
        "alpha": alpha, "marginal_error": alpha/(2*hosts), "by_host": details,
        "method": "Seed-count-weighted host differences; exact win/loss bounds at alpha/(2H)",
    }


def blocked_mean(groups, draws_by_host, key="J", alpha=0.05):
    n = sum(len(a) for a, _ in groups.values())
    draws_n = {len(draws_by_host[h]) for h in groups}
    if draws_n != {10000}:
        raise ValueError("The frozen analysis requires 10000 shared resamples")
    bootstrap = np.zeros(10000)
    mean = 0.0
    details = {}
    for host, (a, b) in groups.items():
        difference = np.asarray([x[key]-y[key] for x, y in zip(a, b)], float)
        draws = draws_by_host[host]
        if draws.shape != (10000, len(a)) or draws.min() < 0 or draws.max() >= len(a):
            raise ValueError("Resampling must stay within each physical host")
        weight = len(a)/n
        bootstrap += weight*difference[draws].mean(axis=1)
        mean += weight*float(difference.mean())
        details[host] = {"n": len(a), "weight": weight,
                         "difference": float(difference.mean())}
    interval = np.quantile(bootstrap, [alpha/2, 1-alpha/2]).tolist()
    return {"n": n, "difference": mean, "interval": interval, "alpha": alpha,
            "degenerate": interval[0] == interval[1], "by_host": details,
            "method": "Paired seed bootstrap within physical host; 10000 shared resamples"}


def blocked_rate(rows, key, alpha=0.05):
    hosts = sorted({str(r["physical_host"]) for r in rows})
    n = len(rows)
    bounds = [0.0, 0.0]
    for host in hosts:
        group = [r for r in rows if str(r["physical_host"]) == host]
        interval = exact_binomial(sum(bool(r[key]) for r in group), len(group), alpha/len(hosts))
        for i in (0, 1):
            bounds[i] += len(group)/n*interval[i]
    return bounds
