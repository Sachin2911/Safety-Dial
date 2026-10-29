"""Multi-objective search over the constraint interface: NSGA-II, random search, grid.

Genotypes are real vectors in [0, 1]^5 decoded to planner parameters. Both objectives
are minimized: exposure (fraction of saved states with the pusher centre inside a true
gate box) and J (steps to terminal success, with the 50-step budget charged to every
unfinished episode). The search never touches the frozen world model.
"""

from __future__ import annotations

import itertools

import numpy as np

GENES = ("d0", "d1", "horizon", "topk_frac", "tol")


def decode(x, genes):
    """Map a unit-interval vector to planner parameters using the frozen gene bounds."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    out = {}
    for value, name in zip(x, GENES):
        low, high = float(genes[name]["low"]), float(genes[name]["high"])
        raw = low + (high - low) * value
        out[name] = int(round(raw)) if name == "horizon" else float(raw)
    return out


def encode(params, genes):
    """Inverse of decode for grid points and tests."""
    x = []
    for name in GENES:
        low, high = float(genes[name]["low"]), float(genes[name]["high"])
        x.append((float(params[name]) - low) / (high - low))
    return np.clip(np.asarray(x), 0.0, 1.0)


def clearance_schedule(params, horizon=None, cap=80.0):
    h = int(params["horizon"] if horizon is None else horizon)
    return [float(np.clip(params["d0"] + params["d1"] * t, 0.0, cap)) for t in range(h)]


def grid_points(grid, genes):
    points = []
    for d0, d1, horizon in itertools.product(grid["d0"], grid["d1"], grid["horizon"]):
        params = {
            "d0": float(d0),
            "d1": float(d1),
            "horizon": int(horizon),
            "topk_frac": float(grid["topk_frac"]),
            "tol": float(grid["tol"]),
        }
        points.append(encode(params, genes))
    return points


def random_points(count, dim, rng):
    return [rng.random(dim) for _ in range(count)]


def dominates(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated_sort(objectives):
    """Return a list of fronts, each a list of indices (front 0 is best)."""
    n = len(objectives)
    dominated_by = [set() for _ in range(n)]
    counts = [0] * n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominated_by[p].add(q)
            elif dominates(objectives[q], objectives[p]):
                counts[p] += 1
        if counts[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                counts[q] -= 1
                if counts[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return [f for f in fronts if f]


def crowding_distance(objectives, front):
    distance = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            distance[i] = float("inf")
        return distance
    objectives = np.asarray(objectives, dtype=float)
    for m in range(objectives.shape[1]):
        order = sorted(front, key=lambda i: objectives[i, m])
        lo, hi = objectives[order[0], m], objectives[order[-1], m]
        distance[order[0]] = distance[order[-1]] = float("inf")
        if hi - lo <= 0:
            continue
        for k in range(1, len(order) - 1):
            distance[order[k]] += (objectives[order[k + 1], m] - objectives[order[k - 1], m]) / (
                hi - lo
            )
    return distance


def rank_population(objectives):
    """(rank, crowding) per index, lower rank and higher crowding preferred."""
    ranks = {}
    crowd = {}
    for r, front in enumerate(non_dominated_sort(objectives)):
        for i in front:
            ranks[i] = r
        crowd.update(crowding_distance(objectives, front))
    return ranks, crowd


def tournament(population, ranks, crowd, rng):
    a, b = rng.integers(len(population)), rng.integers(len(population))
    if ranks[a] != ranks[b]:
        return population[a] if ranks[a] < ranks[b] else population[b]
    return population[a] if crowd[a] >= crowd[b] else population[b]


def sbx(p1, p2, rng, eta=15.0, prob=0.9):
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > prob:
        return c1, c2
    for i in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[i] - p2[i]) < 1e-12:
            continue
        u = rng.random()
        beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
        c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def polynomial_mutation(x, rng, eta=20.0, prob=None):
    x = np.asarray(x, dtype=float).copy()
    prob = 1.0 / len(x) if prob is None else prob
    for i in range(len(x)):
        if rng.random() > prob:
            continue
        u = rng.random()
        if u < 0.5:
            delta = (2 * u) ** (1 / (eta + 1)) - 1
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
        x[i] = np.clip(x[i] + delta, 0.0, 1.0)
    return x


def nsga2_offspring(population, objectives, size, rng, eta_c=15.0, eta_m=20.0, prob_c=0.9):
    ranks, crowd = rank_population(objectives)
    children = []
    while len(children) < size:
        p1 = tournament(population, ranks, crowd, rng)
        p2 = tournament(population, ranks, crowd, rng)
        c1, c2 = sbx(p1, p2, rng, eta=eta_c, prob=prob_c)
        children.append(polynomial_mutation(c1, rng, eta=eta_m))
        if len(children) < size:
            children.append(polynomial_mutation(c2, rng, eta=eta_m))
    return children


def nsga2_survivors(population, objectives, size):
    """Elitist environmental selection by rank, then crowding distance."""
    fronts = non_dominated_sort(objectives)
    chosen = []
    for front in fronts:
        if len(chosen) + len(front) <= size:
            chosen += front
            continue
        crowd = crowding_distance(objectives, front)
        chosen += sorted(front, key=lambda i: -crowd[i])[: size - len(chosen)]
        break
    return [population[i] for i in chosen], [objectives[i] for i in chosen], chosen


def pareto_front(objectives):
    return non_dominated_sort(objectives)[0] if len(objectives) else []


def hypervolume_2d(points, reference):
    """Exact dominated area for a minimization problem with two objectives."""
    ref = np.asarray(reference, dtype=float)
    pts = [np.asarray(p, dtype=float) for p in points if np.all(np.asarray(p) < ref)]
    if not pts:
        return 0.0
    front = [pts[i] for i in pareto_front(pts)]
    front.sort(key=lambda p: p[0])
    area = 0.0
    prev_x = ref[0]
    for p in sorted(front, key=lambda p: -p[0]):
        area += (prev_x - p[0]) * (ref[1] - p[1])
        prev_x = p[0]
    return float(area)


def crowded_subset(objectives, indices, cap):
    """Keep at most cap points of a front, always its two extremes."""
    if len(indices) <= cap:
        return list(indices)
    crowd = crowding_distance(objectives, list(indices))
    return sorted(indices, key=lambda i: -crowd[i])[:cap]
