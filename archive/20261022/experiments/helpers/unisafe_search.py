"""Transparent finite-budget optimizers for the same categorical task adapter."""

import numpy as np


class AdapterSearch:
    def __init__(self, method, domain, population, seed):
        if method not in ("evolution", "random", "policy_gradient"):
            raise ValueError(method)
        if population < 2:
            raise ValueError("population must be at least two")
        self.method = method
        self.domain = domain
        self.keys = list(domain)
        self.population = population
        self.rng = np.random.default_rng(seed)
        self.logits = [np.zeros(len(domain[k])) for k in self.keys]
        self.first_moment = [np.zeros_like(x) for x in self.logits]
        self.second_moment = [np.zeros_like(x) for x in self.logits]
        self.parents = []
        self.archive = []
        self.updates = 0

    def probabilities(self):
        result = []
        for logits in self.logits:
            p = np.exp(logits - logits.max())
            result.append(p / p.sum())
        return result

    def ask(self):
        probabilities = self.probabilities()
        genes = []
        for i in range(self.population):
            if self.method == "evolution" and self.parents:
                if i == 0:
                    child = self.parents[0].copy()  # Reevaluate the elite; never reuse fitness.
                else:
                    parent = self.parents[self.rng.integers(len(self.parents))]
                    child = parent.copy()
                    changed = self.rng.integers(len(self.keys))
                    for j, key in enumerate(self.keys):
                        if j == changed or self.rng.random() < 0.5:
                            if self.rng.random() < 0.2:
                                child[j] = int(self.rng.integers(len(self.domain[key])))
                            else:
                                child[j] = int(np.clip(
                                    child[j] + self.rng.choice([-2, -1, 1, 2]),
                                    0, len(self.domain[key]) - 1,
                                ))
            else:
                child = [int(self.rng.choice(len(p), p=p)) for p in probabilities]
            genes.append(child)
        return genes

    def decode(self, gene):
        return {key: self.domain[key][gene[j]] for j, key in enumerate(self.keys)}

    def tell(self, genes, fitness):
        fitness = np.asarray(fitness, dtype=float)
        if len(genes) != self.population or fitness.shape != (self.population,):
            raise ValueError("Every candidate must have one measured fitness")
        if not np.isfinite(fitness).all():
            raise ValueError("Fitness must be finite")
        for gene, value in zip(genes, fitness):
            self.archive.append({"gene": list(gene), "fitness": float(value)})
        order = np.argsort(-fitness, kind="stable")
        self.parents = [list(genes[i]) for i in order[:2]]
        self.updates += 1
        if self.method == "policy_gradient":
            # REINFORCE with a leave-one-out baseline. One action selects a temporal
            # adapter for an episode group; the environment includes the fixed filter.
            advantages = fitness - (fitness.sum() - fitness) / (len(fitness) - 1)
            for j, p in enumerate(self.probabilities()):
                indicators = np.eye(len(p))[[g[j] for g in genes]]
                gradient = ((indicators - p) * advantages[:, None]).mean(axis=0)
                entropy = -(p * np.log(p.clip(1e-12))).sum()
                gradient += 0.01 * (-p * (np.log(p.clip(1e-12)) + entropy))
                self.first_moment[j] = 0.9 * self.first_moment[j] + 0.1 * gradient
                self.second_moment[j] = 0.999 * self.second_moment[j] + 0.001 * gradient**2
                m = self.first_moment[j] / (1 - 0.9**self.updates)
                v = self.second_moment[j] / (1 - 0.999**self.updates)
                self.logits[j] += 0.5 * m / (np.sqrt(v) + 1e-8)

    def finalists(self, count=3):
        if self.method == "policy_gradient":
            probabilities = self.probabilities()
            products = np.ones(tuple(len(p) for p in probabilities))
            for j, p in enumerate(probabilities):
                shape = [1] * len(probabilities)
                shape[j] = len(p)
                products *= p.reshape(shape)
            order = np.argsort(-products.ravel(), kind="stable")[:count]
            genes = [list(np.unravel_index(i, products.shape)) for i in order]
        else:
            measured = {}
            for entry in self.archive:
                measured.setdefault(tuple(entry["gene"]), []).append(entry["fitness"])
            genes = sorted(measured, key=lambda g: -np.mean(measured[g]))[:count]
        return [self.decode(g) for g in genes]

    def state(self):
        return {
            "method": self.method, "domain": self.domain, "population": self.population,
            "updates": self.updates, "probabilities": [p.tolist() for p in self.probabilities()],
            "parents": self.parents, "archive": self.archive,
            "rng_state": self.rng.bit_generator.state,
            "finalists": self.finalists(),
        }
