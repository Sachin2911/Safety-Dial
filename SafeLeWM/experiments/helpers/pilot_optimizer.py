"""Explicit diagonal CMA-ES configuration and order-preserving updates."""

import cma
import numpy as np
from cma.sigma_adaptation import CMAAdaptSigmaCSA


def make_optimizer(dimension, sigma, seed, population=8, generations=4):
    return cma.CMAEvolutionStrategy(
        np.zeros(dimension),
        sigma,
        dict(
            CMA_diagonal=True,
            AdaptSigma=CMAAdaptSigmaCSA,
            CMA_mirrors=0,
            popsize=population,
            seed=seed,
            verbose=-9,
            verb_log=0,
            verb_disp=0,
            maxiter=generations,
        ),
    )


def tell_ordered(optimizer, genotypes, scores):
    if len(genotypes) != len(scores) or len(scores) != optimizer.popsize:
        raise ValueError("A complete population and one score per candidate are required")
    if not np.isfinite(scores).all():
        raise ValueError("Optimizer scores must be finite")
    optimizer.tell(genotypes, [-float(value) for value in scores])
