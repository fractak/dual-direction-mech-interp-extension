"""Paired statistics; shared resampling indices preserve prompt dependence."""

from __future__ import annotations

import math

import numpy as np


def exact_p(lost, gained):
    n = lost + gained
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(lost, gained) + 1)) / 2**n)


def bootstrap_indices(n, samples=20000, seed=42):
    return np.random.default_rng(seed).integers(0, n, size=(samples, n))


def delta_summary(delta, indices):
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 1 or len(delta) != indices.shape[1]:
        raise ValueError("Bootstrap indices do not match prompt count")
    lower, upper = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975], method="linear")
    return {"change": float(delta.mean()), "bootstrap_95_ci": [float(lower), float(upper)]}


def paired(before, after, indices):
    before, after = np.asarray(before, dtype=bool), np.asarray(after, dtype=bool)
    if before.shape != after.shape:
        raise ValueError("Paired conditions must contain the same prompts")
    lost, gained = int(np.sum(before & ~after)), int(np.sum(~before & after))
    return {
        "n": len(before),
        "before_count": int(before.sum()),
        "after_count": int(after.sum()),
        "lost": lost,
        "gained": gained,
        "exact_p": exact_p(lost, gained),
        **delta_summary(after.astype(float) - before.astype(float), indices),
    }


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    adjusted, previous = {}, 0.0
    for rank, key in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * pvalues[key]))
        adjusted[key] = previous
    return adjusted
