"""Statistical tools for comparing two sets of paired per-query metric scores
(typically a "before" and "after" run over the same fixed evaluation set).

Implemented with the standard library only: a percentile bootstrap confidence
interval, a paired sign-flip permutation test, and Cohen's d for paired samples.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of `values`."""
    if not values:
        raise ValueError("values must be non-empty")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    if len(values) == 1:
        # A single observation carries no resampling variance; the CI collapses
        # to a point rather than raising, so callers can still print a result.
        return (float(values[0]), float(values[0]))

    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(resample))
    means.sort()

    alpha = 1 - confidence
    lower_idx = int((alpha / 2) * n_resamples)
    upper_idx = int((1 - alpha / 2) * n_resamples) - 1
    upper_idx = min(upper_idx, n_resamples - 1)
    return (means[lower_idx], means[upper_idx])


def paired_permutation_test(
    before: Sequence[float],
    after: Sequence[float],
    n_permutations: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided sign-flip permutation test p-value for paired differences.

    Under the null hypothesis that "before" and "after" are exchangeable, the
    sign of each paired difference is equally likely to be positive or
    negative. The p-value is the fraction of random sign-flips whose mean
    absolute difference is at least as extreme as the one actually observed.
    """
    if len(before) != len(after):
        raise ValueError("before and after must have the same length")
    if not before:
        raise ValueError("before and after must be non-empty")

    diffs = [a - b for b, a in zip(before, after)]
    observed = abs(statistics.fmean(diffs))

    if observed == 0.0 and all(d == 0.0 for d in diffs):
        # No difference anywhere: nothing for a sign flip to disturb.
        return 1.0

    rng = random.Random(seed)
    n = len(diffs)
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        flipped = [d if rng.random() < 0.5 else -d for d in diffs]
        if abs(statistics.fmean(flipped)) >= observed - 1e-12:
            at_least_as_extreme += 1

    # +1/+1 smoothing avoids ever reporting a p-value of exactly 0.
    return (at_least_as_extreme + 1) / (n_permutations + 1)


def cohens_d_paired(before: Sequence[float], after: Sequence[float]) -> float:
    """Cohen's d for paired samples: mean difference / stdev of differences.

    Returns 0.0 for the degenerate case of zero-variance, zero-mean
    differences (before and after are identical). If the differences are
    constant but non-zero, the effect is reported as an infinite (signed) d,
    since there is truly no variance to divide a non-zero mean by.
    """
    if len(before) != len(after):
        raise ValueError("before and after must have the same length")
    if not before:
        raise ValueError("before and after must be non-empty")
    if len(before) == 1:
        raise ValueError("cohens_d_paired requires at least 2 paired observations")

    diffs = [a - b for b, a in zip(before, after)]
    mean_diff = statistics.fmean(diffs)
    stdev_diff = statistics.stdev(diffs)

    if stdev_diff == 0.0:
        if mean_diff == 0.0:
            return 0.0
        return float("inf") if mean_diff > 0 else float("-inf")

    return mean_diff / stdev_diff
