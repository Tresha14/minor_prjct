"""General-purpose statistics used by both inter-rater agreement and
system-vs-human-rating correlation (evaluation/validation.py). Pure
functions over plain lists of numbers -- no dependency on how the ratings
were collected, so they're testable with textbook/synthetic examples with
known, hand-derivable answers, independent of whether any real interviews
have been rated yet.

Every function here returns None (never a fabricated 0.0 or 1.0) when the
input can't actually support the statistic -- fewer than 2 observations,
zero variance in one series, or (for kappa) no disagreement was even
possible to measure. This mirrors analyzer/fusion.py's rule for missing
metrics: a statistic that can't be computed must look different from one
that was computed and happened to come out as "no relationship" or
"perfect agreement" -- see each function's docstring for the specific
degenerate case it guards against.
"""
from __future__ import annotations

import math


def pearson_correlation(x: list[float], y: list[float]) -> float | None:
    """Standard Pearson product-moment correlation coefficient.

    Returns None if there are fewer than 2 paired observations, or if
    either series has zero variance (a correlation with a constant series
    is undefined, not 0.0 -- 0.0 would wrongly read as "measured and found
    unrelated").
    """
    if len(x) != len(y):
        raise ValueError(f"x and y must be the same length, got {len(x)} and {len(y)}")
    n = len(x)
    if n < 2:
        return None
    mean_x, mean_y = sum(x) / n, sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _ranks(values: list[float]) -> list[float]:
    """Average ("fractional") ranks: tied values share the mean rank of the
    positions they occupy -- the standard convention for Spearman's rho
    with ties, and the reason a naive `sorted(...).index(...)` approach
    would silently give wrong answers whenever any two ratings are equal
    (common with small ordinal scales like 1-5)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_correlation(x: list[float], y: list[float]) -> float | None:
    """Pearson correlation computed over each series' ranks -- robust to a
    monotonic-but-non-linear relationship and to outliers, and the more
    appropriate statistic when ratings are ordinal (e.g. a 1-5 scale)
    rather than a continuous measurement."""
    if len(x) != len(y):
        raise ValueError(f"x and y must be the same length, got {len(x)} and {len(y)}")
    if len(x) < 2:
        return None
    return pearson_correlation(_ranks(x), _ranks(y))


def cohens_kappa(ratings_a: list, ratings_b: list) -> float | None:
    """Cohen's kappa: inter-rater agreement between exactly two raters on
    categorical/ordinal ratings of the same items, corrected for the
    agreement expected by chance alone (unlike plain % agreement, which
    looks artificially high whenever most items land in one category).

    Returns None if there are no items, or if both raters gave literally
    the same single category to every item -- kappa's denominator
    (1 - expected_agreement) is 0 in that case, meaning there was no
    disagreement variance for kappa to explain, so the statistic is
    undefined rather than a fabricated 1.0 "perfect agreement".
    """
    if len(ratings_a) != len(ratings_b):
        raise ValueError(f"ratings_a and ratings_b must be the same length, got {len(ratings_a)} and {len(ratings_b)}")
    n = len(ratings_a)
    if n == 0:
        return None
    categories = sorted(set(ratings_a) | set(ratings_b), key=str)
    observed_agreement = sum(1 for a, b in zip(ratings_a, ratings_b) if a == b) / n

    p_a = {c: sum(1 for x in ratings_a if x == c) / n for c in categories}
    p_b = {c: sum(1 for x in ratings_b if x == c) / n for c in categories}
    expected_agreement = sum(p_a[c] * p_b[c] for c in categories)

    if expected_agreement >= 1.0:
        return None
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)
