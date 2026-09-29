"""Tests for evaluation/statistics.py. All data here is synthetic/textbook
example data used only to prove the math is correct -- never a real
project result. See evaluation/README.md.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.statistics import cohens_kappa, pearson_correlation, spearman_correlation


# ---------------------------------------------------------------- pearson_correlation

def test_pearson_perfect_positive():
    assert pearson_correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == pytest.approx(1.0)


def test_pearson_perfect_negative():
    assert pearson_correlation([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]) == pytest.approx(1.0 * -1)


def test_pearson_golden_value_non_perfect():
    # x=[1,2,3,4,5], y=[1,4,9,16,25] (monotonic, not linear)
    # mean_x=3, mean_y=11; cov=60; var_x=10; var_y=374 -> r = 60/sqrt(3740)
    import math
    expected = 60 / math.sqrt(10 * 374)
    assert pearson_correlation([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]) == pytest.approx(expected)


def test_pearson_zero_variance_returns_none_not_zero():
    assert pearson_correlation([1, 2, 3], [5, 5, 5]) is None


def test_pearson_too_few_observations_returns_none():
    assert pearson_correlation([1], [1]) is None
    assert pearson_correlation([], []) is None


def test_pearson_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        pearson_correlation([1, 2, 3], [1, 2])


# ---------------------------------------------------------------- spearman_correlation

def test_spearman_perfect_for_monotonic_nonlinear_relationship():
    # y = x^2 for x > 0 is monotonic but not linear -- spearman should be
    # a perfect 1.0 even though pearson (see above test) is not.
    assert spearman_correlation([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]) == pytest.approx(1.0)


def test_spearman_handles_ties_via_average_ranks():
    # x has a tie at value 2 (positions 2,3 -> average rank 2.5 each).
    # ranks_x = [1, 2.5, 2.5, 4], ranks_y = [1, 2, 3, 4] (no ties in y).
    # mean=2.5 for both; dev_x=[-1.5,0,0,1.5], dev_y=[-1.5,-0.5,0.5,1.5]
    # cov=4.5, var_x=4.5, var_y=5.0 -> r = 4.5/sqrt(4.5*5.0) = 4.5/sqrt(22.5)
    import math
    x = [1, 2, 2, 4]
    y = [1, 2, 3, 4]
    result = spearman_correlation(x, y)
    assert result == pytest.approx(4.5 / math.sqrt(22.5))


def test_spearman_too_few_observations_returns_none():
    assert spearman_correlation([1], [1]) is None


def test_spearman_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        spearman_correlation([1, 2, 3], [1, 2])


# ---------------------------------------------------------------- cohens_kappa

def test_cohens_kappa_golden_value():
    # Hand-calculated: observed agreement 4/5=0.8, expected agreement
    # 0.6*0.4 + 0.4*0.6 = 0.48 -> kappa = (0.8-0.48)/(1-0.48) = 0.32/0.52
    ratings_a = ["A", "A", "B", "B", "A"]
    ratings_b = ["A", "B", "B", "B", "A"]
    assert cohens_kappa(ratings_a, ratings_b) == pytest.approx(0.32 / 0.52)


def test_cohens_kappa_perfect_agreement_with_real_variance_is_one():
    ratings_a = ["A", "B", "A", "B"]
    ratings_b = ["A", "B", "A", "B"]
    assert cohens_kappa(ratings_a, ratings_b) == pytest.approx(1.0)


def test_cohens_kappa_no_variance_at_all_returns_none_not_a_fabricated_one():
    """Both raters gave the same single category to every item -- there was
    no disagreement possible to explain, so kappa is undefined, not a
    fabricated "perfect agreement" 1.0."""
    ratings_a = ["A", "A", "A"]
    ratings_b = ["A", "A", "A"]
    assert cohens_kappa(ratings_a, ratings_b) is None


def test_cohens_kappa_no_items_returns_none():
    assert cohens_kappa([], []) is None


def test_cohens_kappa_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        cohens_kappa(["A", "B"], ["A"])


def test_cohens_kappa_worse_than_chance_can_be_negative():
    # Deliberately anti-correlated ratings -- kappa should come back negative.
    ratings_a = ["A", "A", "B", "B"]
    ratings_b = ["B", "B", "A", "A"]
    result = cohens_kappa(ratings_a, ratings_b)
    assert result is not None
    assert result < 0
