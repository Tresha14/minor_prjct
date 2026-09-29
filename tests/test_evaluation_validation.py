"""Tests for evaluation/validation.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.validation import correlate_with_human_ratings, rater_agreement


# ---------------------------------------------------------------- rater_agreement

def test_rater_agreement_reliable_above_default_threshold():
    # kappa = 0.32/0.52 ~= 0.615 (same golden-value data as
    # test_evaluation_statistics.py) -- above the default 0.6 threshold.
    ratings_a = ["A", "A", "B", "B", "A"]
    ratings_b = ["A", "B", "B", "B", "A"]
    result = rater_agreement(ratings_a, ratings_b)
    assert result.kappa == pytest.approx(0.32 / 0.52)
    assert result.n_items == 5
    assert result.reliable is True


def test_rater_agreement_not_reliable_with_a_stricter_threshold():
    ratings_a = ["A", "A", "B", "B", "A"]
    ratings_b = ["A", "B", "B", "B", "A"]
    result = rater_agreement(ratings_a, ratings_b, threshold=0.7)
    assert result.reliable is False


def test_rater_agreement_undefined_kappa_gives_none_reliability_not_false():
    """Reliability being unknowable (kappa undefined) must not silently
    read as "unreliable" (False) -- those are different findings."""
    ratings_a = ["A", "A", "A"]
    ratings_b = ["A", "A", "A"]
    result = rater_agreement(ratings_a, ratings_b)
    assert result.kappa is None
    assert result.reliable is None


# ---------------------------------------------------------------- correlate_with_human_ratings

def test_correlate_with_enough_sessions_has_no_warning_note():
    scores = list(range(1, 11))
    ratings = list(range(1, 11))
    result = correlate_with_human_ratings(scores, ratings)
    assert result.n == 10
    assert result.pearson_r == pytest.approx(1.0)
    assert result.spearman_rho == pytest.approx(1.0)
    assert result.note is None


def test_correlate_with_few_sessions_sets_a_warning_note_but_still_returns_numbers():
    scores = [1, 2, 3]
    ratings = [1, 2, 3]
    result = correlate_with_human_ratings(scores, ratings)
    assert result.n == 3
    assert result.pearson_r == pytest.approx(1.0)  # the number is still computed...
    assert result.note is not None  # ...but flagged as not yet trustworthy
    assert "n=3" in result.note


def test_correlate_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        correlate_with_human_ratings([1, 2, 3], [1, 2])
