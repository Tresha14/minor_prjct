"""Tests for evaluation/ablation.py. Uses analyzer.config.FUSION's real
weight dicts (so the test proves this module stays wired to the actual
project config) with hand-picked driver values whose weighted averages are
computed by hand in each test's comment.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.config import FUSION
from evaluation.ablation import ablate_across_sessions, ablate_one_metric, average_impact


def test_ablate_one_metric_golden_value():
    # engagement_weights = {camera_facing:.45, pause_rate:.20, wpm:.20, pose_stability:.15}
    # baseline = .45*0.9 + .20*1.0 + .20*1.0 + .15*1.0 = 0.955
    # without camera_facing: remaining weight .55, weighted sum .20+.20+.15=.55 -> average 1.0
    # impact = 0.955 - 1.0 = -0.045 (removing a below-1.0 driver raises the renormalised average)
    drivers = {"camera_facing": 0.9, "pause_rate": 1.0, "wpm": 1.0, "pose_stability": 1.0}
    result = ablate_one_metric(drivers, FUSION.engagement_weights, "camera_facing")

    assert result.baseline_score == pytest.approx(0.955)
    assert result.without_metric_score == pytest.approx(1.0)
    assert result.impact == pytest.approx(0.955 - 1.0)


def test_ablate_one_metric_positive_impact_when_removed_metric_was_the_best_one():
    # Removing the highest-scoring driver should lower the renormalised
    # average -> a positive impact (baseline > without).
    drivers = {"camera_facing": 1.0, "pause_rate": 0.5, "wpm": 0.5, "pose_stability": 0.5}
    result = ablate_one_metric(drivers, FUSION.engagement_weights, "camera_facing")
    assert result.impact > 0


def test_ablate_one_metric_raises_if_metric_not_in_this_sessions_drivers():
    drivers = {"pause_rate": 1.0, "wpm": 1.0}  # camera_facing never observed this session
    with pytest.raises(ValueError):
        ablate_one_metric(drivers, FUSION.engagement_weights, "camera_facing")


def test_ablate_across_sessions_skips_sessions_missing_the_metric():
    session_with = {"camera_facing": 0.9, "pause_rate": 1.0, "wpm": 1.0, "pose_stability": 1.0}
    session_without = {"pause_rate": 1.0, "wpm": 1.0}  # no camera_facing this session

    results = ablate_across_sessions(
        [session_with, session_without], FUSION.engagement_weights, "camera_facing",
    )

    assert len(results) == 1  # session_without was skipped, not fabricated a result for
    assert results[0].baseline_score == pytest.approx(0.955)


def test_average_impact_over_multiple_sessions():
    sessions = [
        {"camera_facing": 0.9, "pause_rate": 1.0, "wpm": 1.0, "pose_stability": 1.0},  # impact = -0.045
        {"camera_facing": 1.0, "pause_rate": 0.5, "wpm": 0.5, "pose_stability": 0.5},  # impact > 0
    ]
    results = ablate_across_sessions(sessions, FUSION.engagement_weights, "camera_facing")
    avg = average_impact(results)
    assert avg is not None
    # sanity: average sits between the two individual impacts
    impacts = [r.impact for r in results]
    assert min(impacts) <= avg <= max(impacts)


def test_average_impact_with_no_results_is_none():
    assert average_impact([]) is None
