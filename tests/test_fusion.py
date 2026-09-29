"""Tests for the fusion/scoring layer. This is the module a wrong answer
would matter most in, so it's tested more heavily than most: exact
hand-calculated ("golden value") checks, not just example-based sanity
checks, plus explicit monotonicity sweeps and missing-data handling.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.config import FUSION
from analyzer.events import EventLog
from analyzer.fusion import (
    FusionInputs, NORMALISABLE_METRICS, band_score, compute_from_log, compute_scores,
    identity01, linear_lower_is_better, weighted_average, write_scores,
)


# ---------------------------------------------------------------- linear_lower_is_better

def test_linear_lower_is_better_boundaries():
    assert linear_lower_is_better(0.0, good=0.15, bad=0.45) == 1.0
    assert linear_lower_is_better(0.15, good=0.15, bad=0.45) == 1.0   # at good -> full score
    assert linear_lower_is_better(0.45, good=0.15, bad=0.45) == 0.0   # at bad -> zero
    assert linear_lower_is_better(0.6, good=0.15, bad=0.45) == 0.0    # beyond bad -> still zero


def test_linear_lower_is_better_midpoint_exact():
    # (0.45 - 0.30) / (0.45 - 0.15) = 0.15 / 0.30 = 0.5 exactly
    assert linear_lower_is_better(0.30, good=0.15, bad=0.45) == pytest.approx(0.5)


def test_linear_lower_is_better_rejects_bad_ordering():
    with pytest.raises(ValueError):
        linear_lower_is_better(0.2, good=0.5, bad=0.5)
    with pytest.raises(ValueError):
        linear_lower_is_better(0.2, good=0.6, bad=0.4)


# ---------------------------------------------------------------- band_score

def test_band_score_boundaries():
    # floor=60, good_low=110, good_high=165, ceiling=220
    assert band_score(60, 60, 110, 165, 220) == 0.0    # at floor
    assert band_score(30, 60, 110, 165, 220) == 0.0    # below floor
    assert band_score(220, 60, 110, 165, 220) == 0.0   # at ceiling
    assert band_score(300, 60, 110, 165, 220) == 0.0   # above ceiling
    assert band_score(110, 60, 110, 165, 220) == 1.0   # at good_low
    assert band_score(165, 60, 110, 165, 220) == 1.0   # at good_high
    assert band_score(137.5, 60, 110, 165, 220) == 1.0  # mid good band


def test_band_score_ramp_midpoints_exact():
    # ramp-up midpoint: (85-60)/(110-60) = 25/50 = 0.5 exactly
    assert band_score(85, 60, 110, 165, 220) == pytest.approx(0.5)
    # ramp-down midpoint: (220-192.5)/(220-165) = 27.5/55 = 0.5 exactly
    assert band_score(192.5, 60, 110, 165, 220) == pytest.approx(0.5)


def test_band_score_rejects_bad_ordering():
    with pytest.raises(ValueError):
        band_score(100, floor=110, good_low=60, good_high=165, ceiling=220)  # floor > good_low
    with pytest.raises(ValueError):
        band_score(100, floor=60, good_low=200, good_high=165, ceiling=220)  # good_low > good_high
    with pytest.raises(ValueError):
        band_score(100, floor=60, good_low=110, good_high=230, ceiling=220)  # good_high > ceiling


def test_band_score_is_unimodal_across_the_whole_range():
    """Rises to 1.0, holds, falls back to 0.0 -- never a second peak, never
    dips inside the "good" plateau. This is the property that makes it a
    correct model of "too slow AND too fast are both bad"."""
    floor, good_low, good_high, ceiling = 60, 110, 165, 220
    xs = [floor + i * (ceiling - floor) / 200 for i in range(201)]
    ys = [band_score(x, floor, good_low, good_high, ceiling) for x in xs]
    peak = max(ys)
    assert peak == pytest.approx(1.0)
    peak_i = ys.index(peak)
    last_peak_i = len(ys) - 1 - ys[::-1].index(peak)
    # non-decreasing up to the first peak, non-increasing after the last peak
    rising = ys[:peak_i + 1]
    falling = ys[last_peak_i:]
    assert all(a <= b + 1e-9 for a, b in zip(rising, rising[1:]))
    assert all(a >= b - 1e-9 for a, b in zip(falling, falling[1:]))


# ---------------------------------------------------------------- identity01

def test_identity01_clamps():
    assert identity01(0.5) == 0.5
    assert identity01(-0.2) == 0.0
    assert identity01(1.3) == 1.0
    assert identity01(0.0) == 0.0
    assert identity01(1.0) == 1.0


# ---------------------------------------------------------------- weighted_average

def test_weighted_average_basic():
    avg, drivers = weighted_average({"a": (1.0, 0.5), "b": (1.0, 1.0)})
    assert avg == pytest.approx(0.75)
    assert drivers == {"a": 0.5, "b": 1.0}


def test_weighted_average_drops_missing_and_renormalises():
    avg, drivers = weighted_average({"a": (1.0, 0.5), "b": (2.0, None)})
    assert avg == pytest.approx(0.5)   # only "a" counted, its own weight cancels out
    assert drivers == {"a": 0.5}


def test_weighted_average_all_missing_returns_none():
    avg, drivers = weighted_average({"a": (1.0, None), "b": (2.0, None)})
    assert avg is None
    assert drivers == {}


def test_weighted_average_zero_total_weight_returns_none():
    # the only present entry has weight 0 -- a valid score with no say in
    # the outcome should not silently become "the whole average"
    avg, drivers = weighted_average({"a": (0.0, 0.9)})
    assert avg is None
    assert drivers == {}


# ---------------------------------------------------------------- compute_scores: golden values

def test_compute_scores_golden_values_all_inputs_present():
    """Every input set to a value inside its own "good" range except
    camera_facing (0.9, deliberately not perfect) and hedge_rate (absent,
    to also exercise the missing-input path in the same test). Expected
    values hand-calculated in this test's PR / commit message and in
    tests/test_fusion.py's own comments -- do not "fix" this test to match
    a code change without redoing the hand calculation.

    engagement = 0.45*0.9 + 0.20*1.0 + 0.20*1.0 + 0.15*1.0 = 0.955 -> 95.5
    fluency    = 0.40*1.0 + 0.40*1.0 + 0.20*1.0            = 1.000 -> 100.0
    confidence = (0.30*1.0 + 0.30*0.9 + 0.15*1.0) / 0.75    = 0.960 -> 96.0
                 (hedge_rate's 0.25 weight dropped: absent input)
    """
    inputs = FusionInputs(
        camera_facing=0.9,
        pose_stability=1.0,   # <= good (2.0) -> normalised 1.0
        wpm=140.0,            # inside good band [110, 165] -> normalised 1.0
        pause_rate=0.10,      # <= good (0.15) -> normalised 1.0
        filler_rate=0.5,      # <= good (1.0) -> normalised 1.0
        blink_rate=15.0,      # inside good band [10, 25] -> normalised 1.0
        hedge_rate=None,
    )
    result = compute_scores(inputs)

    assert result.engagement.value == pytest.approx(95.5)
    assert result.fluency.value == pytest.approx(100.0)
    assert result.confidence.value == pytest.approx(96.0)

    # drivers must trace back to exactly the metrics that were present
    assert set(result.engagement.drivers) == {"camera_facing", "pause_rate", "wpm", "pose_stability"}
    assert set(result.confidence.drivers) == {"camera_facing", "pause_rate", "blink_rate"}  # no hedge_rate


def test_compute_scores_all_missing_gives_none_not_a_fabricated_default():
    result = compute_scores(FusionInputs())
    assert result.engagement.value is None
    assert result.fluency.value is None
    assert result.confidence.value is None
    assert result.engagement.drivers == {}


def test_compute_scores_partial_input_renormalises_correctly():
    """Only camera_facing is known. engagement and confidence both use it
    (partially) and should renormalise to just that input; fluency doesn't
    use camera_facing at all and must come back None, not a wrong number.
    """
    inputs = FusionInputs(camera_facing=0.6)
    result = compute_scores(inputs)

    assert result.engagement.value == pytest.approx(60.0)
    assert result.confidence.value == pytest.approx(60.0)
    assert result.fluency.value is None


# ---------------------------------------------------------------- monotonicity

def _sweep(values, build_inputs):
    return [compute_scores(build_inputs(v)) for v in values]


def test_engagement_never_decreases_as_camera_facing_improves():
    values = [i / 20 for i in range(21)]  # 0.0 .. 1.0
    scores = _sweep(values, lambda v: FusionInputs(
        camera_facing=v, pose_stability=1.0, wpm=140.0, pause_rate=0.10,
    ))
    vals = [s.engagement.value for s in scores]
    assert all(a <= b + 1e-9 for a, b in zip(vals, vals[1:]))
    assert vals[0] < vals[-1]  # not flat -- camera_facing actually has an effect


def test_engagement_never_increases_as_pause_rate_worsens():
    values = [i / 20 for i in range(21)]  # 0.0 .. 1.0 raw pause_rate (higher = worse)
    scores = _sweep(values, lambda v: FusionInputs(
        camera_facing=1.0, pose_stability=1.0, wpm=140.0, pause_rate=v,
    ))
    vals = [s.engagement.value for s in scores]
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))
    assert vals[0] > vals[-1]


def test_fluency_never_increases_as_filler_rate_worsens():
    values = [i * 0.5 for i in range(21)]  # 0 .. 10 fillers/100 words
    scores = _sweep(values, lambda v: FusionInputs(
        pause_rate=0.10, wpm=140.0, filler_rate=v,
    ))
    vals = [s.fluency.value for s in scores]
    assert all(a >= b - 1e-9 for a, b in zip(vals, vals[1:]))
    assert vals[0] > vals[-1]


def test_confidence_never_decreases_as_camera_facing_improves():
    values = [i / 20 for i in range(21)]
    scores = _sweep(values, lambda v: FusionInputs(
        camera_facing=v, pause_rate=0.10, blink_rate=15.0,
    ))
    vals = [s.confidence.value for s in scores]
    assert all(a <= b + 1e-9 for a, b in zip(vals, vals[1:]))
    assert vals[0] < vals[-1]


# ---------------------------------------------------------------- config sanity

def test_all_configured_weights_reference_real_normalisable_metrics():
    """Guards against a typo in FUSION.*_weights (e.g. "pause_rte") that
    would otherwise silently contribute nothing to a score forever,
    with no error and no test failure to catch it -- so we check the
    config itself, not just fusion.py's behaviour.
    """
    for weights in (FUSION.engagement_weights, FUSION.fluency_weights, FUSION.confidence_weights):
        unknown = set(weights) - NORMALISABLE_METRICS
        assert not unknown, f"weights reference unknown metric(s): {unknown}"
        assert all(w > 0 for w in weights.values()), "a zero/negative configured weight is almost certainly a mistake"


def test_fusion_inputs_fields_match_normalisable_metrics():
    """The inverse check: every field FusionInputs declares must actually
    have a normalisation rule, so adding a field without wiring it up
    would be caught here rather than silently doing nothing."""
    field_names = set(FusionInputs.__dataclass_fields__.keys())
    assert field_names == NORMALISABLE_METRICS


# ---------------------------------------------------------------- compute_from_log / write_scores (integration)

def test_compute_from_log_uses_the_most_recent_reading(tmp_path):
    log = EventLog(tmp_path)
    log.session_start(t0=0.0)
    # two readings each; compute_from_log must pick the later one
    log.metric("audio_candidate", "wpm", 80.0, t=1.0)
    log.metric("audio_candidate", "wpm", 140.0, t=5.0)
    log.metric("audio_candidate", "pause_rate", 0.5, t=1.0)
    log.metric("audio_candidate", "pause_rate", 0.10, t=5.0)
    log.metric("video", "camera_facing", 0.95, t=5.0)

    result = compute_from_log(log)

    # if it had used the stale (t=1.0) readings instead, wpm=80 would be
    # outside the good band and pause_rate=0.5 would be heavily penalised --
    # so an unexpectedly high score here proves the *latest* row was used.
    assert result.engagement.value > 90.0


def test_write_scores_persists_to_the_log(tmp_path):
    log = EventLog(tmp_path)
    log.session_start(t0=0.0)
    inputs = FusionInputs(camera_facing=0.9, pose_stability=1.0, wpm=140.0, pause_rate=0.10,
                           filler_rate=0.5, blink_rate=15.0)
    result = compute_scores(inputs)

    write_scores(log, result, t=42.0)

    assert log.recent_metrics("engagement_score") == [(42.0, result.engagement.value)]
    assert log.recent_metrics("fluency_score") == [(42.0, result.fluency.value)]
    assert log.recent_metrics("confidence_score") == [(42.0, result.confidence.value)]


def test_write_scores_never_writes_a_none_value(tmp_path):
    log = EventLog(tmp_path)
    log.session_start(t0=0.0)
    result = compute_scores(FusionInputs())  # everything missing -> all three None

    write_scores(log, result, t=1.0)

    assert log.recent_metrics("engagement_score") == []
    assert log.recent_metrics("fluency_score") == []
    assert log.recent_metrics("confidence_score") == []
