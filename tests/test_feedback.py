"""Tests for the coaching-tip generator. Same testing philosophy as
tests/test_fusion.py: hand-calculated expected text/values, explicit
missing-data handling, and a config-consistency check so a metric added to
fusion.py can't silently end up with no tip support.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.config import FEEDBACK, FUSION
from analyzer.events import EventLog
from analyzer.feedback import METRICS_WITH_TIPS, Tip, generate_tips, generate_tips_from_log
from analyzer.fusion import NORMALISABLE_METRICS, FusionInputs


# ---------------------------------------------------------------- config consistency

def test_every_fusion_metric_has_a_tip_spec():
    """hedge_rate is included -- it's in NORMALISABLE_METRICS even though
    analyzer/text.py doesn't produce it yet (same forward-compatibility
    reasoning as fusion.py's own FusionInputs)."""
    assert METRICS_WITH_TIPS == NORMALISABLE_METRICS


def test_thresholds_are_ordered_sanely():
    assert 0.0 < FEEDBACK.needs_work_max < FEEDBACK.praise_min < 1.0


# ---------------------------------------------------------------- missing data

def test_no_data_at_all_returns_single_info_tip():
    tips = generate_tips(FusionInputs())
    assert len(tips) == 1
    assert tips[0].severity == "info"
    assert tips[0].metric is None


def test_missing_metric_never_produces_a_tip_for_it():
    """Only camera_facing is known and it's comfortably "good" (no tip
    either way); every other metric is None and must be silently skipped,
    not guessed at."""
    inputs = FusionInputs(camera_facing=0.95)
    tips = generate_tips(inputs)
    metrics_mentioned = {t.metric for t in tips}
    assert "wpm" not in metrics_mentioned
    assert "pause_rate" not in metrics_mentioned
    assert "filler_rate" not in metrics_mentioned
    assert "hedge_rate" not in metrics_mentioned
    assert "pose_stability" not in metrics_mentioned
    assert "blink_rate" not in metrics_mentioned


def test_all_present_but_all_comfortable_gives_the_nothing_to_flag_info_tip():
    """Every value hand-picked to normalise to roughly 0.5-0.6 -- strictly
    between needs_work_max (0.40) and praise_min (0.85) -- so NEITHER a
    needs_work nor a praise tip should fire for any of them. (Values near
    the "good" end of their range, like camera_facing=0.9 or wpm=140,
    normalise close to 1.0 and would wrongly trigger praise here -- see
    the bug this test used to have, caught by hand-verifying against
    fusion.py's own normalisation before trusting it.)
    """
    inputs = FusionInputs(
        camera_facing=0.6,      # identity01 -> 0.6
        pose_stability=6.0,     # good=2.0, bad=12.0 -> 0.6
        wpm=90.0,               # floor=60, good_low=110 -> (90-60)/(110-60) = 0.6
        pause_rate=0.27,        # good=0.15, bad=0.45 -> 0.6
        filler_rate=3.8,        # good=1.0, bad=8.0 -> 0.6
        blink_rate=6.0,         # floor=2, good_low=10 -> (6-2)/(10-2) = 0.5
        hedge_rate=4.6,         # good=1.0, bad=10.0 -> 0.6
    )
    tips = generate_tips(inputs)
    assert len(tips) == 1
    assert tips[0].severity == "info"
    assert "comfortable range" in tips[0].text


# ---------------------------------------------------------------- one-directional metrics: golden values

def test_camera_facing_needs_work_tip_has_correct_percentage():
    # 0.20 -> normalised = identity01(0.20) = 0.20 <= needs_work_max (0.40)
    inputs = FusionInputs(camera_facing=0.20)
    tips = generate_tips(inputs)
    assert len(tips) == 1
    tip = tips[0]
    assert tip.metric == "camera_facing"
    assert tip.severity == "needs_work"
    assert tip.raw_value == 0.20
    assert "20%" in tip.text


def test_camera_facing_praise_tip_has_correct_percentage():
    # 0.95 -> normalised = 0.95 >= praise_min (0.85)
    inputs = FusionInputs(camera_facing=0.95)
    tips = generate_tips(inputs)
    assert len(tips) == 1
    assert tips[0].severity == "praise"
    assert "95%" in tips[0].text


def test_pause_rate_needs_work_uses_percentage_wording():
    # pause_rate_good=0.15, pause_rate_bad=0.45 -> 0.45 normalises to 0.0
    inputs = FusionInputs(pause_rate=0.45)
    tips = generate_tips(inputs)
    assert len(tips) == 1
    assert tips[0].metric == "pause_rate"
    assert tips[0].severity == "needs_work"
    assert "45%" in tips[0].text


def test_filler_rate_praise_uses_one_decimal_rate_wording():
    # filler_rate_good=1.0 -> 0.3 is comfortably good
    inputs = FusionInputs(filler_rate=0.3)
    tips = generate_tips(inputs)
    assert tips[0].severity == "praise"
    assert "0.3" in tips[0].text


def test_hedge_rate_needs_work_mentions_hedging():
    # hedge_rate_bad=10.0
    inputs = FusionInputs(hedge_rate=10.0)
    tips = generate_tips(inputs)
    assert tips[0].metric == "hedge_rate"
    assert tips[0].severity == "needs_work"
    assert "hedging" in tips[0].text.lower()


def test_pose_stability_needs_work_mentions_head_movement():
    # pose_stability_bad=12.0
    inputs = FusionInputs(pose_stability=12.0)
    tips = generate_tips(inputs)
    assert tips[0].metric == "pose_stability"
    assert tips[0].severity == "needs_work"
    assert "head" in tips[0].text.lower()


# ---------------------------------------------------------------- band metrics: direction matters

def test_wpm_too_slow_and_too_fast_produce_different_wording():
    slow = generate_tips(FusionInputs(wpm=60.0))[0]   # at floor -> normalised 0.0
    fast = generate_tips(FusionInputs(wpm=220.0))[0]  # at ceiling -> normalised 0.0
    assert slow.severity == "needs_work"
    assert fast.severity == "needs_work"
    assert "slow" in slow.text.lower()
    assert "fast" in fast.text.lower()
    assert slow.text != fast.text


def test_wpm_praise_in_the_comfortable_band():
    inputs = FusionInputs(wpm=140.0)  # inside [good_low=110, good_high=165]
    tips = generate_tips(inputs)
    assert tips[0].severity == "praise"
    assert "140" in tips[0].text


def test_blink_rate_too_low_and_too_high_produce_different_wording():
    low = generate_tips(FusionInputs(blink_rate=2.0))[0]    # at floor -> 0.0
    high = generate_tips(FusionInputs(blink_rate=40.0))[0]  # at ceiling -> 0.0
    assert low.severity == "needs_work"
    assert high.severity == "needs_work"
    assert "low" in low.text.lower()
    assert "high" in high.text.lower()
    assert low.text != high.text


# ---------------------------------------------------------------- ordering and truncation

def test_needs_work_tips_are_sorted_worst_first():
    # camera_facing=0.0 -> normalised 0.0 (worst); pause_rate=0.30 is
    # roughly midway between good(0.15) and bad(0.45) -> normalised 0.5,
    # which is above needs_work_max(0.4) so it produces NO tip at all --
    # use a genuinely bad pause_rate instead so both are needs_work but
    # camera_facing is clearly worse.
    inputs = FusionInputs(camera_facing=0.0, pause_rate=0.40)
    tips = generate_tips(inputs)
    assert [t.metric for t in tips] == ["camera_facing", "pause_rate"]


def test_praise_tips_come_after_needs_work_tips():
    inputs = FusionInputs(camera_facing=0.0, wpm=140.0)  # one bad, one praised
    tips = generate_tips(inputs)
    severities = [t.severity for t in tips]
    assert severities == ["needs_work", "praise"]


def test_result_never_exceeds_max_tips():
    # all seven metrics pushed into needs_work territory at once
    inputs = FusionInputs(
        camera_facing=0.0, pose_stability=20.0, wpm=250.0, pause_rate=0.9,
        filler_rate=20.0, blink_rate=1.0, hedge_rate=20.0,
    )
    tips = generate_tips(inputs)
    assert len(tips) == FEEDBACK.max_tips


# ---------------------------------------------------------------- generate_tips_from_log (integration)

def test_generate_tips_from_log_uses_the_most_recent_reading(tmp_path):
    log = EventLog(tmp_path)
    log.session_start(t0=0.0)
    log.metric("video", "camera_facing", 0.10, t=1.0)   # stale, would be needs_work
    log.metric("video", "camera_facing", 0.95, t=5.0)   # latest -> praise

    tips = generate_tips_from_log(log)

    assert len(tips) == 1
    assert tips[0].metric == "camera_facing"
    assert tips[0].severity == "praise"


def test_generate_tips_from_log_with_no_metrics_yet_gives_info_tip(tmp_path):
    log = EventLog(tmp_path)
    log.session_start(t0=0.0)
    tips = generate_tips_from_log(log)
    assert len(tips) == 1
    assert tips[0].severity == "info"
