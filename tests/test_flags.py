"""Combined-flag detection: several signals worse than the speaker's OWN
baseline in the same window. Scenarios are built so each window's outcome
can be worked out by hand (see comments)."""
import sys
from collections import namedtuple
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.flags import CombinedFlag, find_combined_flags
from analyzer.fusion import band_score, normalise_metric
from analyzer.config import FUSION

Seg = namedtuple("Seg", "question_id t_start t_end")


def series(fn, start=0, stop=120, step=2):
    return [(float(t), fn(t)) for t in range(start, stop, step)]


def bad_between(lo, hi, good, bad):
    return lambda t: bad if lo <= t < hi else good


# Healthy 120 s session: eye contact 0.9, hedges 0.5/100 words. Normalised
# camera 0.9 / hedge 1.0 are the baselines (medians).
def healthy_and_bad(lo, hi):
    return {
        "camera_facing": series(bad_between(lo, hi, 0.9, 0.2), step=2),   # bad -> normalised 0.2
        "hedge_rate": series(bad_between(lo, hi, 0.5, 6.0), step=10),     # bad -> normalised 0.444
    }


def test_two_signals_bad_together_are_flagged_as_one_span():
    # Window (60,90): camera all 0.2 (norm .2, drop .7), hedges 6.0 (norm .444 <= .6, drop .56) -> both fire.
    # Window (45,75): camera mean .55 fires, hedge mean (1+.444+.444)/3=.63 > .6 does not -> only one signal.
    flags = find_combined_flags(healthy_and_bad(60, 90), [], duration=120.0)
    assert [(f.t_start, f.t_end, f.signals) for f in flags] == [(60.0, 90.0, ["hedge_rate", "camera_facing"])]
    assert flags[0].text == "01:00–01:30 — hedged phrasing and looking away from the camera together"


def test_a_single_deviating_signal_is_never_flagged():
    metrics = {"camera_facing": healthy_and_bad(60, 90)["camera_facing"],
               "hedge_rate": series(lambda t: 0.5, step=10)}
    assert find_combined_flags(metrics, [], duration=120.0) == []


def test_long_pause_event_counts_as_a_signal():
    metrics = {"camera_facing": healthy_and_bad(60, 90)["camera_facing"]}
    events = [(70.0, 74.0, "audio_candidate", "long_pause", "{}")]
    flags = find_combined_flags(metrics, events, duration=120.0)
    # windows (45,75) and (60,90) both hold the pause and a camera drop -> merged
    assert [(f.t_start, f.t_end, f.signals) for f in flags] == [(45.0, 90.0, ["long_pause", "camera_facing"])]


def test_long_pause_alone_is_not_flagged():
    events = [(70.0, 74.0, "audio_candidate", "long_pause", "{}")]
    assert find_combined_flags({"camera_facing": series(lambda t: 0.9)}, events, duration=120.0) == []


def test_relative_to_own_baseline_uniformly_weak_session_is_not_flagged():
    metrics = {"camera_facing": series(lambda t: 0.2), "hedge_rate": series(lambda t: 6.0, step=10)}
    assert find_combined_flags(metrics, [], duration=120.0) == []


def test_drop_must_also_reach_the_needs_work_side():
    # camera 1.0 -> 0.7 and hedge norm 1.0 -> 0.7: each drops 0.3 (>= .25) but 0.7 > norm_max .6 -> no flag
    metrics = {"camera_facing": series(bad_between(60, 90, 1.0, 0.7)),
               "hedge_rate": series(bad_between(60, 90, 0.5, 3.7), step=10)}
    assert find_combined_flags(metrics, [], duration=120.0) == []


def test_signal_without_enough_baseline_readings_cannot_fire():
    metrics = {"camera_facing": [(65.0, 0.2), (70.0, 0.2)]}           # only 2 readings < min_baseline_readings
    events = [(70.0, 74.0, "audio_candidate", "long_pause", "{}")]
    assert find_combined_flags(metrics, events, duration=120.0) == []


def test_separate_episodes_stay_separate():
    def cam(t): return 0.2 if (t < 25 or t >= 95) else 0.9
    def hed(t): return 6.0 if (t < 25 or t >= 95) else 0.5
    metrics = {"camera_facing": series(cam, step=2), "hedge_rate": series(hed, step=5)}
    flags = find_combined_flags(metrics, [], duration=120.0)
    assert [(f.t_start, f.t_end) for f in flags] == [(0.0, 30.0), (90.0, 120.0)]


def test_flag_is_tagged_with_the_question_it_overlaps_most():
    segs = [Seg(1, 0.0, 60.0), Seg(2, 60.0, None)]
    flags = find_combined_flags(healthy_and_bad(60, 90), [], duration=120.0, segments=segs)
    assert flags[0].question_id == 2
    flags = find_combined_flags(healthy_and_bad(60, 90), [], duration=120.0, segments=[Seg(7, 0.0, 200.0)])
    assert flags[0].question_id == 7
    assert find_combined_flags(healthy_and_bad(60, 90), [], duration=120.0)[0].question_id is None


def test_nothing_to_flag_in_empty_or_zero_length_sessions():
    assert find_combined_flags({}, [], duration=0.0) == []
    assert find_combined_flags({}, [], duration=90.0) == []


def test_short_session_gets_a_single_window():
    metrics = {"camera_facing": series(lambda t: 0.2, stop=20)}
    assert find_combined_flags(metrics, [], duration=20.0) == []      # no crash, no baseline drop


def test_flag_text_joins_one_two_and_three_signals():
    assert CombinedFlag(0, 30, ["hedge_rate"]).text == "00:00–00:30 — hedged phrasing together"
    assert CombinedFlag(75, 105, ["long_pause", "camera_facing", "wpm"]).text == (
        "01:15–01:45 — a long pause, looking away from the camera and an unusual speaking pace together")


def test_normalise_metric_is_the_same_transform_fusion_uses():
    assert normalise_metric("wpm", 150.0) == band_score(150.0, FUSION.wpm_floor, FUSION.wpm_good_low,
                                                        FUSION.wpm_good_high, FUSION.wpm_ceiling)
    assert normalise_metric("camera_facing", 0.9) == 0.9
    with pytest.raises(ValueError):
        normalise_metric("not_a_metric", 1.0)
