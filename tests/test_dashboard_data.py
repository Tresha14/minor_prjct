import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.dashboard_data import (
    format_event, format_seconds, latest_value, metric_delta_label, summarize_events,
)


def test_latest_value_returns_most_recent():
    rows = [(1.0, 10.0), (2.0, 20.0), (3.0, 30.0)]
    assert latest_value(rows) == 30.0


def test_latest_value_empty_returns_none():
    assert latest_value([]) is None


def test_format_seconds():
    assert format_seconds(0) == "00:00"
    assert format_seconds(12.4) == "00:12"
    assert format_seconds(75.0) == "01:15"
    assert format_seconds(3661) == "61:01"


def test_format_seconds_clamps_negative():
    assert format_seconds(-5.0) == "00:00"


def test_format_event_long_pause():
    row = (12.0, 16.3, "audio_candidate", "long_pause", json.dumps({"duration": 4.3}))
    s = format_event(row)
    assert s == "[00:12] Long pause (4.3s)"


def test_format_event_gaze_away():
    row = (30.0, 34.0, "video", "gaze_away", json.dumps({"camera_facing_fraction": 0.12}))
    s = format_event(row)
    assert s == "[00:30] Looking away from camera (12% facing camera)"


def test_format_event_face_lost():
    row = (5.0, 8.0, "video", "face_lost", json.dumps({"duration": 3.0}))
    s = format_event(row)
    assert s == "[00:05] Not visible in frame (3.0s)"


def test_format_event_unknown_kind_falls_back_to_raw_name():
    row = (1.0, None, "video", "something_new", "{}")
    s = format_event(row)
    assert s == "[00:01] something_new"


def test_format_event_never_crashes_on_bad_json():
    row = (1.0, None, "video", "long_pause", "not valid json{{{")
    s = format_event(row)  # must not raise
    assert s.startswith("[00:01] Long pause")


def test_format_event_handles_none_detail():
    row = (1.0, None, "video", "calibration_done", None)
    s = format_event(row)
    assert s == "[00:01] Calibration complete"


def test_summarize_events_filters_by_kind():
    rows = [
        (10.0, None, "video", "calibration_done", "{}"),
        (5.0, 8.0, "audio_candidate", "long_pause", json.dumps({"duration": 3.0})),
        (2.0, 3.0, "video", "gaze_away", json.dumps({"camera_facing_fraction": 0.1})),
    ]
    out = summarize_events(rows, kinds={"long_pause", "gaze_away"})
    assert len(out) == 2
    assert all("Calibration" not in s for s in out)


def test_summarize_events_respects_limit():
    rows = [(float(i), None, "video", "long_pause", json.dumps({"duration": 1.0})) for i in range(10)]
    out = summarize_events(rows, limit=3)
    assert len(out) == 3


def test_metric_delta_label():
    assert metric_delta_label([(1.0, 100.0), (2.0, 110.0)]) == "+10.0"
    assert metric_delta_label([(1.0, 100.0), (2.0, 90.0)]) == "-10.0"
    assert metric_delta_label([(1.0, 100.0)]) is None
    assert metric_delta_label([]) is None
    assert metric_delta_label([(1.0, 100.0), (2.0, 100.0)]) is None  # no change
