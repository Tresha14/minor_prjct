import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.events import EventLog


def test_metric_roundtrip(tmp_path):
    log = EventLog(tmp_path)
    log.session_start(t0=1000.0)
    log.metric("audio", "wpm", 120.0, t=5.0)
    log.metric("audio", "wpm", 130.0, t=6.0)

    rows = log.recent_metrics("wpm")
    assert rows == [(5.0, 120.0), (6.0, 130.0)]


def test_event_roundtrip(tmp_path):
    log = EventLog(tmp_path)
    log.session_start(t0=1000.0)
    log.event("audio", "long_pause", t_start=10.0, t_end=14.2, detail={"duration": 4.2})

    rows = log.recent_events()
    assert len(rows) == 1
    t_start, t_end, source, kind, detail = rows[0]
    assert (t_start, t_end, source, kind) == (10.0, 14.2, "audio", "long_pause")
    assert "4.2" in detail


def test_transcript_roundtrip(tmp_path):
    log = EventLog(tmp_path)
    log.session_start(t0=1000.0)
    log.word("hello", 0.0, 0.4, question_id=1)
    log.word("world", 0.4, 0.8, question_id=1)

    rows = log.all_transcript()
    assert [r[2] for r in rows] == ["hello", "world"]


def test_t0_persists_across_reopen(tmp_path):
    log = EventLog(tmp_path)
    log.session_start(t0=500.0)

    reopened = EventLog(tmp_path)
    assert reopened.get_t0() == 500.0
