"""Question markers, timestamp-based per-question segmentation, and the
end-of-session answer-relevance pass. Uses a real EventLog on tmp_path (it's
just SQLite) and no hardware."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.events import EventLog
from analyzer.questions import (
    QuestionMarker, build_segments, mark_question, read_markers, score_answer_relevance,
)
from analyzer.text import TextWorker


def make_log(tmp_path) -> EventLog:
    log = EventLog(tmp_path)
    log.session_start(t0=0.0)
    return log


def add_words(log, words_with_t):
    for text, t in words_with_t:
        log.word(text, t, t + 0.3)


def test_mark_question_numbers_sequentially_and_keeps_text(tmp_path):
    log = make_log(tmp_path)
    assert mark_question(log, "  Tell me about yourself  ", 5.0) == 1
    assert mark_question(log, "", 60.0) == 2
    markers = read_markers(log.recent_events())
    assert [(m.question_id, m.text, m.t_start) for m in markers] == [
        (1, "Tell me about yourself", 5.0), (2, "", 60.0),
    ]


def test_legacy_markers_without_detail_still_read(tmp_path):
    log = make_log(tmp_path)
    log.event("session", "question_marker", t_start=10.0)   # old app.py format
    log.event("session", "question_marker", t_start=20.0)
    markers = read_markers(log.recent_events())
    assert [(m.question_id, m.text) for m in markers] == [(1, ""), (2, "")]


def test_words_assigned_by_timestamp_boundaries():
    markers = [QuestionMarker(1, "q1", 10.0), QuestionMarker(2, "q2", 30.0)]
    words = [(5.0, 5.3, "before", None),      # before first marker -> no question
             (10.0, 10.3, "a", None),         # exactly on a boundary belongs to the NEW question
             (29.9, 30.2, "b", None),         # starts before q2 -> q1, even if it ends after
             (30.0, 30.3, "c", None),
             (55.0, 55.3, "d", None)]
    segs = build_segments(markers, words, t_end=60.0)
    assert [s.answer_text for s in segs] == ["a b", "c d"]
    assert [s.word_count for s in segs] == [2, 2]
    assert (segs[0].t_end, segs[1].t_end) == (30.0, 60.0)


def test_last_question_without_t_end_takes_rest_of_transcript():
    segs = build_segments([QuestionMarker(1, "q", 0.0)], [(1.0, 1.2, "x", None), (999.0, 999.2, "y", None)])
    assert segs[0].answer_text == "x y" and segs[0].t_end is None


def test_late_transcribed_word_lands_in_the_right_question(tmp_path):
    # The word is *spoken* at t=28 (question 1) but only *written* to the log
    # after question 2 was marked -- the bug write-time tagging would have.
    log = make_log(tmp_path)
    mark_question(log, "first", 10.0)
    mark_question(log, "second", 30.0)
    add_words(log, [("late", 28.0), ("next", 31.0)])
    segs = build_segments(read_markers(log.recent_events()), log.all_transcript(), t_end=40.0)
    assert segs[0].answer_text == "late" and segs[1].answer_text == "next"


def test_score_answer_relevance_writes_one_metric_per_answered_question(tmp_path):
    log = make_log(tmp_path)
    mark_question(log, "Describe your experience with databases", 0.0)
    mark_question(log, "", 30.0)                                   # no wording -> skipped
    mark_question(log, "What are your weaknesses", 60.0)           # no answer -> skipped
    add_words(log, [("I", 2.0), ("have", 2.3), ("databases", 2.6), ("experience", 3.0), ("daily", 3.4),
                    ("random", 35.0)])
    n = score_answer_relevance(log, TextWorker(log), t_end=70.0)
    assert n == 1
    rows = log.recent_metrics("answer_relevance")
    assert len(rows) == 1
    t, value = rows[0]
    assert t == 30.0
    # question keywords {describe, experience, databases}; the answer covers
    # "databases" and "experience" but not "describe" -> 2 / 3
    assert abs(value - 2 / 3) < 1e-9


def test_score_answer_relevance_with_no_questions_writes_nothing(tmp_path):
    log = make_log(tmp_path)
    add_words(log, [("hello", 1.0)])
    assert score_answer_relevance(log, TextWorker(log), t_end=5.0) == 0
    assert log.recent_metrics("answer_relevance") == []
