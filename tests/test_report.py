"""Post-session report: window averaging, per-question splitting, scores
matching fusion.py exactly, honest handling of missing data, HTML escaping,
and session listing. Real EventLog on tmp_path; no hardware."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.events import EventLog
from analyzer.feedback import generate_tips
from analyzer.fusion import FusionInputs, compute_scores
from analyzer.questions import mark_question
from analyzer.report import build_report, list_sessions, load_report, render_html


def make_log(tmp_path) -> EventLog:
    log = EventLog(tmp_path)
    log.session_start(t0=0.0)
    return log


def seed_two_questions(log):
    """Q1 = [10, 60), Q2 = [60, end). Q1 has poor eye contact and a
    hedge-heavy answer; Q2 has good eye contact."""
    mark_question(log, "Describe your experience with databases", 10.0)
    mark_question(log, "What are your weaknesses", 60.0)
    for t, v in ((20.0, 0.2), (40.0, 0.4)):
        log.metric("video", "camera_facing", v, t=t)
    for t, v in ((70.0, 0.9), (90.0, 0.9)):
        log.metric("video", "camera_facing", v, t=t)
    log.metric("audio_candidate", "wpm", 140.0, t=50.0)
    log.metric("text", "hedge_rate", 6.0, t=50.0)
    log.metric("text", "sentiment", 0.5, t=50.0)
    for text, t in (("I", 15.0), ("know", 15.5), ("databases", 16.0), ("well", 16.5),
                    ("Patience", 65.0), ("mostly", 66.0)):
        log.word(text, t, t + 0.3)
    log.event("audio_candidate", "long_pause", t_start=30.0, t_end=34.0, detail={"duration": 4.0})
    log.event("video", "gaze_away", t_start=75.0, t_end=80.0, detail={"camera_facing_fraction": 0.1})
    log.event("video", "calibration_done", t_start=8.0)   # bookkeeping, must not be flagged


def test_question_windows_average_only_their_own_readings(tmp_path):
    log = make_log(tmp_path)
    seed_two_questions(log)
    r = build_report(log, "s1")
    q1, q2 = r.questions
    assert q1.inputs.camera_facing == pytest.approx(0.3)     # mean(0.2, 0.4)
    assert q2.inputs.camera_facing == pytest.approx(0.9)
    assert r.overall.inputs.camera_facing == pytest.approx((0.2 + 0.4 + 0.9 + 0.9) / 4)
    assert q1.inputs.wpm == 140.0 and q2.inputs.wpm is None   # never borrowed from another window


def test_scores_and_tips_are_exactly_fusions_and_feedbacks_for_the_window_means(tmp_path):
    log = make_log(tmp_path)
    seed_two_questions(log)
    q1 = build_report(log).questions[0]
    expected_inputs = FusionInputs(camera_facing=0.3, wpm=140.0, hedge_rate=6.0)
    assert q1.scores.engagement.value == compute_scores(expected_inputs).engagement.value
    assert q1.scores.confidence.coverage == compute_scores(expected_inputs).confidence.coverage
    assert [t.text for t in q1.tips] == [t.text for t in generate_tips(expected_inputs)]


def test_missing_data_stays_none_not_a_default(tmp_path):
    log = make_log(tmp_path)
    seed_two_questions(log)
    q2 = build_report(log).questions[1]
    assert q2.scores.fluency.value is None       # q2 has no audio metrics at all
    assert q2.inputs.hedge_rate is None


def test_transcript_relevance_and_boundaries(tmp_path):
    log = make_log(tmp_path)
    seed_two_questions(log)
    q1, q2 = build_report(log).questions
    assert q1.answer_text == "I know databases well" and q1.word_count == 4
    assert q2.answer_text == "Patience mostly"
    # q1 keywords {describe, experience, databases}: only "databases" is in the answer
    assert q1.relevance == pytest.approx(1 / 3)
    assert q1.question_text.startswith("Describe")
    assert q1.sentiment == pytest.approx(0.5) and q2.sentiment is None


def test_relevance_not_scored_without_wording_or_answer(tmp_path):
    log = make_log(tmp_path)
    mark_question(log, "", 5.0)                        # no wording
    mark_question(log, "What is your name", 20.0)      # wording but no answer
    log.word("hello", 6.0, 6.3)
    q1, q2 = build_report(log).questions
    assert q1.relevance is None and q2.relevance is None


def test_flagged_moments_are_chronological_and_tagged_with_their_question(tmp_path):
    log = make_log(tmp_path)
    seed_two_questions(log)
    flagged = build_report(log).flagged
    assert [(f.t_start, f.question_id) for f in flagged] == [(30.0, 1), (75.0, 2)]
    assert flagged[0].text == "[00:30] Long pause (4.0s)"     # calibration_done excluded


def test_session_without_questions_still_reports_overall(tmp_path):
    log = make_log(tmp_path)
    log.metric("video", "camera_facing", 0.8, t=5.0)
    r = build_report(log)
    assert r.questions == [] and r.overall.inputs.camera_facing == 0.8
    assert "No questions were marked" in render_html(r)


def test_empty_session_does_not_crash_and_says_not_enough_data(tmp_path):
    r = build_report(make_log(tmp_path), "empty")
    assert r.duration == 0.0 and r.overall.scores.engagement.value is None
    page = render_html(r)
    assert "Not enough data yet" in page and "Nothing was flagged" in page


def test_duration_is_the_last_recorded_time(tmp_path):
    log = make_log(tmp_path)
    seed_two_questions(log)
    assert build_report(log).duration == 90.0


def test_html_escapes_user_supplied_text(tmp_path):
    log = make_log(tmp_path)
    mark_question(log, "<script>alert('q')</script>", 1.0)
    log.word("<img src=x onerror=alert(1)>", 2.0, 2.3)
    page = render_html(build_report(log, "<b>name</b>"))
    assert "<script>alert" not in page and "<img src=x" not in page and "<b>name</b>" not in page
    assert "&lt;script&gt;" in page


def test_html_is_self_contained_and_states_the_caveats(tmp_path):
    log = make_log(tmp_path)
    seed_two_questions(log)
    page = render_html(build_report(log, "s1"))
    assert "http://" not in page and "https://" not in page and "<script" not in page
    assert "not for hiring" in page.lower() or "not for hiring, grading" in page
    assert "Question 1" in page and "Question 2" in page
    assert "of usual signals" in page


def test_low_coverage_is_visibly_marked(tmp_path):
    log = make_log(tmp_path)
    log.metric("video", "blink_rate", 15.0, t=1.0)    # confidence gets only blink_rate: coverage 0.15
    page = render_html(build_report(log))
    assert "⚠️ built from 15% of usual signals" in page


def test_list_sessions_newest_first_and_ignores_folders_without_a_log(tmp_path):
    for name in ("20260101_100000", "20260301_100000", "20260201_100000"):
        EventLog(tmp_path / name)
    (tmp_path / "not_a_session").mkdir()
    assert [p.name for p in list_sessions(tmp_path)] == ["20260301_100000", "20260201_100000", "20260101_100000"]
    assert list_sessions(tmp_path / "missing") == []


def test_load_report_reads_a_saved_session_from_disk(tmp_path):
    log = make_log(tmp_path / "20260101_100000")
    log.metric("video", "camera_facing", 0.5, t=3.0)
    r = load_report(tmp_path / "20260101_100000")
    assert r.session_name == "20260101_100000" and r.overall.inputs.camera_facing == 0.5


def test_report_includes_combined_flags_and_renders_them(tmp_path):
    log = make_log(tmp_path)
    mark_question(log, "Why this role", 0.0)
    for t in range(0, 120, 2):
        log.metric("video", "camera_facing", 0.2 if 60 <= t < 90 else 0.9, t=float(t))
    for t in range(0, 120, 10):
        log.metric("text", "hedge_rate", 6.0 if 60 <= t < 90 else 0.5, t=float(t))
    r = build_report(log)
    assert [(f.t_start, f.t_end, f.question_id) for f in r.combined_flags] == [(60.0, 90.0, 1)]
    page = render_html(r)
    assert "01:00–01:30 — hedged phrasing and looking away from the camera together" in page
    assert "Several signals together during this answer" in page


def test_report_without_combined_flags_says_so(tmp_path):
    log = make_log(tmp_path)
    log.metric("video", "camera_facing", 0.9, t=5.0)
    assert "No stretch had several signals worse than usual at once." in render_html(build_report(log))
