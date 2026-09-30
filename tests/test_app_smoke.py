"""Smoke test for app.py using Streamlit's own AppTest framework: verifies
the dashboard loads without an exception in its idle (session not started)
state, and that the key controls are present. Deliberately does NOT click
"Start session" -- that loads the real GPU speech model and opens a real
camera/microphone, which isn't appropriate for an automated test run. That
path is instead verified manually against real hardware (see README.md's
status section for how, and app.py's own docstring).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_app_loads_without_exception_in_idle_state():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    at.run(timeout=30)

    assert not at.exception, f"app.py raised on load: {at.exception}"


def test_start_and_stop_buttons_present_when_idle():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    at.run(timeout=30)

    labels = [b.label for b in at.button]
    assert any("Start session" in l for l in labels)
    assert not any("Stop session" in l for l in labels)  # not running yet, so no stop button


def test_mark_question_button_is_disabled_when_idle():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    at.run(timeout=30)

    mark_button = next(b for b in at.button if "Mark next question" in b.label)
    assert mark_button.disabled


class _FakeVideoWorker:
    calibration_seconds = 8.0

    def calibration_seconds_remaining(self, elapsed):
        return 0.0


class _FakeSession:
    """Just enough of SessionCapture for app.py's live view to render,
    backed by a real EventLog so the real fusion/feedback code runs."""

    def __init__(self, log, elapsed=60.0):
        import time
        self.log = log
        self._t0 = time.time() - elapsed
        self.solo_test_mode = True
        self.video_worker = _FakeVideoWorker()
        self.problems: list[str] = []
        self.candidate_source = None
        self.interviewer_source = None

    def audio_problems(self):
        return self.problems

    def video_problems(self):
        return []


def test_live_view_renders_scores_and_coaching_tips_from_seeded_log(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from analyzer.events import EventLog

    log = EventLog(tmp_path)
    log.session_start(t0=0.0)
    log.metric("video", "camera_facing", 0.20, t=10.0)     # poor -> a needs_work tip
    log.metric("audio_candidate", "wpm", 140.0, t=20.0)    # comfortable -> a praise tip
    log.metric("text", "hedge_rate", 3.0, t=20.0)

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    at.session_state["session"] = _FakeSession(log)
    at.session_state["face_tracker"] = None
    at.session_state["session_dir"] = str(tmp_path)
    at.run(timeout=30)

    assert not at.exception, f"live view raised: {at.exception}"
    text = " ".join(m.value for m in at.markdown)
    assert "Coaching tips (live)" in " ".join(s.value for s in at.subheader)
    assert "Camera-facing time was 20%" in text          # needs_work tip from the seeded reading
    assert "Speaking pace was 140 words/min" in text     # praise tip
    labels = {m.label for m in at.metric}
    assert "Hedges / 100 words" in labels


def test_review_mode_renders_a_saved_session_and_offers_download(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from analyzer.events import EventLog

    session_dir = tmp_path / "20260101_100000"
    log = EventLog(session_dir)
    log.session_start(t0=0.0)
    log.metric("video", "camera_facing", 0.6, t=5.0)

    import review_page   # app.py imports this same module object when it enters review mode
    monkeypatch.setattr(review_page, "PATHS", type("P", (), {"data_dir": str(tmp_path)})())

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    at.session_state["page_mode"] = "Review saved session"
    at.run(timeout=30)

    assert not at.exception, f"review mode raised: {at.exception}"
    assert any("Session Report" in t.value for t in at.title)
    assert at.selectbox[0].options  # the saved session is listed
    assert not any("Start session" in b.label for b in at.button)  # live controls not rendered in review mode


def test_live_view_shows_a_banner_when_an_audio_device_stopped(tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    from analyzer.events import EventLog

    log = EventLog(tmp_path)
    log.session_start(t0=0.0)
    session = _FakeSession(log)
    session.problems = ["Interviewer microphone stopped: RuntimeError: Error 0x8889000a"]

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    at.session_state["session"] = session
    at.session_state["face_tracker"] = None
    at.session_state["session_dir"] = str(tmp_path)
    at.run(timeout=30)

    assert not at.exception
    assert any("0x8889000a" in e.value for e in at.error)


class _FakeAudioSource:
    def __init__(self, level, alive=True):
        import time
        self.level = level
        self.last_chunk_time = time.time() if alive else None


class _FakeDetector:
    questions_marked = 2


def _render_live(tmp_path, *, solo, elapsed, cand=None, intr=None, detector=None, words=(), turn=None):
    from streamlit.testing.v1 import AppTest
    from analyzer.events import EventLog

    log = EventLog(tmp_path)
    log.session_start(t0=0.0)
    for i, w in enumerate(words):
        log.word(w, float(i), float(i) + 0.5)
    if turn:
        log.event("audio_interviewer", "interviewer_turn", t_start=5.0, t_end=8.0, detail={"text": turn})
    session = _FakeSession(log, elapsed=elapsed)
    session.solo_test_mode = solo
    session.candidate_source, session.interviewer_source = cand, intr
    session.question_detector = detector
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    at.session_state["session"] = session
    at.session_state["face_tracker"] = None
    at.session_state["session_dir"] = str(tmp_path)
    at.run(timeout=30)
    assert not at.exception, f"live view raised: {at.exception}"
    return " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def test_audio_channels_panel_shows_both_channels_and_what_you_said(tmp_path):
    pytest.importorskip("streamlit")
    text = _render_live(tmp_path, solo=False, elapsed=60, cand=_FakeAudioSource(0.0), intr=_FakeAudioSource(0.1),
                        detector=_FakeDetector(), words=["thanks", "for", "having", "me"],
                        turn="Tell me about yourself?")
    assert "You (interviewer mic)" in text and "hearing sound" in text       # your mic is heard
    assert 'You said: "Tell me about yourself?"' in text                      # and understood
    assert "Questions detected: 2" in text
    assert "never added to the candidate's transcript" in text               # why it isn't in the transcript
    assert "thanks for having me" in text                                     # candidate words shown


def test_audio_channels_panel_hints_when_nothing_plays_through_the_default_output(tmp_path):
    pytest.importorskip("streamlit")
    text = _render_live(tmp_path, solo=False, elapsed=60, cand=_FakeAudioSource(0.0), intr=_FakeAudioSource(0.1),
                        detector=_FakeDetector())
    assert "Nothing is playing through your default output device" in text


def test_audio_channels_panel_says_when_a_device_delivers_no_audio(tmp_path):
    pytest.importorskip("streamlit")
    text = _render_live(tmp_path, solo=False, elapsed=60, cand=_FakeAudioSource(0.0, alive=False),
                        intr=_FakeAudioSource(0.1), detector=_FakeDetector())
    assert "no audio arriving from this device" in text


def test_audio_channels_panel_in_solo_mode_explains_the_mic_is_the_candidate(tmp_path):
    pytest.importorskip("streamlit")
    text = _render_live(tmp_path, solo=True, elapsed=60, cand=_FakeAudioSource(0.1), intr=None)
    assert "Not used in solo test mode" in text
    assert "your own mic, solo test mode" in text
