"""Interviewer-only live dashboard.

Run with:
    ./.venv/Scripts/streamlit.exe run app.py

Shows live indicators while a session is running (camera-facing %, words
per minute, pause rate, filler rate, blink rate), a timeline of flagged
moments (long pauses, looking away, not in frame), and the live transcript.

This view is for the interviewer only. Do not show this screen to the
candidate -- live scores can distract or influence them, and the tool is
meant for coaching, not for judging in the moment. See the synopsis'
ethics section.

Design notes:
  - The heavy GPU speech model (Transcriber) is cached with
    st.cache_resource so it's loaded once per server process, not on every
    Streamlit rerun or every "Start session" click.
  - FaceTracker is deliberately NOT cached the same way: MediaPipe's VIDEO
    running mode requires strictly increasing frame timestamps for the
    life of one FaceLandmarker instance, and a fresh session's clock starts
    back at 0. Reusing one across sessions would silently degrade its
    internal temporal smoothing on every session after the first (the
    per-frame timestamp guard in analyzer/video.py stops it from crashing,
    but not from being wrong). It's cheap to create (well under a second,
    unlike the ~4.5s GPU Whisper load), so a fresh one per session is the
    correct choice, not just a convenient one.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from analyzer.capture import SessionCapture
from analyzer.config import PATHS
from analyzer.dashboard_data import (
    FLAGGED_EVENT_KINDS, format_seconds, latest_value, metric_delta_label, summarize_events,
)
from analyzer.fusion import compute_from_log, write_scores

st.set_page_config(page_title="Interview Coach — Live View", layout="wide")


@st.cache_resource(show_spinner="Loading the speech model (first run downloads a few hundred MB)...")
def get_transcriber():
    from analyzer.transcribe import Transcriber
    return Transcriber()


def start_session() -> None:
    from analyzer.video import FaceTracker  # imported lazily so the page loads fast before this is needed

    transcriber = get_transcriber()
    tracker = FaceTracker()
    session_dir = Path(PATHS.data_dir) / time.strftime("%Y%m%d_%H%M%S")
    session = SessionCapture(session_dir, transcriber, tracker,
                              solo_test_mode=st.session_state.get("solo_test_mode", False))
    session.start()

    st.session_state.session = session
    st.session_state.face_tracker = tracker
    st.session_state.session_dir = str(session_dir)


def stop_session() -> None:
    session: SessionCapture | None = st.session_state.get("session")
    if session is not None:
        session.stop()
    tracker = st.session_state.get("face_tracker")
    if tracker is not None:
        tracker.close()
    st.session_state.session = None
    st.session_state.face_tracker = None


def mark_question() -> None:
    session: SessionCapture | None = st.session_state.get("session")
    if session is None or session._t0 is None:
        return
    elapsed = time.time() - session._t0
    session.log.event("session", "question_marker", t_start=elapsed)
    st.toast(f"Marked a new question at {format_seconds(elapsed)}")


if "session" not in st.session_state:
    st.session_state.session = None
    st.session_state.face_tracker = None

st.title("Interview Coach — Live View")
st.caption("Interviewer-only. This screen is never shown to the candidate.")

session: SessionCapture | None = st.session_state.session

control_cols = st.columns([1, 1, 4])
with control_cols[0]:
    if session is None:
        st.button("▶ Start session", type="primary", on_click=start_session)
    else:
        st.button("■ Stop session", on_click=stop_session)
with control_cols[1]:
    st.button("Mark next question", on_click=mark_question, disabled=session is None)

if session is None:
    st.info(
        "Press **Start session** to begin capturing from the webcam and system audio. "
        "The first start will take a few seconds while the speech model loads."
    )
    st.checkbox(
        "🧪 Solo test mode — transcribe my own microphone as if I were the candidate",
        key="solo_test_mode",
        help=(
            "In a real interview, the candidate's voice comes over the call (captured "
            "via speaker loopback) and is kept separate from the interviewer's "
            "microphone. For testing alone without a real call, this points the "
            "transcribed channel at your own microphone instead, so you can talk and "
            "see it transcribed. Turn this off for an actual interview session."
        ),
    )
else:
    @st.fragment(run_every=1.5)
    def live_view():
        session: SessionCapture | None = st.session_state.session
        if session is None:
            return  # session was stopped from another rerun; nothing to show
        log = session.log

        # The elapsed-time caption lives inside this fragment, not as a
        # plain st.caption() above it, specifically so it keeps ticking on
        # every fragment refresh -- outside the fragment it would only
        # update on a full page rerun (i.e. only when a button is clicked),
        # which looked frozen during live testing.
        elapsed = time.time() - session._t0 if session._t0 else 0.0
        mode_note = " · 🧪 solo test mode (own mic transcribed as candidate)" if session.solo_test_mode else ""
        st.caption(f"Session running — {format_seconds(elapsed)} elapsed · "
                   f"saving to `{st.session_state.session_dir}`{mode_note}")

        # Calibration happens silently otherwise -- found by testing with a
        # real person who wasn't told when it was happening: whatever
        # position they were in during this window becomes "normal" for
        # the rest of the session, so if they were looking at this page
        # instead of the camera, completely ordinary posture afterward
        # reads as "looking away". This banner is the fix -- it tells
        # whoever is on camera exactly when to look at the lens.
        remaining = session.video_worker.calibration_seconds_remaining(elapsed)
        if remaining > 0:
            st.warning(
                f"🎯 **Calibrating — look directly at the camera now.** "
                f"{remaining:.0f}s remaining. Whatever position the camera sees during "
                f"this window becomes the \"normal\" baseline for the rest of the session."
            )
            st.progress(1 - (remaining / session.video_worker.calibration_seconds))

        camera_facing = log.recent_metrics("camera_facing")
        pose_stability = log.recent_metrics("pose_stability")
        blink_rate = log.recent_metrics("blink_rate")
        wpm = log.recent_metrics("wpm")
        pause_rate = log.recent_metrics("pause_rate")
        filler_rate = log.recent_metrics("filler_rate")

        st.subheader("Live indicators")
        metric_cols = st.columns(5)
        cf = latest_value(camera_facing)
        metric_cols[0].metric("Camera-facing", f"{cf * 100:.0f}%" if cf is not None else "—",
                               delta=metric_delta_label(camera_facing))
        w = latest_value(wpm)
        metric_cols[1].metric("Words / min", f"{w:.0f}" if w is not None else "—",
                               delta=metric_delta_label(wpm))
        pr = latest_value(pause_rate)
        metric_cols[2].metric("Pause rate", f"{pr * 100:.0f}%" if pr is not None else "—",
                               delta=metric_delta_label(pause_rate))
        fr = latest_value(filler_rate)
        metric_cols[3].metric("Fillers / 100 words", f"{fr:.1f}" if fr is not None else "—",
                               delta=metric_delta_label(filler_rate))
        br = latest_value(blink_rate)
        metric_cols[4].metric("Blinks / min", f"{br:.0f}" if br is not None else "—",
                               delta=metric_delta_label(blink_rate))

        # Fusion scores: computed fresh from the latest readings above and
        # saved back to the log every refresh, so the post-session report
        # can later read a full history of them like any other metric.
        # Each is None (rendered as "—") rather than a fabricated number
        # if nothing has fed it yet -- see analyzer/fusion.py's docstring.
        fusion_result = compute_from_log(log)
        write_scores(log, fusion_result, t=elapsed)

        st.subheader("Fused scores")
        st.caption("Rule-based, not a black box — expand \"How these scores were built\" below to see exactly what fed each one.")
        score_cols = st.columns(3)
        for col, label, sub in zip(
            score_cols,
            ("Engagement", "Fluency", "Confidence"),
            (fusion_result.engagement, fusion_result.fluency, fusion_result.confidence),
        ):
            col.metric(label, f"{sub.value:.0f}" if sub.value is not None else "—")
        with st.expander("How these scores were built"):
            for label, sub in (
                ("Engagement", fusion_result.engagement),
                ("Fluency", fusion_result.fluency),
                ("Confidence", fusion_result.confidence),
            ):
                if sub.drivers:
                    parts = ", ".join(f"{name} {score * 100:.0f}%" for name, score in sub.drivers.items())
                    st.text(f"{label}: {parts}")
                else:
                    st.text(f"{label}: not enough data yet")

        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.caption("Camera-facing % over time")
            if camera_facing:
                df = pd.DataFrame(camera_facing, columns=["t", "camera_facing"])
                st.line_chart(df.set_index("t"), height=180)
            else:
                st.caption("Waiting for the calibration window to finish…")
        with chart_cols[1]:
            st.caption("Words per minute over time")
            if wpm:
                df = pd.DataFrame(wpm, columns=["t", "wpm"])
                st.line_chart(df.set_index("t"), height=180)
            else:
                st.caption("Waiting for the first transcribed segment…")

        st.subheader("Moments flagged for follow-up")
        events = log.recent_events(limit=200)
        flagged = summarize_events(events, kinds=FLAGGED_EVENT_KINDS, limit=15)
        if flagged:
            for line in flagged:
                st.markdown(f"- {line}")
        else:
            st.caption("Nothing flagged yet.")

        with st.expander("Live transcript"):
            words = log.all_transcript()
            if words:
                text = " ".join(w[2] for w in words)
                st.write(text)
            else:
                st.caption("No speech transcribed yet.")

        with st.expander("All recent events (debug)"):
            for line in summarize_events(events, kinds=None, limit=30):
                st.text(line)

    live_view()
