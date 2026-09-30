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
    FLAGGED_EVENT_KINDS, format_seconds, format_tip, latest_value, metric_delta_label, summarize_events,
    window_average, channel_status, interviewer_turns,
)
from analyzer.feedback import generate_tips_from_log
from analyzer.fusion import compute_from_log, write_scores
from analyzer.questions import mark_question as write_question_marker

st.set_page_config(page_title="Interview Coach — Live View", layout="wide")


def stop_preview() -> None:
    """Stops the live window preview if one is running. Called whenever it could
    otherwise be left capturing in the background (session start, other mode)."""
    entry = st.session_state.get("preview")
    if entry is not None:
        entry["preview"].stop()
    st.session_state.preview = None


# Two modes in one app: the live interviewer view (below) and a review page
# for saved sessions. The review branch stops the script so none of the live
# machinery (session state, camera, model) is touched while reviewing.
page_mode = st.sidebar.radio("Mode", ["Live session", "Review saved session"], key="page_mode")
if page_mode == "Review saved session":
    stop_preview()
    import review_page
    review_page.render()
    st.stop()


CAMERA_MODE = "Camera (webcam or OBS Virtual Camera)"
WINDOW_MODE = "Capture a window (no OBS needed)"


def sync_preview(want: bool) -> None:
    """Keeps exactly the right preview running: one for the selected window and
    method if `want`, otherwise none. Restarts it if the window or method changed
    or it stopped by itself."""
    from analyzer.config import VIDEO
    from analyzer.window_capture import LivePreview, make_window_source

    key = (st.session_state.get("window_hwnd"), current_method())
    entry = st.session_state.get("preview")
    if not want or key[0] is None:
        stop_preview()
        return
    if entry is not None and entry["key"] == key and entry["preview"].running:
        return
    stop_preview()
    try:
        source, note = make_window_source(key[0], key[1], 5.0, (0.0, 0.0, 0.0, 0.0))
        preview = LivePreview(source)
        preview.start()
        st.session_state.preview = {"key": key, "preview": preview, "note": note}
    except Exception as exc:  # noqa: BLE001 - the picker must keep working if capture fails
        st.session_state.preview = None
        st.error(f"Couldn't start the live preview: {exc}")


def render_audio_channels(session, log, elapsed: float) -> None:
    """Shows, for each audio channel, whether sound is arriving and what was
    understood from it -- so "my speech isn't detected" can be told apart from
    "it was heard but goes to the interviewer channel, which isn't transcribed
    into the candidate's transcript" and from "no audio is arriving at all"."""
    now = time.time()
    solo = session.solo_test_mode
    detector = getattr(session, "question_detector", None)
    st.subheader("Audio channels")
    left, right = st.columns(2)

    with left:
        if solo:
            st.markdown("**🎙 Interviewer mic**")
            st.caption("Not used in solo test mode: your own mic is the *candidate* channel (right).")
        else:
            src = getattr(session, "interviewer_source", None)
            label, frac = channel_status(getattr(src, "level", None), getattr(src, "last_chunk_time", None), now)
            st.markdown(f"**🎙 You (interviewer mic)** — {label}")
            st.progress(frac)
            turns = interviewer_turns(log.recent_events(limit=400))
            if turns:
                st.caption(f'You said: "{turns[0]}"')
            elif detector is None:
                st.caption("Your speech is measured here but not transcribed "
                           "(automatic question detection is off).")
            else:
                st.caption("Nothing understood from you yet.")
            if detector is not None:
                st.caption(f"Questions detected: {detector.questions_marked}. "
                           "Your speech is used for this only; it is never added to the candidate's transcript.")

    with right:
        src = getattr(session, "candidate_source", None)
        label, frac = channel_status(getattr(src, "level", None), getattr(src, "last_chunk_time", None), now)
        title = "🗣 Candidate (your own mic, solo test mode)" if solo else "🗣 Candidate (call audio from your speakers)"
        st.markdown(f"**{title}** — {label}")
        st.progress(frac)
        words = [w[2] for w in log.all_transcript()]
        if words:
            st.caption("Heard: “… " + " ".join(words[-14:]) + "”")
        else:
            st.caption("Nothing transcribed yet (it appears a few seconds after they speak).")
        if not solo and elapsed > 15 and frac == 0.0 and not words:
            st.caption("🔎 Nothing is playing through your default output device. Check that the call's "
                       "sound comes out of the same speakers/headphones Windows has as the default.")


@st.fragment(run_every=0.6)
def live_preview_view() -> None:
    """Shows the newest captured frame with the trim applied, plus whether the
    face tracker can find a face in it -- i.e. exactly what would be analysed."""
    from analyzer.window_capture import crop_frame

    entry = st.session_state.get("preview")
    if entry is None:
        return
    preview = entry["preview"]
    if entry.get("note"):
        st.info(entry["note"])
    if preview.error:
        st.error(f"Capture stopped: {preview.error}")
        return
    if not preview.running:
        st.caption("Preview paused (no one was looking). Toggle **Live preview** off and on to restart it.")
        return
    frame = preview.latest()
    if frame is None:
        st.caption("Waiting for the first frame...")
        return
    cropped = crop_frame(frame, current_crop())
    st.image(cropped[:, :, ::-1], use_container_width=True)
    found = preview.face_found(cropped)
    if found is True:
        st.success(f"Face found -- this is what will be analysed. (live, ~{preview.fps:.0f} fps)")
    elif found is False:
        st.warning("No face found in this view yet. Trim the sides until the candidate's face fills "
                   "the picture (and pick their video tile, not your own), or make their video bigger.")
    else:
        st.caption(f"Live (~{preview.fps:.0f} fps). Face check unavailable.")


def current_crop() -> tuple[float, float, float, float]:
    """Edge trim from the sliders, as 0-1 fractions: (left, top, right, bottom)."""
    return tuple(st.session_state.get(k, 0) / 100 for k in ("crop_left", "crop_top", "crop_right", "crop_bottom"))


DIRECT_METHOD = "Direct capture (recommended)"
SCREEN_METHOD = "Screen copy (window must stay uncovered)"
OFFSCREEN_METHOD = "Off-screen (may miss video)"


def current_method() -> str:
    chosen = st.session_state.get("capture_method")
    return {SCREEN_METHOD: "screen", OFFSCREEN_METHOD: "offscreen"}.get(chosen, "direct")


def render_window_picker() -> None:
    """Window dropdown, edge-trim sliders and a one-shot preview. The trim is
    there because the face tracker needs the face to be big: with a whole
    Meet/Zoom window captured, a small video tile reads as 'no face'."""
    from analyzer.window_capture import list_windows

    windows = list_windows()
    titles = dict(windows)
    st.selectbox(
        "Window to capture", options=list(titles), key="window_hwnd",
        format_func=lambda h: titles[h][:90], index=None if not titles else 0,
        placeholder="Open your call window first, then pick it here",
        help="Works even if another window is on top of it, but not if it's minimised.",
    )
    st.radio(
        "Capture method", [DIRECT_METHOD, SCREEN_METHOD, OFFSCREEN_METHOD], key="capture_method",
        horizontal=True,
        help=("Direct capture uses Windows' own window-capture: it works even if the call window is "
              "covered or behind other windows, and it sees video. If it can't run on this PC the app "
              "switches to screen copy and tells you. Screen copy records what is displayed, so the "
              "call window must stay fully visible. Off-screen can come out black where the call "
              "video is drawn by the GPU."),
    )
    cols = st.columns(4)
    for col, label, key in zip(cols, ("Trim left %", "Trim top %", "Trim right %", "Trim bottom %"),
                               ("crop_left", "crop_top", "crop_right", "crop_bottom")):
        col.slider(label, 0, 60, 0, key=key)
    st.toggle("Live preview (shows what will be analysed)", value=True, key="live_preview",
              help="A live view of the chosen window with your trim applied, and whether a face is "
                   "found in it. It stops by itself when you start a session.")
    sync_preview(st.session_state.get("live_preview", True))
    if st.session_state.get("preview") is not None:
        live_preview_view()


@st.cache_resource(show_spinner="Loading the speech model (first run downloads a few hundred MB)...")
def get_transcriber():
    from analyzer.transcribe import Transcriber
    return Transcriber()


def start_session() -> None:
    from analyzer.video import FaceTracker  # imported lazily so the page loads fast before this is needed

    stop_preview()  # the session opens its own capture; don't run two

    transcriber = get_transcriber()
    tracker = FaceTracker()
    session_dir = Path(PATHS.data_dir) / time.strftime("%Y%m%d_%H%M%S")
    video_source = None
    if st.session_state.get("video_mode") == WINDOW_MODE:
        from analyzer.config import VIDEO
        from analyzer.window_capture import make_window_source
        hwnd = st.session_state.get("window_hwnd")
        if not hwnd:
            tracker.close()
            st.session_state.start_error = "Pick a window to capture first."
            return
        video_source, note = make_window_source(hwnd, current_method(), VIDEO.target_fps, current_crop())
        st.session_state.capture_note = note
    else:
        st.session_state.capture_note = None
    st.session_state.start_error = None
    session = SessionCapture(session_dir, transcriber, tracker,
                              camera_index=int(st.session_state.get("camera_index", 0)),
                              solo_test_mode=st.session_state.get("solo_test_mode", False),
                              video_source=video_source,
                              auto_questions=st.session_state.get("auto_questions", True))
    try:
        session.start()
    except Exception as exc:  # noqa: BLE001 - don't leave half-started capture threads behind
        try:
            session.stop()
        except Exception:  # noqa: BLE001
            pass
        tracker.close()
        st.session_state.start_error = f"Couldn't start the session: {type(exc).__name__}: {exc}"
        return

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
    st.session_state.last_session_dir = st.session_state.get("session_dir")
    st.session_state.session = None
    st.session_state.face_tracker = None


def mark_question() -> None:
    session: SessionCapture | None = st.session_state.get("session")
    if session is None or session._t0 is None:
        return
    elapsed = time.time() - session._t0
    text = st.session_state.get("question_text", "")
    n = write_question_marker(session.log, text, elapsed)
    st.session_state.question_text = ""  # ready for the next question
    st.toast(f"Marked question {n} at {format_seconds(elapsed)}")


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
with control_cols[2]:
    st.text_input(
        "Question wording (optional)", key="question_text", disabled=session is None,
        label_visibility="collapsed",
        placeholder="Type or paste the question you're asking, then press Mark next question "
                    "(enables the answer-relevance score)",
    )

if session is None:
    st.info(
        "Press **Start session** to begin capturing from the webcam and system audio. "
        "The first start will take a few seconds while the speech model loads."
    )
    if st.session_state.get("last_session_dir"):
        st.success("Last session saved. Switch **Mode → Review saved session** in the sidebar "
                   "to see its report.")
    if st.session_state.get("start_error"):
        st.error(st.session_state.start_error)
    st.radio("Video source", [CAMERA_MODE, WINDOW_MODE], key="video_mode",
             help="To analyse someone on a Zoom/Meet/Teams call, capture that call window directly.")
    if st.session_state.get("video_mode") == WINDOW_MODE:
        render_window_picker()
    else:
        stop_preview()
        st.number_input(
            "Camera index", min_value=0, max_value=9, value=0, step=1, key="camera_index",
            help="0 is usually the built-in webcam; an OBS Virtual Camera is usually 1.",
        )
    st.checkbox(
        "Detect questions automatically from my microphone", value=True, key="auto_questions",
        help=(
            "Listens to the interviewer's microphone, and each time you ask a question it is "
            "marked for you, with its wording -- no typing, no button. Use headphones so the "
            "candidate's voice doesn't leak into your mic. Doesn't apply in solo test mode "
            "(there's no separate interviewer channel); the manual button always still works."
        ),
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
        if getattr(session, "question_detector", None) is not None:
            mode_note += (f" · 🎙 auto question detection on "
                          f"({session.question_detector.questions_marked} found)")
        st.caption(f"Session running — {format_seconds(elapsed)} elapsed · "
                   f"saving to `{st.session_state.session_dir}`{mode_note}")

        # Calibration happens silently otherwise -- found by testing with a
        # real person who wasn't told when it was happening: whatever
        # position they were in during this window becomes "normal" for
        # the rest of the session, so if they were looking at this page
        # instead of the camera, completely ordinary posture afterward
        # reads as "looking away". This banner is the fix -- it tells
        # whoever is on camera exactly when to look at the lens.
        if st.session_state.get("capture_note"):
            st.info(st.session_state.capture_note)
        for problem in session.video_problems():
            st.error(f"🎥 {problem}. Stop the session and pick the window again.")
        for problem in session.audio_problems():
            st.error(
                f"🔇 {problem}. Another app (often Zoom/Meet) may be holding that device, or Windows' "
                "default input/output changed. Set the right defaults in Windows Sound settings, "
                "then stop and restart the session."
            )

        remaining = session.video_worker.calibration_seconds_remaining(elapsed)
        if remaining > 0:
            st.warning(
                f"🎯 **Calibrating — look directly at the camera now.** "
                f"{remaining:.0f}s remaining. Whatever position the camera sees during "
                f"this window becomes the \"normal\" baseline for the rest of the session."
            )
            st.progress(1 - (remaining / session.video_worker.calibration_seconds))

        render_audio_channels(session, log, elapsed)

        camera_facing = log.recent_metrics("camera_facing")
        pose_stability = log.recent_metrics("pose_stability")
        blink_rate = log.recent_metrics("blink_rate")
        wpm = log.recent_metrics("wpm")
        pause_rate = log.recent_metrics("pause_rate")
        filler_rate = log.recent_metrics("filler_rate")

        hedge_rate = log.recent_metrics("hedge_rate")
        st.subheader("Live indicators")
        metric_cols = st.columns(6)
        st.caption("Big number = latest reading (jumps around). Line under it = average of the last 30 s (steadier).")

        def tile(col, label, rows, scale=1.0, fmt="{:.0f}", suffix=""):
            live, smooth = latest_value(rows), window_average(rows, elapsed)
            show = lambda v: (fmt.format(v * scale) + suffix) if v is not None else "—"
            col.metric(label, show(live), delta=metric_delta_label(rows))
            col.caption(f"last 30 s: **{show(smooth)}**")

        tile(metric_cols[0], "Camera-facing", camera_facing, scale=100, suffix="%")
        tile(metric_cols[1], "Words / min", wpm)
        tile(metric_cols[2], "Pause rate", pause_rate, scale=100, suffix="%")
        tile(metric_cols[3], "Fillers / 100 words", filler_rate, fmt="{:.1f}")
        tile(metric_cols[4], "Blinks / min", blink_rate)
        tile(metric_cols[5], "Hedges / 100 words", hedge_rate, fmt="{:.1f}")

        # Silence on the candidate channel for a while is normal at the very
        # start (calibration, no one's spoken yet) but confusing later on --
        # found by a real user who spoke and stuttered for real but saw
        # every audio metric stay empty the whole session, because their
        # mic was going to the (untranscribed) interviewer channel instead.
        if elapsed > 15 and not wpm:
            if session.solo_test_mode:
                st.warning(
                    "⚠️ No speech detected yet from your microphone, even though solo "
                    "test mode is on. Check the right microphone is selected as your "
                    "system default, and that you're actually speaking near it."
                )
            else:
                st.warning(
                    "⚠️ No candidate speech detected yet. If you're testing alone "
                    "without a real call, your voice is going to the microphone "
                    "channel, which is never transcribed by design — stop this "
                    "session and restart with **Solo test mode** checked."
                )

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
            # Coverage is shown next to EVERY score, always, not just when
            # low -- a score built from half its normal signals renders as
            # a plain number that looks exactly as confident as a complete
            # one unless this is visible. Found by a real user testing with
            # no candidate audio captured: Confidence showed "67" built
            # from only 2 of 4 signals, with no visual difference from a
            # fully-informed score. See analyzer/fusion.py's SubScore
            # docstring.
            if sub.value is None:
                col.caption("no data yet")
            elif sub.coverage < 0.5:
                col.caption(f"⚠️ only {sub.coverage * 100:.0f}% of usual signals")
            else:
                col.caption(f"{sub.coverage * 100:.0f}% of usual signals")
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

        # Tips are built from the same latest readings as the tiles above
        # (feedback.py reuses fusion.py's thresholds), so they move as the
        # readings move. Treat them as a live nudge for the interviewer, not
        # a verdict -- the post-session report summarises the whole session.
        st.subheader("Coaching tips (live)")
        for tip in generate_tips_from_log(log):
            st.markdown(format_tip(tip))

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
