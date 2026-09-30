"""Live capture: wires a real camera and real audio devices into the
already-tested VideoWorker / AudioWorker pipelines.

Everything in video_worker.py and audio_worker.py has, until now, only ever
been driven by fakes (for the test suite) or by one-off test scripts feeding
pre-recorded clips / a short-lived webcam loop. This module is what makes it
a real live system: three background threads (video, candidate audio,
interviewer audio) each read from a hardware source and push into the
workers continuously.

Sources used (all confirmed working manually earlier in this project, see
README.md "Known gotchas" / status section):
  - Video: a plain webcam via OpenCV (`cv2.VideoCapture`), an OBS Virtual
    Camera (just a different `camera_index`), or -- with no extra software --
    a call window captured directly by `window_capture.WindowSource`.
  - Candidate audio: system-audio loopback via `soundcard` (captures
    whatever the interviewer's speakers are playing -- i.e. the candidate's
    side of the call).
  - Interviewer audio: the default microphone via `soundcard`. We only run
    a lightweight VAD-based "speaking fraction" on this channel, not full
    transcription -- the interviewer isn't who's being coached, so spending
    GPU time transcribing them isn't worth it. It's kept as a separate
    stream mainly so candidate/interviewer speech never gets mixed
    together (see the synopsis, Sec 5.1: "kept separate, no speaker
    mixing").

Threading note: SQLite (via EventLog) handles concurrent writers itself
(each call opens its own short-lived connection in WAL mode), so the three
threads writing to the same event log is safe. VideoWorker / AudioWorker
are each only ever driven by their own single thread, so no locking is
needed around them.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from analyzer.audio_worker import AudioWorker
from analyzer.config import AUDIO, VIDEO
from analyzer.events import EventLog
from analyzer.question_detector import InterviewerQuestionDetector
from analyzer.questions import score_answer_relevance
from analyzer.text import TextWorker
from analyzer.vad import VoiceActivityDetector
from analyzer.video_worker import VideoWorker

logger = logging.getLogger(__name__)


def to_mono_float32(chunk: np.ndarray) -> np.ndarray:
    """Averages a (frames, channels) array down to mono float32. Also
    accepts an already-mono 1-D array unchanged. Pure function, unit tested
    in tests/test_capture.py without touching any audio device.
    """
    arr = np.asarray(chunk, dtype=np.float32)
    if arr.ndim > 1 and arr.shape[1] > 1:
        arr = arr.mean(axis=1)
    return arr.reshape(-1).astype(np.float32)


class VideoSource:
    """Reads frames from a camera (webcam, or an OBS Virtual Camera once
    installed -- it appears as an ordinary camera device index) in a
    background thread and calls `on_frame(frame_bgr, t)` for each one.
    """

    def __init__(self, camera_index: int = 0, target_fps: float | None = None,
                 frame_size: tuple[int, int] | None = (1280, 720)):
        self.camera_index = camera_index
        self.target_fps = target_fps
        # Requested explicitly because DirectShow's 640x480 default comes out
        # as garbled, dark frames from an OBS Virtual Camera (found by testing
        # it: zero faces detected). A webcam that can't do this size just
        # falls back to the nearest one it supports.
        self.frame_size = frame_size
        self._cap = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self, on_frame, t0: float) -> None:
        import cv2
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")
        if self.frame_size:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])

        min_interval = (1.0 / self.target_fps) if self.target_fps else 0.0

        def _loop():
            last_frame_t = 0.0
            while not self._stop.is_set():
                ok, frame = self._cap.read()
                if not ok:
                    logger.warning("VideoSource: camera read failed, stopping")
                    break
                t = time.time() - t0
                if t - last_frame_t < min_interval:
                    continue  # throttle to target_fps without blocking the capture thread
                last_frame_t = t
                on_frame(frame, t)
            self._cap.release()

        self._thread = threading.Thread(target=_loop, name="VideoSource", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)


class AudioSource:
    """Reads audio from a `soundcard` device (a loopback speaker-as-mic, or
    an ordinary microphone) in a background thread and calls
    `on_chunk(mono_float32_chunk, t)` for each piece captured.

    `chunk_seconds` trades latency for robustness: too small and
    `soundcard` logs "data discontinuity" warnings (dropped audio) under
    load; too large and live metrics lag further behind real time. 0.25s
    was chosen empirically as a reasonable middle ground -- see README.md
    "Known gotchas" for the discontinuity issue this is working around.
    """

    def __init__(self, device, sample_rate: int = AUDIO.sample_rate, chunk_seconds: float = 0.25):
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.error: str | None = None  # set if the device stops working mid-session

    def start(self, on_chunk, t0: float) -> None:
        numframes = int(self.chunk_seconds * self.sample_rate)

        def _loop():
            # A failure here used to die silently in this background thread
            # (only a traceback in the terminal), leaving the dashboard just
            # empty. Keep the reason so the UI can show it.
            try:
                with self.device.recorder(samplerate=self.sample_rate) as rec:
                    while not self._stop.is_set():
                        raw = rec.record(numframes=numframes)
                        t = time.time() - t0
                        on_chunk(to_mono_float32(raw), t)
            except Exception as exc:  # noqa: BLE001 - any device failure should be surfaced, not swallowed
                logger.exception("AudioSource(%s) stopped", self.device)
                self.error = f"{type(exc).__name__}: {exc}"

        self._thread = threading.Thread(target=_loop, name=f"AudioSource({self.device})", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)


class QueuedConsumer:
    """Decouples a fast producer (audio/video capture) from a potentially
    slow consumer (e.g. GPU transcription) via a background thread reading
    off a queue, so the consumer's own latency can never block the capture
    thread that's supposed to be reading from the microphone/camera in
    real time.

    This isn't a hypothetical concern: an earlier version of SessionCapture
    called AudioWorker.push_audio() directly from the audio-capture thread.
    Whisper transcription runs synchronously inside push_audio() and can
    take a second or more, so for that whole time the same thread wasn't
    reading from the microphone either -- confirmed by testing to cause
    real, permanent word loss (not just lag), because the OS-level audio
    buffer overflows while nobody's draining it. See README.md's "Known
    gotchas" section.
    """

    def __init__(self, handler):
        self._handler = handler
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def put(self, *args) -> None:
        self._queue.put(args)

    def start(self) -> None:
        def _loop():
            while not (self._stop.is_set() and self._queue.empty()):
                try:
                    args = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                self._handler(*args)

        self._thread = threading.Thread(target=_loop, name="QueuedConsumer", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Signals no more items will arrive and waits for the queue to
        drain -- so audio captured right up to the last moment still gets
        transcribed rather than silently dropped on shutdown."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)


def default_loopback_device():
    """The candidate's audio: whatever the interviewer's speakers are
    playing, captured via loopback (confirmed working -- see README.md)."""
    import soundcard as sc
    speaker = sc.default_speaker()
    return sc.get_microphone(speaker.name, include_loopback=True)


def default_microphone_device():
    """The interviewer's own voice."""
    import soundcard as sc
    return sc.default_microphone()


@dataclass
class InterviewerMonitor:
    """Lightweight stand-in for a full AudioWorker on the interviewer's
    channel: just tracks how much of each interval was spent speaking, via
    VAD, with no transcription. See this module's docstring for why.
    """
    log: EventLog
    vad: VoiceActivityDetector
    source: str = "audio_interviewer"
    interval_s: float = 2.0
    _buffer: np.ndarray = None
    _last_metric_t: float = 0.0

    def __post_init__(self):
        self._buffer = np.zeros(0, dtype=np.float32)

    def push_audio(self, chunk: np.ndarray, t: float) -> None:
        self._buffer = np.concatenate([self._buffer, chunk])
        if t - self._last_metric_t >= self.interval_s and len(self._buffer) > 0:
            fraction = self.vad.speaking_fraction(self._buffer)
            self.log.metric(self.source, "speaking_fraction", fraction, t=t)
            self._buffer = np.zeros(0, dtype=np.float32)
            self._last_metric_t = t


class SessionCapture:
    """Owns the whole live capture session: opens the event log, starts the
    three background sources, and wires them into VideoWorker / AudioWorker
    / InterviewerMonitor. Use as a context manager or call start()/stop().

        with SessionCapture(session_dir, transcriber, tracker) as session:
            time.sleep(60)   # let it run
        # workers are flushed and the log is finalised on exit
    """

    def __init__(self, session_dir, transcriber, face_tracker,
                 camera_index: int = 0, candidate_device=None, interviewer_device=None,
                 solo_test_mode: bool = False, video_source=None,
                 auto_questions: bool = False):
        """`video_source` replaces the camera with anything that has
        `start(on_frame, t0)` / `stop()` -- e.g. `window_capture.WindowSource`,
        which captures a call window directly so OBS isn't needed.

        `auto_questions=True` transcribes the interviewer's microphone turns and
        marks each one that reads as a question (see question_detector.py), so
        nobody has to press "Mark next question". Ignored in solo test mode,
        which has no separate interviewer channel.

        `solo_test_mode=True` points the candidate (transcribed) channel
        at your own microphone instead of speaker loopback, and skips the
        separate interviewer channel entirely (both would otherwise fight
        over the same physical microphone). This exists purely for testing
        and demos by one person: talk into your own mic and see it
        transcribed as if you were the candidate. It is NOT the real
        two-channel design (see this module's docstring) -- for an actual
        interview, leave this False so the candidate's voice comes from the
        call (via loopback) and never gets mixed with the interviewer's.
        """
        self.log = EventLog(session_dir)
        self.solo_test_mode = solo_test_mode
        self.video_worker = VideoWorker(self.log, face_tracker, source="video")
        self.text_worker = TextWorker(self.log)
        self.candidate_worker = AudioWorker(self.log, transcriber, VoiceActivityDetector(),
                                             source="audio_candidate", text_worker=self.text_worker)
        self.interviewer_monitor = InterviewerMonitor(self.log, VoiceActivityDetector())

        # target_fps matters beyond just analysis granularity: running
        # MediaPipe on every raw webcam frame (unthrottled, ~30fps) was
        # found by testing to starve the audio-capture threads of CPU time
        # badly enough to drop words near the end of a clip (see README.md
        # "Known gotchas"). Throttling video to VIDEO.target_fps fixed it.
        self.video_source = video_source or VideoSource(camera_index=camera_index, target_fps=VIDEO.target_fps)
        if solo_test_mode:
            self.candidate_source = AudioSource(candidate_device or default_microphone_device())
            self.interviewer_source = None
        else:
            self.candidate_source = AudioSource(candidate_device or default_loopback_device())
            self.interviewer_source = AudioSource(interviewer_device or default_microphone_device())

        # Candidate audio goes through a queue (see QueuedConsumer) because
        # its handler does GPU transcription, which is slow enough to cause
        # real dropped audio if it ran on the capture thread itself. The
        # interviewer channel and video frames are cheap enough (VAD-only,
        # and a throttled MediaPipe call) that this hasn't been needed for
        # them -- if that ever changes, the same pattern applies.
        self._candidate_queue = QueuedConsumer(self.candidate_worker.push_audio)

        # With auto question detection the interviewer channel also does
        # transcription, so it gets its own queue for the same reason.
        self.question_detector = None
        self._interviewer_queue = None
        if auto_questions and self.interviewer_source is not None:
            self.question_detector = InterviewerQuestionDetector(
                self.log, VoiceActivityDetector(), transcriber, monitor=self.interviewer_monitor)
            self._interviewer_queue = QueuedConsumer(self.question_detector.push_audio)

        self._t0 = None

    def start(self) -> None:
        self._t0 = time.time()
        self.log.session_start(t0=self._t0)
        self._candidate_queue.start()
        self.video_source.start(self.video_worker.process_frame, self._t0)
        self.candidate_source.start(self._candidate_queue.put, self._t0)
        if self.interviewer_source is not None:
            if self._interviewer_queue is not None:
                self._interviewer_queue.start()
                self.interviewer_source.start(self._interviewer_queue.put, self._t0)
            else:
                self.interviewer_source.start(self.interviewer_monitor.push_audio, self._t0)
        logger.info("SessionCapture started (solo_test_mode=%s), writing to %s",
                    self.solo_test_mode, self.log.db_path)

    def audio_problems(self) -> list[str]:
        """Human-readable reasons an audio channel stopped, empty if all is well."""
        problems = []
        for label, src in (("Candidate audio (speaker loopback, or the mic in solo mode)", self.candidate_source),
                           ("Interviewer microphone", self.interviewer_source)):
            if src is not None and getattr(src, "error", None):
                problems.append(f"{label} stopped: {src.error}")
        return problems

    def stop(self) -> None:
        self.video_source.stop()
        self.candidate_source.stop()
        if self.interviewer_source is not None:
            self.interviewer_source.stop()
        if self._interviewer_queue is not None:
            self._interviewer_queue.stop()
        self._candidate_queue.stop()  # drains whatever was captured right up to the end
        t_end = time.time() - self._t0 if self._t0 else 0.0
        self.video_worker.finish(t_end)
        self.candidate_worker.flush()
        scored = score_answer_relevance(self.log, self.text_worker, t_end)
        logger.info("Scored answer relevance for %d question(s)", scored)
        logger.info("SessionCapture stopped at t=%.1fs", t_end)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
