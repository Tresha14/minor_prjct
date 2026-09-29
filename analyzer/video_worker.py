"""Live video pipeline: calibration + rolling stats + event detection,
writing continuously to the shared EventLog. Mirrors audio_worker.py's
shape: a thin stateful driver around the pure functions in
video_features.py, with the camera/model access injected so it can be
tested without a real camera (see tests/test_video_worker.py).

Session flow:
  1. For the first `calibration_seconds`, every frame with a detected face
     updates a running-average `Baseline` pose instead of being scored --
     this is the "look at the camera for a few seconds" step the synopsis
     describes, and it's what lets "camera-facing" mean "facing the way
     they were at the start" rather than an absolute, uncalibrated 0.
  2. After that, every frame is compared against the baseline
     (video_features.camera_facing) and checked for a blink
     (video_features.is_blinking), and poses are kept in a rolling window
     for pose_stability(). Every `metric_interval_s`, these get reduced to
     three numbers and written to the log; a bad interval also raises a
     gaze_away event.
  3. Any stretch with no face detected at all (candidate out of frame, bad
     lighting) is logged as a face_lost event once it's over.
"""
from __future__ import annotations

import logging

from analyzer.config import VIDEO
from analyzer.video_features import (
    Baseline, BlinkCounter, camera_facing, is_blinking, video_segment_stats,
)

logger = logging.getLogger(__name__)


class VideoWorker:
    """`tracker` needs one method:
        process_frame(frame_bgr, timestamp_ms) -> FaceReading
            (.face_detected: bool, .pose: HeadPose|None, .blendshapes: dict|None)
    matching analyzer.video.FaceTracker -- inject a fake for tests.
    """

    def __init__(self, event_log, tracker, source: str = "video",
                 calibration_seconds: float = VIDEO.calibration_seconds,
                 pose_window_s: float = VIDEO.pose_window_s,
                 metric_interval_s: float = VIDEO.metric_interval_s,
                 gaze_away_threshold: float = VIDEO.gaze_away_threshold,
                 face_lost_min_s: float = VIDEO.face_lost_min_s):
        self.log = event_log
        self.tracker = tracker
        self.source = source
        self.calibration_seconds = calibration_seconds
        self.pose_window_s = pose_window_s
        self.metric_interval_s = metric_interval_s
        self.gaze_away_threshold = gaze_away_threshold
        self.face_lost_min_s = face_lost_min_s

        self.baseline = Baseline()
        self.calibrated = False
        self._session_start_t: float | None = None
        self._blink_counter = BlinkCounter()
        self._blink_count_at_last_flush = 0
        self._pose_window: list[tuple[float, "HeadPose"]] = []
        self._facing_flags: list[bool] = []
        self._last_metric_t: float | None = None
        self._face_absent_since: float | None = None

    def calibration_seconds_remaining(self, t: float) -> float:
        """0.0 once calibrated. Before that: the full calibration window if
        no frame has been processed yet, otherwise how many seconds of it
        remain at time `t`.
        Exists so a UI can show "look at the camera now, N seconds left" --
        without this, calibration happens silently and whoever is on
        camera has no way to know those specific seconds are the ones
        being averaged into their "normal" baseline. A bad calibration
        (looking at a screen/dashboard instead of the lens during this
        window) then makes completely normal posture read as "looking
        away" for the rest of the session -- this was found to be exactly
        that, not a scoring bug, by testing with a real person who wasn't
        told when calibration was happening.
        """
        if self.calibrated:
            return 0.0
        if self._session_start_t is None:
            return self.calibration_seconds  # no frame processed yet -> the full window is still ahead
        return max(0.0, self.calibration_seconds - (t - self._session_start_t))

    def process_frame(self, frame_bgr, t: float) -> None:
        if self._session_start_t is None:
            self._session_start_t = t
            self._last_metric_t = t

        reading = self.tracker.process_frame(frame_bgr, timestamp_ms=int(t * 1000))

        if not reading.face_detected or reading.pose is None:
            self._handle_absent(t)
            return
        self._handle_present(t, reading)

    def _handle_absent(self, t: float) -> None:
        if self._face_absent_since is None:
            self._face_absent_since = t

    def _handle_present(self, t: float, reading) -> None:
        if self._face_absent_since is not None:
            self._maybe_log_face_lost(t)
            self._face_absent_since = None

        if not self.calibrated:
            self.baseline.update(reading.pose)
            if t - self._session_start_t >= self.calibration_seconds:
                self.calibrated = True
                self.log.event(self.source, "calibration_done", t_start=t,
                                detail={"baseline_yaw": round(self.baseline.yaw, 2),
                                        "baseline_pitch": round(self.baseline.pitch, 2)})
            return  # frames during calibration aren't scored

        facing = camera_facing(reading.pose, self.baseline.as_pose())
        self._facing_flags.append(facing)

        if reading.blendshapes is not None:
            self._blink_counter.update(is_blinking(reading.blendshapes))

        self._pose_window.append((t, reading.pose))
        cutoff = t - self.pose_window_s
        self._pose_window = [(pt, p) for (pt, p) in self._pose_window if pt >= cutoff]

        if t - self._last_metric_t >= self.metric_interval_s:
            self._flush_metrics(t)

    def _maybe_log_face_lost(self, t: float) -> None:
        duration = t - self._face_absent_since
        if duration >= self.face_lost_min_s:
            self.log.event(self.source, "face_lost", t_start=self._face_absent_since, t_end=t,
                            detail={"duration": round(duration, 2)})

    def _flush_metrics(self, t: float) -> None:
        duration = t - self._last_metric_t
        stats = video_segment_stats(
            poses=[p for (_, p) in self._pose_window],
            camera_facing_flags=self._facing_flags,
            blink_count=self._blink_counter.count - self._blink_count_at_last_flush,
            duration_s=duration,
        )
        self.log.metric(self.source, "camera_facing", stats.camera_facing_fraction, t=t)
        self.log.metric(self.source, "pose_stability", stats.pose_stability, t=t)
        self.log.metric(self.source, "blink_rate", stats.blink_rate_per_min, t=t)

        if stats.camera_facing_fraction < self.gaze_away_threshold:
            self.log.event(self.source, "gaze_away", t_start=self._last_metric_t, t_end=t,
                            detail={"camera_facing_fraction": round(stats.camera_facing_fraction, 2)})

        self._blink_count_at_last_flush = self._blink_counter.count
        self._facing_flags = []
        self._last_metric_t = t

    def finish(self, t: float) -> None:
        """Call at the end of a session to flush any pending interval and
        close out a face_lost span still in progress."""
        if self._face_absent_since is not None:
            self._maybe_log_face_lost(t)
            self._face_absent_since = None
        if self.calibrated and self._last_metric_t is not None and t > self._last_metric_t:
            self._flush_metrics(t)
