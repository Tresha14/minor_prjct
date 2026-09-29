"""Tests VideoWorker's calibration/scoring/event logic using a fake face
tracker, so these run fast with no camera or MediaPipe model needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.events import EventLog
from analyzer.video_features import HeadPose
from analyzer.video_worker import VideoWorker


class FakeReading:
    def __init__(self, face_detected=True, pose=None, blendshapes=None):
        self.face_detected = face_detected
        self.pose = pose
        self.blendshapes = blendshapes


class FakeTracker:
    """Given a schedule of {timestamp_ms: FakeReading}-style entries (a
    plain list in call order), returns them one at a time regardless of the
    timestamp passed in -- the worker drives timing via `t`, not this fake.
    """

    def __init__(self, readings: list[FakeReading]):
        self._readings = list(readings)
        self._i = 0

    def process_frame(self, frame_bgr, timestamp_ms: int):
        reading = self._readings[self._i]
        self._i = min(self._i + 1, len(self._readings) - 1)
        return reading


def make_log(tmp_path) -> EventLog:
    log = EventLog(tmp_path)
    log.session_start(t0=0.0)
    return log


def centered_pose():
    return HeadPose(yaw=0.0, pitch=0.0, roll=0.0)


def turned_away_pose():
    return HeadPose(yaw=45.0, pitch=0.0, roll=0.0)


def blendshapes(blinking: bool):
    score = 0.9 if blinking else 0.05
    return {"eyeBlinkLeft": score, "eyeBlinkRight": score}


def test_calibration_completes_and_logs_event(tmp_path):
    log = make_log(tmp_path)
    readings = [FakeReading(pose=centered_pose(), blendshapes=blendshapes(False))] * 20
    worker = VideoWorker(log, FakeTracker(readings), calibration_seconds=2.0,
                          metric_interval_s=1.0)

    for i in range(20):
        worker.process_frame(frame_bgr=None, t=i * 0.2)  # 4 seconds total, 0.2s per frame

    assert worker.calibrated
    events = log.recent_events()
    assert any(e[3] == "calibration_done" for e in events)


def test_facing_camera_produces_high_camera_facing_metric(tmp_path):
    log = make_log(tmp_path)
    calib = [FakeReading(pose=centered_pose(), blendshapes=blendshapes(False))] * 10
    scored = [FakeReading(pose=centered_pose(), blendshapes=blendshapes(False))] * 10
    worker = VideoWorker(log, FakeTracker(calib + scored), calibration_seconds=1.0,
                          metric_interval_s=1.0)

    t = 0.0
    for _ in range(20):
        worker.process_frame(frame_bgr=None, t=t)
        t += 0.2
    worker.finish(t)

    rows = log.recent_metrics("camera_facing")
    assert len(rows) >= 1
    assert all(value > 0.9 for (_t, value) in rows)


def test_turning_away_triggers_gaze_away_event(tmp_path):
    log = make_log(tmp_path)
    calib = [FakeReading(pose=centered_pose(), blendshapes=blendshapes(False))] * 10
    away = [FakeReading(pose=turned_away_pose(), blendshapes=blendshapes(False))] * 10
    worker = VideoWorker(log, FakeTracker(calib + away), calibration_seconds=1.0,
                          metric_interval_s=1.0, gaze_away_threshold=0.3)

    t = 0.0
    for _ in range(20):
        worker.process_frame(frame_bgr=None, t=t)
        t += 0.2
    worker.finish(t)

    events = log.recent_events()
    assert any(e[3] == "gaze_away" for e in events)


def test_blinks_are_counted(tmp_path):
    log = make_log(tmp_path)
    calib = [FakeReading(pose=centered_pose(), blendshapes=blendshapes(False))] * 10
    # one blink: closed for 2 frames in the middle of an otherwise-open stream
    pattern = [False, False, False, True, True, False, False, False, False, False]
    scored = [FakeReading(pose=centered_pose(), blendshapes=blendshapes(b)) for b in pattern]
    worker = VideoWorker(log, FakeTracker(calib + scored), calibration_seconds=1.0,
                          metric_interval_s=1.0)

    t = 0.0
    for _ in range(20):
        worker.process_frame(frame_bgr=None, t=t)
        t += 0.2
    worker.finish(t)

    rows = log.recent_metrics("blink_rate")
    assert any(value > 0 for (_t, value) in rows)


def test_face_lost_is_logged_after_min_duration(tmp_path):
    log = make_log(tmp_path)
    calib = [FakeReading(pose=centered_pose(), blendshapes=blendshapes(False))] * 10
    absent = [FakeReading(face_detected=False)] * 10   # 2 seconds absent (0.2s * 10)
    back = [FakeReading(pose=centered_pose(), blendshapes=blendshapes(False))] * 5
    worker = VideoWorker(log, FakeTracker(calib + absent + back), calibration_seconds=1.0,
                          metric_interval_s=1.0, face_lost_min_s=1.0)

    t = 0.0
    for _ in range(25):
        worker.process_frame(frame_bgr=None, t=t)
        t += 0.2
    worker.finish(t)

    events = log.recent_events()
    face_lost = [e for e in events if e[3] == "face_lost"]
    assert len(face_lost) == 1
    assert face_lost[0][1] - face_lost[0][0] >= 1.0  # t_end - t_start


def test_brief_face_dropout_is_not_logged(tmp_path):
    # a single dropped frame (e.g. a motion blur miss) shouldn't be flagged
    log = make_log(tmp_path)
    calib = [FakeReading(pose=centered_pose(), blendshapes=blendshapes(False))] * 10
    blip = [FakeReading(face_detected=False)] * 1
    back = [FakeReading(pose=centered_pose(), blendshapes=blendshapes(False))] * 10
    worker = VideoWorker(log, FakeTracker(calib + blip + back), calibration_seconds=1.0,
                          metric_interval_s=1.0, face_lost_min_s=1.0)

    t = 0.0
    for _ in range(21):
        worker.process_frame(frame_bgr=None, t=t)
        t += 0.2
    worker.finish(t)

    events = log.recent_events()
    assert not any(e[3] == "face_lost" for e in events)
