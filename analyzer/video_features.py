"""Pure feature computation from face-tracking output (a head-pose rotation
matrix, blendshape scores). No dependency on MediaPipe, OpenCV or a camera --
takes plain numbers in, returns metrics out. Mirrors the audio/video split
in audio_features.py / transcribe.py: this stays fast and hardware-free to
test, analyzer/video.py is the thin layer that plugs a real camera + model
into it.

Coordinate convention note: `matrix_to_euler` / `euler_to_matrix` are exact
inverses of each other (both go through scipy's Rotation, so the round trip
is verified in tests/test_video_features.py without needing a camera). What
is NOT verified here is whether MediaPipe's own sign convention for its
facial_transformation_matrixes output matches "positive yaw = turned toward
their right" etc. -- that needs a quick check with a real camera once
analyzer/video.py exists (look at the sign while turning your head right and
confirm yaw increases). It matters less than it sounds: every comparison in
this project is against a per-session calibrated baseline (see Baseline
below), not an absolute "0 = facing camera" assumption, so a consistent
sign flip would still work correctly -- it would just make the raw numbers
counter-intuitive to read in a debugger.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from analyzer.config import VIDEO

_EULER_ORDER = "yxz"  # yaw (about Y), pitch (about X), roll (about Z), intrinsic


@dataclass
class HeadPose:
    yaw: float    # degrees
    pitch: float  # degrees
    roll: float   # degrees


def matrix_to_euler(matrix) -> HeadPose:
    """Decomposes a 4x4 (or 3x3) rotation matrix -- as produced by
    MediaPipe's facial_transformation_matrixes -- into yaw/pitch/roll."""
    r = np.asarray(matrix)[:3, :3]
    yaw, pitch, roll = Rotation.from_matrix(r).as_euler(_EULER_ORDER, degrees=True)
    return HeadPose(yaw=float(yaw), pitch=float(pitch), roll=float(roll))


def euler_to_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Inverse of matrix_to_euler. Mainly for tests and calibration
    debugging, but genuinely useful for either."""
    r = Rotation.from_euler(_EULER_ORDER, [yaw_deg, pitch_deg, roll_deg], degrees=True).as_matrix()
    m = np.eye(4)
    m[:3, :3] = r
    return m


@dataclass
class Baseline:
    """A running average head pose, built up during the calibration window
    at the start of a session (see config.VIDEO.calibration_seconds), used
    as the "facing the camera" reference for the rest of the session.
    """
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    n: int = 0

    def update(self, pose: HeadPose) -> None:
        self.n += 1
        self.yaw += (pose.yaw - self.yaw) / self.n
        self.pitch += (pose.pitch - self.pitch) / self.n
        self.roll += (pose.roll - self.roll) / self.n

    def as_pose(self) -> HeadPose:
        return HeadPose(self.yaw, self.pitch, self.roll)


def camera_facing(pose: HeadPose, baseline: HeadPose,
                   yaw_tolerance: float = VIDEO.facing_yaw_tolerance_deg,
                   pitch_tolerance: float = VIDEO.facing_pitch_tolerance_deg) -> bool:
    """True if `pose` is within tolerance of the calibrated baseline -- i.e.
    still roughly facing the camera the way they were during calibration."""
    return (abs(pose.yaw - baseline.yaw) <= yaw_tolerance
            and abs(pose.pitch - baseline.pitch) <= pitch_tolerance)


def pose_stability(poses: list[HeadPose]) -> float:
    """Lower = steadier. The mean of the yaw/pitch/roll standard deviations
    (degrees) across the given poses -- meant to be called on a rolling
    window (see config.VIDEO.pose_window_s), not a whole session."""
    if len(poses) < 2:
        return 0.0
    yaws = [p.yaw for p in poses]
    pitches = [p.pitch for p in poses]
    rolls = [p.roll for p in poses]
    return float(np.mean([np.std(yaws), np.std(pitches), np.std(rolls)]))


def is_blinking(blendshapes: dict, threshold: float = VIDEO.blink_score_threshold) -> bool:
    """`blendshapes` is a {name: score} dict as produced by MediaPipe's
    output_face_blendshapes (score in [0, 1], 1 = fully closed)."""
    left = blendshapes.get("eyeBlinkLeft", 0.0)
    right = blendshapes.get("eyeBlinkRight", 0.0)
    return (left + right) / 2 >= threshold


@dataclass
class BlinkCounter:
    """Counts discrete blink events from a stream of per-frame is_blinking
    booleans. A blink is a closed<-open transition, so a sustained closed
    eye across several frames is one blink, not one per frame.
    """
    count: int = 0
    _was_blinking: bool = field(default=False, repr=False)

    def update(self, blinking_now: bool) -> None:
        if blinking_now and not self._was_blinking:
            self.count += 1
        self._was_blinking = blinking_now


def blink_rate_per_minute(blink_count: int, duration_s: float) -> float:
    if duration_s <= 0:
        return 0.0
    return blink_count / (duration_s / 60.0)


@dataclass
class VideoSegmentStats:
    duration_s: float
    camera_facing_fraction: float  # 0-1
    pose_stability: float          # degrees; lower = steadier
    blink_rate_per_min: float


def video_segment_stats(poses: list[HeadPose], camera_facing_flags: list[bool],
                         blink_count: int, duration_s: float) -> VideoSegmentStats:
    facing_fraction = (sum(camera_facing_flags) / len(camera_facing_flags)
                        if camera_facing_flags else 0.0)
    return VideoSegmentStats(
        duration_s=duration_s,
        camera_facing_fraction=facing_fraction,
        pose_stability=pose_stability(poses),
        blink_rate_per_min=blink_rate_per_minute(blink_count, duration_s),
    )
