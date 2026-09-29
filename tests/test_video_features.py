import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.video_features import (
    HeadPose, Baseline, BlinkCounter,
    matrix_to_euler, euler_to_matrix, camera_facing, pose_stability,
    is_blinking, blink_rate_per_minute, video_segment_stats,
)


@pytest.mark.parametrize("yaw,pitch,roll", [
    (0.0, 0.0, 0.0),
    (20.0, -10.0, 5.0),
    (-35.0, 15.0, -8.0),
    (45.0, 0.0, 0.0),
])
def test_matrix_euler_round_trip(yaw, pitch, roll):
    matrix = euler_to_matrix(yaw, pitch, roll)
    pose = matrix_to_euler(matrix)
    assert pose.yaw == pytest.approx(yaw, abs=0.5)
    assert pose.pitch == pytest.approx(pitch, abs=0.5)
    assert pose.roll == pytest.approx(roll, abs=0.5)


def test_camera_facing_within_tolerance():
    baseline = HeadPose(yaw=0.0, pitch=0.0, roll=0.0)
    assert camera_facing(HeadPose(5.0, -5.0, 2.0), baseline)
    assert not camera_facing(HeadPose(30.0, 0.0, 0.0), baseline)
    assert not camera_facing(HeadPose(0.0, 25.0, 0.0), baseline)


def test_camera_facing_relative_to_nonzero_baseline():
    # baseline itself is off-center (e.g. camera mounted to one side) --
    # facing should be judged relative to it, not absolute zero.
    baseline = HeadPose(yaw=20.0, pitch=10.0, roll=0.0)
    assert camera_facing(HeadPose(22.0, 8.0, 0.0), baseline)
    assert not camera_facing(HeadPose(0.0, 0.0, 0.0), baseline)  # "centered" is actually a big turn away


def test_baseline_running_average():
    b = Baseline()
    b.update(HeadPose(0.0, 0.0, 0.0))
    b.update(HeadPose(10.0, 10.0, 10.0))
    pose = b.as_pose()
    assert pose.yaw == pytest.approx(5.0)
    assert pose.pitch == pytest.approx(5.0)
    assert pose.roll == pytest.approx(5.0)
    assert b.n == 2


def test_pose_stability_zero_when_identical():
    poses = [HeadPose(1.0, 2.0, 3.0)] * 5
    assert pose_stability(poses) == pytest.approx(0.0)


def test_pose_stability_higher_when_varied():
    steady = [HeadPose(0.0, 0.0, 0.0), HeadPose(0.5, 0.5, 0.5), HeadPose(-0.5, -0.5, -0.5)]
    jittery = [HeadPose(0.0, 0.0, 0.0), HeadPose(15.0, 15.0, 15.0), HeadPose(-15.0, -15.0, -15.0)]
    assert pose_stability(jittery) > pose_stability(steady)


def test_pose_stability_needs_at_least_two_poses():
    assert pose_stability([]) == 0.0
    assert pose_stability([HeadPose(1.0, 1.0, 1.0)]) == 0.0


def test_is_blinking_threshold():
    assert is_blinking({"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.9})
    assert not is_blinking({"eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.05})
    assert not is_blinking({})  # missing keys default to 0 (eyes open)


def test_blink_counter_counts_transitions_not_frames():
    counter = BlinkCounter()
    # eyes open, then closed for 3 frames (one blink), open, closed again (second blink)
    for blinking in [False, True, True, True, False, True, False]:
        counter.update(blinking)
    assert counter.count == 2


def test_blink_rate_per_minute():
    assert blink_rate_per_minute(6, 60.0) == 6.0
    assert blink_rate_per_minute(3, 30.0) == 6.0
    assert blink_rate_per_minute(1, 0.0) == 0.0


def test_video_segment_stats_aggregation():
    poses = [HeadPose(0.0, 0.0, 0.0), HeadPose(2.0, 1.0, 0.0)]
    flags = [True, True, False, True]  # 3/4 facing
    stats = video_segment_stats(poses, flags, blink_count=2, duration_s=30.0)
    assert stats.camera_facing_fraction == pytest.approx(0.75)
    assert stats.blink_rate_per_min == pytest.approx(4.0)
    assert stats.pose_stability >= 0.0
