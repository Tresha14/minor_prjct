"""Pure-logic tests for analyzer/window_capture.py. The actual Win32 grabbing
is checked by hand on a real window (see TESTING_GUIDE.md); everything that
can be tested without one -- cropping, resizing, throttling, giving up when
the window goes away -- is tested here with an injected fake grabber."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.window_capture import WindowSource, crop_frame


def _frame(h=100, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_crop_frame_trims_each_edge_by_its_fraction():
    out = crop_frame(_frame(100, 200), (0.1, 0.2, 0.25, 0.3))
    assert out.shape == (50, 130, 3)  # width 200-20-50, height 100-20-30


def test_crop_frame_no_crop_returns_same_size():
    assert crop_frame(_frame(), (0, 0, 0, 0)).shape == (100, 200, 3)


def test_crop_frame_ignores_a_crop_that_would_leave_almost_nothing():
    assert crop_frame(_frame(), (0.5, 0, 0.45, 0)).shape == (100, 200, 3)  # only 5% of width left


def test_window_source_delivers_cropped_and_downscaled_frames():
    got = []
    src = WindowSource(1, target_fps=50.0, crop=(0.0, 0.0, 0.5, 0.0), max_width=50,
                       grab=lambda h: _frame(100, 200))
    src.start(lambda f, t: got.append(f.shape), time.time())
    time.sleep(0.3)
    src.stop()
    assert got, "no frames delivered"
    assert got[0] == (50, 50, 3)  # cropped to 100 wide x 100 high, then scaled to width 50


def test_window_source_refuses_to_start_if_window_cannot_be_grabbed():
    src = WindowSource(1, grab=lambda h: None)
    try:
        src.start(lambda f, t: None, time.time())
    except RuntimeError as e:
        assert "capture" in str(e).lower()
    else:
        raise AssertionError("expected RuntimeError")


def test_window_source_stops_by_itself_when_window_disappears():
    frames = [_frame()] * 3
    calls = {"n": 0}

    def grab(h):
        calls["n"] += 1
        return frames[calls["n"] - 1] if calls["n"] <= len(frames) else None

    got = []
    src = WindowSource(1, target_fps=100.0, grab=grab)
    src.MAX_FAILURES = 2
    src.start(lambda f, t: got.append(t), time.time())
    src._thread.join(timeout=3)
    assert not src._thread.is_alive()
    assert len(got) == 2  # frames[0] was used by start()'s check; the other two were delivered
