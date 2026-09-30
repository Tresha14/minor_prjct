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


# ---------------------------------------------------------------------------
# DirectWindowSource / make_window_source, against a fake `windows_capture`
# package so these run without any real window or graphics hardware.
# ---------------------------------------------------------------------------
import threading
import types

import pytest

import analyzer.window_capture as wc


class _FakeControl:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True

    def is_finished(self):
        return self.stopped


def _install_fake_capture(monkeypatch, *, frames=5, reject_borderless=False, close_after=False,
                          never_start=False, pixel=(10, 20, 30, 255)):
    """Puts a stand-in `windows_capture` module in place. Each fake frame is a
    4-channel BGRA image filled with `pixel`."""
    calls = {"draw_border": []}

    class FakeFrame:
        def __init__(self):
            arr = np.zeros((40, 60, 4), dtype=np.uint8)
            arr[:, :] = pixel
            self.frame_buffer = arr

    class FakeWindowsCapture:
        def __init__(self, cursor_capture=True, draw_border=None, window_hwnd=None, **kw):
            calls["draw_border"].append(draw_border)
            if never_start:
                raise OSError("window not capturable")
            if reject_borderless and draw_border is False:
                raise OSError("borderless capture not supported")
            self.frame_handler = None
            self.closed_handler = None

        def start_free_threaded(self):
            control = _FakeControl()

            def pump():
                for _ in range(frames):
                    if control.stopped:
                        return
                    time.sleep(0.02)
                    self.frame_handler(FakeFrame(), control)
                if close_after:
                    self.closed_handler()

            threading.Thread(target=pump, daemon=True).start()
            return control

    module = types.ModuleType("windows_capture")
    module.WindowsCapture = FakeWindowsCapture
    monkeypatch.setitem(sys.modules, "windows_capture", module)
    return calls


def test_direct_source_delivers_bgr_frames_not_bgra(monkeypatch):
    _install_fake_capture(monkeypatch, frames=30)
    got = []
    src = wc.DirectWindowSource(1, target_fps=50.0)
    src.start(lambda f, t: got.append(f), time.time())
    time.sleep(0.3)
    src.stop()
    assert got
    assert got[0].shape == (40, 60, 3)           # alpha channel dropped
    assert tuple(got[0][0, 0]) == (10, 20, 30)   # colours untouched, same order
    assert got[0].flags["C_CONTIGUOUS"]          # an owned copy, not a view of the library's buffer


def test_direct_source_retries_without_borderless_when_unsupported(monkeypatch):
    calls = _install_fake_capture(monkeypatch, reject_borderless=True, frames=30)
    src = wc.DirectWindowSource(1, target_fps=50.0)
    src.start(lambda f, t: None, time.time())
    src.stop()
    assert calls["draw_border"] == [False, None]


def test_direct_source_reports_when_the_window_closes(monkeypatch):
    _install_fake_capture(monkeypatch, frames=3, close_after=True)
    src = wc.DirectWindowSource(1, target_fps=50.0)
    src.start(lambda f, t: None, time.time())
    deadline = time.time() + 3
    while src.error is None and time.time() < deadline:
        time.sleep(0.02)
    src.stop()
    assert src.error and "closed" in src.error.lower()


def test_direct_source_refuses_to_start_if_no_frame_ever_arrives(monkeypatch):
    _install_fake_capture(monkeypatch, frames=0)
    src = wc.DirectWindowSource(1)
    src.FIRST_FRAME_TIMEOUT_S = 0.2
    with pytest.raises(RuntimeError, match="no frames"):
        src.start(lambda f, t: None, time.time())


def test_direct_source_stop_twice_and_before_start_is_harmless(monkeypatch):
    _install_fake_capture(monkeypatch, frames=30)
    wc.DirectWindowSource(1).stop()              # never started
    src = wc.DirectWindowSource(1, target_fps=50.0)
    src.start(lambda f, t: None, time.time())
    src.stop()
    src.stop()                                   # already stopped


def test_make_window_source_falls_back_to_screen_copy_with_a_note(monkeypatch):
    _install_fake_capture(monkeypatch, never_start=True)
    source, note = wc.make_window_source(1, "direct", 10.0, (0, 0, 0, 0))
    assert isinstance(source, wc.WindowSource) and source.method == "screen"
    assert note and "screen copy" in note.lower()


def test_make_window_source_uses_direct_capture_when_it_works(monkeypatch):
    _install_fake_capture(monkeypatch, frames=30)
    source, note = wc.make_window_source(1, "direct", 10.0, (0, 0, 0, 0))
    assert isinstance(source, wc.DirectWindowSource) and note is None


def test_make_window_source_respects_an_explicit_screen_choice():
    source, note = wc.make_window_source(1, "screen", 10.0, (0, 0, 0, 0))
    assert isinstance(source, wc.WindowSource) and source.method == "screen" and note is None


def test_grab_once_returns_a_frame_and_leaves_nothing_running(monkeypatch):
    _install_fake_capture(monkeypatch, frames=30)
    src = wc.DirectWindowSource(1)
    frame = src.grab_once()
    assert frame is not None and frame.shape == (40, 60, 3)
    assert src._control is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Graphics Capture is Windows-only")
def test_real_capture_of_a_real_window_sees_its_actual_pixels():
    """End to end with no fakes: open a small window painted solid red, capture
    it through the real Windows API, and check the pixels really are red (and
    in BGR order). Skipped if tkinter or the capture package is unavailable."""
    tk = pytest.importorskip("tkinter")
    pytest.importorskip("windows_capture")
    root = tk.Tk()
    try:
        root.title("wc_pixel_test_window")
        root.geometry("320x240+200+200")
        root.configure(bg="#ff0000")
        root.update()
        for _ in range(20):
            root.update()
            time.sleep(0.02)
        hwnd = next((h for h, t in wc.list_windows() if t == "wc_pixel_test_window"), None)
        if hwnd is None:
            pytest.skip("test window not visible to the window list")
        frame = wc.DirectWindowSource(hwnd).grab_once()
        assert frame is not None, "no frame captured from a real window"
        h, w = frame.shape[:2]
        centre = frame[h // 2 - 10:h // 2 + 10, w // 2 - 10:w // 2 + 10].reshape(-1, 3).mean(axis=0)
        b, g, r = centre
        assert r > 200 and g < 60 and b < 60, f"expected red in BGR order, got B,G,R = {centre}"
    finally:
        root.destroy()


# ---------------------------------------------------------------------------
# LivePreview: the live panel's engine.
# ---------------------------------------------------------------------------
class _FakeSource:
    def __init__(self, n_frames=1000, error=None):
        self.stopped = False
        self.error = error
        self._n = n_frames

    def start(self, on_frame, t0):
        def run():
            for i in range(self._n):
                if self.stopped:
                    return
                on_frame(np.full((20, 30, 3), i % 250, dtype=np.uint8), time.time() - t0)
                time.sleep(0.01)
        threading.Thread(target=run, daemon=True).start()

    def stop(self):
        self.stopped = True


def test_live_preview_serves_the_newest_frame_and_counts_fps():
    src = _FakeSource()
    prev = wc.LivePreview(src)
    prev.start()
    time.sleep(0.3)
    first = prev.latest()
    time.sleep(0.2)
    second = prev.latest()
    prev.stop()
    assert first is not None and second is not None
    assert int(second[0, 0, 0]) != int(first[0, 0, 0])   # frames are advancing, not a still image
    assert prev.fps > 10
    assert src.stopped and not prev.running


def test_live_preview_latest_is_none_before_any_frame():
    prev = wc.LivePreview(_FakeSource(n_frames=0))
    prev.start()
    assert prev.latest() is None
    prev.stop()


def test_live_preview_stops_itself_when_nobody_is_looking():
    src = _FakeSource()
    prev = wc.LivePreview(src)
    prev.IDLE_TIMEOUT_S, prev.WATCH_INTERVAL_S = 0.2, 0.05
    prev.start()
    deadline = time.time() + 3
    while prev.running and time.time() < deadline:
        time.sleep(0.05)                                  # no latest() calls = nobody looking
    assert not prev.running and src.stopped


def test_live_preview_keeps_running_while_someone_keeps_looking():
    src = _FakeSource()
    prev = wc.LivePreview(src)
    prev.IDLE_TIMEOUT_S, prev.WATCH_INTERVAL_S = 0.3, 0.05
    prev.start()
    for _ in range(12):
        prev.latest()
        time.sleep(0.05)
    assert prev.running
    prev.stop()


def test_live_preview_stop_is_idempotent_and_passes_errors_through():
    src = _FakeSource(error="The captured window was closed.")
    prev = wc.LivePreview(src)
    prev.start()
    assert prev.error == "The captured window was closed."
    prev.stop()
    prev.stop()


def test_live_preview_face_check_reports_unavailable_instead_of_raising(monkeypatch):
    import analyzer.video as video

    def boom(*a, **k):
        raise FileNotFoundError("no model")
    monkeypatch.setattr(video, "FaceTracker", boom)
    prev = wc.LivePreview(_FakeSource())
    assert prev.face_found(np.zeros((20, 30, 3), dtype=np.uint8)) is None
    assert prev.face_found(np.zeros((20, 30, 3), dtype=np.uint8)) is None   # and doesn't retry forever
