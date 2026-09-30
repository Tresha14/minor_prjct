"""Capture a call window straight from the screen -- no OBS, no virtual camera.

Windows only. It exists because OBS's Virtual Camera was the only way to get
the candidate's video tile out of a Zoom/Meet/Teams window, and asking every
user to install and configure OBS is a lot of friction. `WindowSource` and
`DirectWindowSource` have the same start()/stop() shape as
`capture.VideoSource`, so `SessionCapture` can take any of them.

Three ways to grab a frame (`method`):
  - "direct" (default): Windows Graphics Capture, the same API OBS's window
    capture uses, via the `windows-capture` package (pinned in
    requirements.txt and installed by pip like everything else -- users
    install nothing separately). It captures the window itself, so it works
    when the window is covered or behind others, and it sees GPU-drawn video.
    If it can't start (old Windows, driver problem) `make_window_source`
    falls back to "screen" and tells the user.
  - "screen": copy that rectangle of the screen. It sees exactly
    what is displayed, including GPU-drawn video -- which is what a browser
    call (Meet, Zoom in Chrome) needs -- but the window must stay visible and
    not be covered by another window.
  - "offscreen": `PrintWindow` asks the window to draw itself into a bitmap,
    which works even when covered but comes back BLANK where the window uses
    GPU video (found on a real Google Meet call: controls captured, the
    video tile black). Fine for windows without video.

The face tracker needs the face to be reasonably big: with the whole Meet
window captured, a small tile reads as "no face". `crop` trims fractions off
each edge so the user can zoom in on the tile (see `crop_frame`).
"""
from __future__ import annotations

import ctypes
import logging
import threading
import time
from ctypes import wintypes

import numpy as np

logger = logging.getLogger(__name__)

PW_RENDERFULLCONTENT = 2
SRCCOPY = 0x00CC0020
DWMWA_CLOAKED = 14
MIN_WINDOW_W, MIN_WINDOW_H = 200, 150   # skip tiny helper windows in the picker


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


_api_ready = False


def _api():
    """(user32, gdi32), with handle-sized argument/return types set once.
    Without this ctypes treats handles as 32-bit ints and truncates them on
    64-bit Windows."""
    global _api_ready
    u, g = ctypes.windll.user32, ctypes.windll.gdi32
    if not _api_ready:
        try:
            u.SetProcessDPIAware()  # real pixel sizes, not DPI-scaled ones
        except Exception:
            pass
        u.GetWindowDC.restype = wintypes.HDC
        u.GetWindowDC.argtypes = [wintypes.HWND]
        u.GetDC.restype = wintypes.HDC
        u.GetDC.argtypes = [wintypes.HWND]
        u.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
        u.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
        u.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        u.IsWindow.argtypes = [wintypes.HWND]
        u.IsIconic.argtypes = [wintypes.HWND]
        u.IsWindowVisible.argtypes = [wintypes.HWND]
        u.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        u.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        g.CreateCompatibleDC.restype = wintypes.HDC
        g.CreateCompatibleDC.argtypes = [wintypes.HDC]
        g.CreateCompatibleBitmap.restype = wintypes.HBITMAP
        g.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
        g.SelectObject.restype = wintypes.HGDIOBJ
        g.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
        g.DeleteObject.argtypes = [wintypes.HGDIOBJ]
        g.DeleteDC.argtypes = [wintypes.HDC]
        g.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
        g.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                                ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
        _api_ready = True
    return u, g


def _window_size(hwnd) -> tuple[wintypes.RECT, int, int]:
    u, _ = _api()
    rect = wintypes.RECT()
    u.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect, rect.right - rect.left, rect.bottom - rect.top


def list_windows() -> list[tuple[int, str]]:
    """Visible, titled, non-minimised top-level windows big enough to hold a
    video call, as [(hwnd, title), ...] sorted by title."""
    u, _ = _api()
    found: list[tuple[int, str]] = []
    dwm = ctypes.windll.dwmapi

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _visit(hwnd, _lparam):
        if not u.IsWindowVisible(hwnd) or u.IsIconic(hwnd):
            return True
        cloaked = wintypes.DWORD(0)  # hidden UWP shells report "visible" but are cloaked
        dwm.DwmGetWindowAttribute(wintypes.HWND(hwnd), DWMWA_CLOAKED, ctypes.byref(cloaked), 4)
        if cloaked.value:
            return True
        n = u.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(hwnd, buf, n + 1)
        _, w, h = _window_size(hwnd)
        if w >= MIN_WINDOW_W and h >= MIN_WINDOW_H and buf.value != "Program Manager":
            found.append((int(hwnd), buf.value))
        return True

    u.EnumWindows(_visit, 0)
    return sorted(found, key=lambda x: x[1].lower())


def _read_bitmap(g, mem, bmp, w: int, h: int) -> np.ndarray:
    bi = _BITMAPINFOHEADER(ctypes.sizeof(_BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)  # -h: top-down
    buf = (ctypes.c_ubyte * (w * h * 4))()
    g.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(bi), 0)
    return np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].copy()  # BGRA -> BGR


def grab_window(hwnd, method: str = "direct") -> np.ndarray | None:
    """One BGR frame of the window, or None if it's gone, minimised or empty.
    See the module docstring for `method`."""
    if method == "direct":
        return DirectWindowSource(hwnd).grab_once()
    u, g = _api()
    if not u.IsWindow(hwnd) or u.IsIconic(hwnd):
        return None
    rect, w, h = _window_size(hwnd)
    if w <= 0 or h <= 0:
        return None

    win_dc = u.GetWindowDC(hwnd)
    mem = g.CreateCompatibleDC(win_dc)
    bmp = g.CreateCompatibleBitmap(win_dc, w, h)
    old = g.SelectObject(mem, bmp)
    try:
        if method == "offscreen":
            u.PrintWindow(hwnd, mem, PW_RENDERFULLCONTENT)
        else:
            screen_dc = u.GetDC(None)
            try:
                g.BitBlt(mem, 0, 0, w, h, screen_dc, rect.left, rect.top, SRCCOPY)
            finally:
                u.ReleaseDC(None, screen_dc)
        return _read_bitmap(g, mem, bmp, w, h)
    finally:
        g.SelectObject(mem, old)
        g.DeleteObject(bmp)
        g.DeleteDC(mem)
        u.ReleaseDC(hwnd, win_dc)


def prepare_frame(frame: np.ndarray, crop: tuple[float, float, float, float], max_width: int) -> np.ndarray:
    """Crop the edges, then shrink to at most `max_width` wide."""
    import cv2
    frame = crop_frame(frame, crop)
    h, w = frame.shape[:2]
    if w > max_width:
        frame = cv2.resize(frame, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    return frame


def crop_frame(frame: np.ndarray, crop: tuple[float, float, float, float]) -> np.ndarray:
    """Trim `crop` = (left, top, right, bottom), each a 0-1 fraction of the
    frame's width/height to remove from that edge. A crop that would leave
    less than 10% of either side is ignored rather than returning an empty
    or useless image."""
    left, top, right, bottom = crop
    h, w = frame.shape[:2]
    x0, x1 = int(w * left), int(w * (1 - right))
    y0, y1 = int(h * top), int(h * (1 - bottom))
    if x1 - x0 < w * 0.1 or y1 - y0 < h * 0.1:
        return frame
    return frame[y0:y1, x0:x1]


class WindowSource:
    """Same start()/stop() contract as `capture.VideoSource`: calls
    `on_frame(frame_bgr, t)` from a background thread, throttled to
    `target_fps`. Stops itself if the window closes or stays unreadable."""

    MAX_FAILURES = 25  # consecutive empty grabs (~5s at 0.2s apart) before giving up

    def __init__(self, hwnd: int, target_fps: float | None = None,
                 crop: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
                 max_width: int = 1280, method: str = "screen", grab=None):
        self.hwnd = hwnd
        self.method = method
        self.target_fps = target_fps
        self.crop = crop
        self.max_width = max_width
        self._grab = grab or (lambda h: grab_window(h, method))
        self.error: str | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        return prepare_frame(frame, self.crop, self.max_width)

    def start(self, on_frame, t0: float) -> None:
        if self._grab(self.hwnd) is None:
            raise RuntimeError("Could not capture that window (closed or minimised?)")
        interval = (1.0 / self.target_fps) if self.target_fps else 0.0

        def _loop():
            failures = 0
            while not self._stop.is_set():
                started = time.time()
                frame = self._grab(self.hwnd)
                if frame is None:
                    failures += 1
                    if failures >= self.MAX_FAILURES:
                        logger.warning("WindowSource: window unavailable, stopping")
                        self.error = "The captured window was closed or minimised."
                        break
                    self._stop.wait(0.2)
                    continue
                failures = 0
                on_frame(self._prepare(frame), time.time() - t0)
                self._stop.wait(max(0.0, interval - (time.time() - started)))

        self._thread = threading.Thread(target=_loop, name="WindowSource", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)


class DirectWindowSource:
    """Windows Graphics Capture (see the module docstring). Same start()/stop()
    contract as `WindowSource`.

    The capture library calls back on its own native thread with a frame
    buffer that is only valid inside the callback, so the callback does the
    bare minimum: skip the frame if one was taken very recently, otherwise
    copy it and keep it as "latest". A separate thread of ours then hands the
    latest frame to the analysis at `target_fps`, so slow analysis can never
    stall the capture thread, and every callback is wrapped so an exception
    can never escape into native code.

    Windows only redraw when their content changes, so a still window
    produces no new frames; the last one is simply re-delivered."""

    FIRST_FRAME_TIMEOUT_S = 3.0

    def __init__(self, hwnd: int, target_fps: float | None = None,
                 crop: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
                 max_width: int = 1280):
        self.hwnd = hwnd
        self.target_fps = target_fps
        self.crop = crop
        self.max_width = max_width
        self.error: str | None = None      # set if the window closes or capture breaks mid-session
        self._interval = (1.0 / target_fps) if target_fps else 0.0
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._have_frame = threading.Event()
        self._stop = threading.Event()
        self._control = None
        self._thread: threading.Thread | None = None
        self._last_accept = 0.0

    def _start_capture(self):
        from windows_capture import WindowsCapture

        def on_frame_arrived(frame, control):
            try:
                if self._stop.is_set():
                    control.stop()
                    return
                now = time.time()
                if now - self._last_accept < self._interval:
                    return  # too soon after the last one: skip without copying
                self._last_accept = now
                img = np.ascontiguousarray(frame.frame_buffer[:, :, :3])  # BGRA view -> owned BGR copy
                with self._lock:
                    self._latest = img
                self._have_frame.set()
            except Exception as exc:  # noqa: BLE001 - must never propagate into the native thread
                self.error = f"{type(exc).__name__}: {exc}"

        def on_closed():
            if not self._stop.is_set():
                self.error = "The captured window was closed."

        last_exc: Exception | None = None
        for draw_border in (False, None):  # no yellow border where Windows allows it, else the default
            try:
                cap = WindowsCapture(cursor_capture=False, draw_border=draw_border, window_hwnd=self.hwnd)
                cap.frame_handler, cap.closed_handler = on_frame_arrived, on_closed
                return cap.start_free_threaded()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        raise RuntimeError(f"Windows Graphics Capture could not start: {last_exc}")

    def start(self, on_frame, t0: float) -> None:
        self._control = self._start_capture()
        if not self._have_frame.wait(self.FIRST_FRAME_TIMEOUT_S):
            self._shutdown_capture()
            raise RuntimeError("Windows Graphics Capture produced no frames (window minimised or protected?)")
        interval = self._interval or 0.1

        def _loop():
            while not self._stop.is_set() and self.error is None:
                with self._lock:
                    img = self._latest
                if img is not None:
                    on_frame(prepare_frame(img, self.crop, self.max_width), time.time() - t0)
                self._stop.wait(interval)
            if self.error:
                logger.warning("DirectWindowSource stopped: %s", self.error)

        self._thread = threading.Thread(target=_loop, name="DirectWindowSource", daemon=True)
        self._thread.start()

    def _shutdown_capture(self) -> None:
        self._stop.set()
        control, self._control = self._control, None
        if control is None:
            return
        try:
            control.stop()
            deadline = time.time() + 2.0
            while not control.is_finished() and time.time() < deadline:
                time.sleep(0.05)
        except Exception:  # noqa: BLE001 - shutting down; nothing useful to do with a failure here
            logger.exception("DirectWindowSource: error while stopping capture")

    def stop(self, timeout: float = 2.0) -> None:
        self._shutdown_capture()
        if self._thread:
            self._thread.join(timeout=timeout)

    def grab_once(self) -> np.ndarray | None:
        """One uncropped BGR frame (starts and stops a capture), or None."""
        frame = None
        try:
            self._control = self._start_capture()
            if self._have_frame.wait(self.FIRST_FRAME_TIMEOUT_S):
                with self._lock:
                    frame = None if self._latest is None else self._latest.copy()
        except Exception:  # noqa: BLE001
            logger.exception("DirectWindowSource.grab_once failed")
        finally:
            self._shutdown_capture()
        return frame


def make_window_source(hwnd: int, method: str, target_fps: float | None,
                       crop: tuple[float, float, float, float]):
    """Returns (source, note). Tries the requested method; if "direct" can't
    produce a frame it falls back to "screen" and returns a note saying so,
    so the session still runs and the user is told why."""
    if method == "direct":
        if DirectWindowSource(hwnd).grab_once() is not None:
            return DirectWindowSource(hwnd, target_fps=target_fps, crop=crop), None
        return (WindowSource(hwnd, target_fps=target_fps, crop=crop, method="screen"),
                "Direct capture isn't available for this window on this PC, so screen copy is being "
                "used instead -- keep the call window fully visible and uncovered.")
    return WindowSource(hwnd, target_fps=target_fps, crop=crop, method=method), None


class LivePreview:
    """Keeps a capture running so the UI can show the newest frame of the
    chosen window while the user lines things up (trim, window choice).

    Safety: it is only meant to run while someone is looking at it, so a
    watchdog stops the capture if nobody has asked for a frame for
    `IDLE_TIMEOUT_S` (e.g. the browser tab was closed). The app also stops it
    when a session starts or the user leaves the screen."""

    IDLE_TIMEOUT_S = 20.0
    WATCH_INTERVAL_S = 1.0

    def __init__(self, source):
        self._source = source
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._frames = 0
        self._started_at = time.time()
        self._last_read = time.time()
        self._stopped = threading.Event()
        self._tracker = None
        self._tracker_failed = False
        self._last_ts_ms = 0

    def start(self) -> None:
        self._started_at = self._last_read = time.time()
        self._source.start(self._on_frame, time.time())
        threading.Thread(target=self._watch, name="LivePreviewWatchdog", daemon=True).start()

    def _on_frame(self, frame: np.ndarray, t: float) -> None:
        with self._lock:
            self._latest = frame
            self._frames += 1

    def _watch(self) -> None:
        while not self._stopped.wait(self.WATCH_INTERVAL_S):
            if time.time() - self._last_read > self.IDLE_TIMEOUT_S:
                logger.info("LivePreview: nobody is looking, stopping the capture")
                self.stop()
                return

    def latest(self) -> np.ndarray | None:
        """The newest uncropped frame, or None if none has arrived yet."""
        self._last_read = time.time()
        with self._lock:
            return self._latest

    @property
    def running(self) -> bool:
        return not self._stopped.is_set()

    @property
    def error(self) -> str | None:
        return getattr(self._source, "error", None)

    @property
    def fps(self) -> float:
        elapsed = time.time() - self._started_at
        return self._frames / elapsed if elapsed > 0 else 0.0

    def face_found(self, frame_bgr: np.ndarray) -> bool | None:
        """Whether the face tracker finds a face in `frame_bgr` (the cropped
        frame that would be analysed). None if the tracker isn't available."""
        if self._tracker_failed:
            return None
        try:
            if self._tracker is None:
                from analyzer.video import FaceTracker
                self._tracker = FaceTracker()
            self._last_ts_ms = max(self._last_ts_ms + 1, int((time.time() - self._started_at) * 1000))
            reading = self._tracker.process_frame(frame_bgr, timestamp_ms=self._last_ts_ms)
            return bool(reading.face_detected)
        except Exception:  # noqa: BLE001 - a preview helper must never break the page
            logger.exception("LivePreview: face check unavailable")
            self._tracker_failed = True
            return None

    def stop(self) -> None:
        if self._stopped.is_set():
            return
        self._stopped.set()
        try:
            self._source.stop()
        except Exception:  # noqa: BLE001
            logger.exception("LivePreview: error while stopping the capture")
        if self._tracker is not None:
            try:
                self._tracker.close()
            except Exception:  # noqa: BLE001
                pass
            self._tracker = None
