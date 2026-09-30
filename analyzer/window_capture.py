"""Capture a call window straight from the screen -- no OBS, no virtual camera.

Windows only, and pure `ctypes` (no extra pip package). It exists because
OBS's Virtual Camera was the only way to get the candidate's video tile out
of a Zoom/Meet/Teams window, and asking every user to install and configure
OBS is a lot of friction. `WindowSource` has the same start()/stop() shape as
`capture.VideoSource`, so `SessionCapture` can take either.

Two ways to grab a frame (`method`):
  - "screen" (default): copy that rectangle of the screen. It sees exactly
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


def grab_window(hwnd, method: str = "screen") -> np.ndarray | None:
    """One BGR frame of the window, or None if it's gone, minimised or empty.
    See the module docstring for `method`."""
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
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        import cv2
        frame = crop_frame(frame, self.crop)
        h, w = frame.shape[:2]
        if w > self.max_width:
            frame = cv2.resize(frame, (self.max_width, int(h * self.max_width / w)), interpolation=cv2.INTER_AREA)
        return frame

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
