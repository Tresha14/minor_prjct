"""Pure formatting/aggregation helpers for the Streamlit dashboard.

Kept separate from app.py, same reasoning as every other module split in
this project: these are plain functions over plain data (the tuples
EventLog.recent_metrics()/recent_events() return), so they're fast and easy
to unit test without importing Streamlit or touching a camera/mic. app.py
should stay thin UI wiring around these -- if you're tempted to put logic
directly in app.py, it likely belongs here instead so it can be tested.
"""
from __future__ import annotations

import json


def latest_value(rows: list[tuple[float, float]]) -> float | None:
    """`rows`: [(t, value), ...] as returned by EventLog.recent_metrics(),
    which is already ordered oldest-to-newest. Returns the most recent
    value, or None if there are no rows yet."""
    if not rows:
        return None
    return rows[-1][1]


def format_seconds(t: float) -> str:
    """12.4 -> '00:12', 75.0 -> '01:15'. Negative/garbage input clamps to 0
    rather than producing a confusing negative timestamp on screen."""
    t = max(0, int(t))
    return f"{t // 60:02d}:{t % 60:02d}"


EVENT_LABELS = {
    "long_pause": "Long pause",
    "gaze_away": "Looking away from camera",
    "face_lost": "Not visible in frame",
    "calibration_done": "Calibration complete",
    "question_marker": "New question marked",
}

# Events worth showing in the "needs follow-up" timeline. calibration_done
# and question_marker are session bookkeeping, not something to flag.
FLAGGED_EVENT_KINDS = {"long_pause", "gaze_away", "face_lost"}


def format_event(row: tuple) -> str:
    """`row`: (t_start, t_end, source, kind, detail_json) as returned by
    EventLog.recent_events(). Returns a one-line, human-readable string.
    Never raises on malformed detail JSON -- a display helper shouldn't be
    able to crash the dashboard over a logging quirk.
    """
    t_start, t_end, source, kind, detail_json = row
    label = EVENT_LABELS.get(kind, kind)
    ts = format_seconds(t_start)

    detail = {}
    if detail_json:
        try:
            detail = json.loads(detail_json)
        except (json.JSONDecodeError, TypeError):
            detail = {}

    extra = ""
    if kind in ("long_pause", "face_lost") and isinstance(detail.get("duration"), (int, float)):
        extra = f" ({detail['duration']:.1f}s)"
    elif kind == "gaze_away" and isinstance(detail.get("camera_facing_fraction"), (int, float)):
        extra = f" ({detail['camera_facing_fraction'] * 100:.0f}% facing camera)"

    elif kind == "question_marker":
        qid, qtext = detail.get("question_id"), detail.get("text")
        if isinstance(qid, int):
            extra = f" (Q{qid}" + (f": {qtext}" if qtext else "") + ")"

    return f"[{ts}] {label}{extra}"


def summarize_events(rows: list[tuple], kinds: set[str] | None = None, limit: int = 20) -> list[str]:
    """Filters (optionally, to `kinds`) and formats event rows for display.
    `rows` is assumed newest-first, matching EventLog.recent_events()'s
    actual ordering (ORDER BY t_start DESC) -- so no re-sorting here.
    """
    filtered = [r for r in rows if kinds is None or r[3] in kinds]
    return [format_event(r) for r in filtered[:limit]]


def metric_delta_label(rows: list[tuple[float, float]]) -> str | None:
    """A short "up/down since last reading" string for st.metric's delta,
    or None if there aren't at least two readings yet to compare."""
    if len(rows) < 2:
        return None
    prev, curr = rows[-2][1], rows[-1][1]
    diff = curr - prev
    if abs(diff) < 1e-9:
        return None
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.1f}"


TIP_ICONS = {"needs_work": "🔸", "praise": "✅", "info": "ℹ️"}


def format_tip(tip) -> str:
    """One markdown line for a feedback.Tip: an icon for its severity plus
    its text. Unknown severities fall back to the info icon rather than
    raising -- a display helper shouldn't be able to crash the dashboard."""
    return f"{TIP_ICONS.get(tip.severity, TIP_ICONS['info'])} {tip.text}"
