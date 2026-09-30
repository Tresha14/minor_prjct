"""Combined flags: moments where SEVERAL signals got worse together, judged
against the speaker's own baseline.

Why this exists (synopsis Sec 5.6): a single deviation -- one long pause, one
glance away -- is normal in any conversation and flagging each one produces
noise. What's worth a follow-up is co-occurrence: a long pause AND hedged
phrasing AND looking away in the same stretch of the answer. And "worse"
must be relative to *this* speaker, not a universal ideal, because gaze and
pace vary with glasses, camera position and personal style (synopsis Sec 9,
fairness).

How it works:
  1. Every reading is converted to fusion.py's 0-1 "how good is this" score,
     so thresholds here line up with the scores by construction.
  2. Each signal's baseline is the MEDIAN of its readings over the whole
     session (robust to a few bad stretches, unlike a mean). A signal with
     fewer than FLAGS.min_baseline_readings readings has no baseline and
     simply can't fire -- never guessed.
  3. Slide a window (FLAGS.window_s, half-overlapping) over the session. A
     signal *deviates* in a window when the window's average is at least
     FLAGS.norm_drop worse than its baseline AND is itself below
     FLAGS.norm_max. A long_pause event in the window counts as one more
     deviating signal.
  4. A window with >= FLAGS.min_signals deviating signals is flagged;
     overlapping/adjacent flagged windows are merged into one span with the
     union of their signals.

Deliberate limits, stated in the rendered report too:
  - The baseline is computed after the fact from the whole session; the
    synopsis' "calibrate early, then update slowly" live baseline is not
    what runs here.
  - Relative-to-self means a session that is uniformly weak deviates from
    nothing and won't produce combined flags; the absolute coaching tips
    still cover that case.
  - Blink rate is excluded (the synopsis treats it as a weak, secondary
    signal).

Pure functions over the plain rows EventLog returns; no I/O.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

from analyzer.config import FLAGS
from analyzer.dashboard_data import format_seconds
from analyzer.fusion import normalise_metric

# metric name -> short phrase used in the flag text. Order = display order.
SIGNAL_PHRASES = {
    "long_pause": "a long pause",
    "pause_rate": "more pausing than usual",
    "hedge_rate": "hedged phrasing",
    "filler_rate": "more filler words",
    "camera_facing": "looking away from the camera",
    "wpm": "an unusual speaking pace",
    "pose_stability": "restless head movement",
}
_METRIC_SIGNALS = tuple(k for k in SIGNAL_PHRASES if k != "long_pause")


@dataclass
class CombinedFlag:
    t_start: float
    t_end: float
    signals: list[str]            # keys of SIGNAL_PHRASES, in display order
    question_id: int | None = None

    @property
    def text(self) -> str:
        phrases = [SIGNAL_PHRASES[s] for s in self.signals]
        joined = phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]
        return f"{format_seconds(self.t_start)}–{format_seconds(self.t_end)} — {joined} together"


def _baselines(norm_rows: dict[str, list[tuple[float, float]]]) -> dict[str, float | None]:
    return {
        name: (statistics.median(v for _, v in rows) if len(rows) >= FLAGS.min_baseline_readings else None)
        for name, rows in norm_rows.items()
    }


def _windows(duration: float) -> list[tuple[float, float]]:
    out, lo = [], 0.0
    while lo < duration:
        hi = min(lo + FLAGS.window_s, duration)
        # A short tail is already covered by the previous (half-overlapping) window.
        if out and hi - lo < FLAGS.window_s / 2:
            break
        out.append((lo, hi))
        if hi >= duration:
            break
        lo += FLAGS.step_s
    return out


def _question_for(lo: float, hi: float, segments) -> int | None:
    """The question whose time range overlaps [lo, hi) the most, or None.
    `segments`: iterable of objects with .question_id, .t_start, .t_end
    (t_end None = open-ended)."""
    best, best_overlap = None, 0.0
    for s in segments:
        end = s.t_end if s.t_end is not None else float("inf")
        overlap = min(hi, end) - max(lo, s.t_start)
        if overlap > best_overlap:
            best, best_overlap = s.question_id, overlap
    return best


def find_combined_flags(metrics: dict[str, list[tuple[float, float]]], events: list[tuple],
                        duration: float, segments=()) -> list[CombinedFlag]:
    """`metrics`: {name: [(t, raw_value), ...]} (missing names are fine).
    `events`: (t_start, t_end, source, kind, detail) rows. `segments`:
    per-question ranges used only to tag each flag with its question."""
    if duration <= 0:
        return []

    norm_rows = {
        name: [(t, normalise_metric(name, v)) for t, v in metrics.get(name, [])]
        for name in _METRIC_SIGNALS
    }
    baseline = _baselines(norm_rows)
    long_pause_starts = [e[0] for e in events if e[3] == "long_pause"]

    raw_flags: list[CombinedFlag] = []
    for lo, hi in _windows(duration):
        fired = []
        for name in _METRIC_SIGNALS:
            base = baseline[name]
            vals = [v for t, v in norm_rows[name] if lo <= t < hi]
            if base is None or not vals:
                continue
            window_score = sum(vals) / len(vals)
            if window_score <= base - FLAGS.norm_drop and window_score <= FLAGS.norm_max:
                fired.append(name)
        if any(lo <= t < hi for t in long_pause_starts):
            fired.append("long_pause")
        if len(fired) >= FLAGS.min_signals:
            raw_flags.append(CombinedFlag(lo, hi, fired))

    merged: list[CombinedFlag] = []
    for f in raw_flags:                     # already in time order
        if merged and f.t_start <= merged[-1].t_end:
            m = merged[-1]
            m.t_end = max(m.t_end, f.t_end)
            m.signals = sorted(set(m.signals) | set(f.signals), key=list(SIGNAL_PHRASES).index)
        else:
            merged.append(CombinedFlag(f.t_start, f.t_end, sorted(f.signals, key=list(SIGNAL_PHRASES).index)))

    for f in merged:
        f.question_id = _question_for(f.t_start, f.t_end, segments)
    return merged
