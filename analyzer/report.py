"""Post-session report: turns a saved session's event log into session-wide
and per-question scores, coaching tips and a flagged-moments timeline, and
renders it as one self-contained HTML page.

Everything here reuses the live pipeline's own logic rather than
re-implementing it, so the report can never disagree with the dashboard:
  - scores come from fusion.compute_scores() (same weights, same
    "missing metric is dropped, never guessed" rule, same coverage),
  - tips come from feedback.generate_tips() (same thresholds),
  - flagged moments use dashboard_data's event formatting,
  - per-question splitting uses questions.build_segments() (words assigned
    by timestamp).

Differences from the live view, on purpose: the live view scores the
*latest* reading; the report scores the *average of all readings* in a
window (the whole session, or one question's time range), which is far
steadier and is what the coaching tips are really meant for.

Known simplifications, stated in the rendered report too:
  - Averages are plain means of the interval readings, not time-weighted.
  - Audio readings are stamped when a transcribed chunk *ends*, so the
    first seconds of an answer can be attributed to the previous question.

No Streamlit / hardware here (same split as every other module): pure
functions over the plain rows EventLog returns.
"""
from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path

from analyzer.config import PATHS
from analyzer.dashboard_data import FLAGGED_EVENT_KINDS, format_event, format_seconds
from analyzer.events import EventLog
from analyzer.feedback import Tip, generate_tips
from analyzer.flags import CombinedFlag, find_combined_flags
from analyzer.fusion import FusionInputs, FusionResult, compute_scores
from analyzer.questions import QuestionSegment, build_segments, read_markers
from analyzer.text import answer_relevance

_ALL_ROWS = 10_000_000  # recent_metrics() caps at 2000 by default; a report wants everything


@dataclass
class Section:
    """One scored window: the whole session, or a single question."""
    title: str
    t_start: float
    t_end: float
    inputs: FusionInputs
    scores: FusionResult
    tips: list[Tip]
    sentiment: float | None = None            # -1..+1, mean of readings
    relevance: float | None = None            # 0..1, per-question only
    question_id: int | None = None
    question_text: str = ""
    answer_text: str = ""
    word_count: int = 0


@dataclass
class FlaggedMoment:
    t_start: float
    text: str
    question_id: int | None     # which question it happened during, if any


@dataclass
class Report:
    session_name: str
    duration: float
    overall: Section
    questions: list[Section] = field(default_factory=list)
    flagged: list[FlaggedMoment] = field(default_factory=list)
    combined_flags: list[CombinedFlag] = field(default_factory=list)


# ----------------------------------------------------------------- building

def _window_mean(rows: list[tuple[float, float]], lo: float, hi: float | None) -> float | None:
    vals = [v for t, v in rows if t >= lo and (hi is None or t < hi)]
    return sum(vals) / len(vals) if vals else None


def _inputs_for(metrics: dict[str, list], lo: float, hi: float | None) -> FusionInputs:
    return FusionInputs(**{
        name: _window_mean(metrics[name], lo, hi)
        for name in ("camera_facing", "pose_stability", "blink_rate", "wpm",
                     "pause_rate", "filler_rate", "hedge_rate")
    })


def _section(title: str, metrics: dict[str, list], lo: float, hi: float | None, duration: float,
             **extra) -> Section:
    inputs = _inputs_for(metrics, lo, hi)
    return Section(
        title=title, t_start=lo, t_end=hi if hi is not None else duration, inputs=inputs,
        scores=compute_scores(inputs), tips=generate_tips(inputs),
        sentiment=_window_mean(metrics["sentiment"], lo, hi), **extra,
    )


def _session_duration(log: EventLog, metrics: dict[str, list], words: list[tuple], events: list[tuple]) -> float:
    times = [t for rows in metrics.values() for t, _ in rows]
    times += [w[1] for w in words]
    times += [(e[1] if e[1] is not None else e[0]) for e in events]
    return max(times) if times else 0.0


def build_report(log: EventLog, session_name: str = "") -> Report:
    names = ("camera_facing", "pose_stability", "blink_rate", "wpm", "pause_rate",
             "filler_rate", "hedge_rate", "sentiment")
    metrics = {n: log.recent_metrics(n, limit=_ALL_ROWS) for n in names}
    words = log.all_transcript()
    events = log.recent_events(limit=_ALL_ROWS)
    duration = _session_duration(log, metrics, words, events)

    overall = _section("Whole session", metrics, 0.0, None, duration,
                       answer_text=" ".join(w[2] for w in words), word_count=len(words))

    segments: list[QuestionSegment] = build_segments(read_markers(events), words, t_end=duration)
    questions = []
    for seg in segments:
        relevance = None
        if seg.text and seg.word_count > 0:
            relevance = answer_relevance(seg.text, seg.answer_text).score
        questions.append(_section(
            f"Question {seg.question_id}", metrics, seg.t_start, seg.t_end, duration,
            relevance=relevance, question_id=seg.question_id, question_text=seg.text,
            answer_text=seg.answer_text, word_count=seg.word_count,
        ))

    flagged = []
    for row in sorted(events, key=lambda r: r[0]):
        if row[3] not in FLAGGED_EVENT_KINDS:
            continue
        qid = next((s.question_id for s in segments
                    if row[0] >= s.t_start and (s.t_end is None or row[0] < s.t_end)), None)
        flagged.append(FlaggedMoment(t_start=row[0], text=format_event(row), question_id=qid))

    combined = find_combined_flags(metrics, events, duration, segments)

    return Report(session_name=session_name, duration=duration, overall=overall,
                  questions=questions, flagged=flagged, combined_flags=combined)


def list_sessions(data_dir: str | Path = PATHS.data_dir) -> list[Path]:
    """Session folders that contain an event log, newest first (folder
    names are timestamps, so a reverse name sort is a reverse time sort)."""
    root = Path(data_dir)
    if not root.is_dir():
        return []
    return sorted((d for d in root.iterdir() if (d / PATHS.db_filename).is_file()),
                  key=lambda d: d.name, reverse=True)


def load_report(session_dir: str | Path) -> Report:
    session_dir = Path(session_dir)
    return build_report(EventLog(session_dir), session_name=session_dir.name)


# ---------------------------------------------------------------- rendering

_CSS = """
:root { color-scheme: light; }
body { font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: #1c2430;
       background: #f5f6f8; margin: 0; padding: 24px 16px; }
.wrap { max-width: 880px; margin: 0 auto; }
h1 { font-size: 24px; margin: 0 0 4px; } h2 { font-size: 18px; margin: 28px 0 10px; }
h3 { font-size: 15px; margin: 0 0 6px; }
.sub { color: #5b6675; margin: 0 0 18px; }
.card { background: #fff; border: 1px solid #dfe3e8; border-radius: 10px; padding: 16px 18px; margin: 12px 0; }
.scores { display: flex; gap: 12px; flex-wrap: wrap; }
.score { flex: 1 1 150px; border: 1px solid #dfe3e8; border-radius: 8px; padding: 10px 12px; background: #fafbfc; }
.score .n { font-size: 28px; font-weight: 650; } .score .l { color: #5b6675; font-size: 13px; }
.score .c { font-size: 12px; color: #5b6675; } .score .c.low { color: #b45309; font-weight: 600; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
td, th { text-align: left; padding: 5px 8px; border-bottom: 1px solid #edf0f3; }
th { color: #5b6675; font-weight: 600; }
ul { margin: 6px 0; padding-left: 20px; } li { margin: 4px 0; }
.q { font-style: italic; } .muted { color: #5b6675; } .ans { color: #33404f; white-space: pre-wrap; }
.foot { font-size: 12.5px; color: #5b6675; margin-top: 30px; border-top: 1px solid #dfe3e8; padding-top: 12px; }
@media print { body { background: #fff; } .card { break-inside: avoid; } }
"""

_TIP_ICONS = {"needs_work": "🔸", "praise": "✅", "info": "ℹ️"}

_METRIC_ROWS = (
    ("camera_facing", "Camera-facing", lambda v: f"{v * 100:.0f}%"),
    ("wpm", "Words per minute", lambda v: f"{v:.0f}"),
    ("pause_rate", "Time spent in pauses", lambda v: f"{v * 100:.0f}%"),
    ("filler_rate", "Fillers per 100 words", lambda v: f"{v:.1f}"),
    ("hedge_rate", "Hedges per 100 words", lambda v: f"{v:.1f}"),
    ("pose_stability", "Head movement (° variation)", lambda v: f"{v:.1f}"),
    ("blink_rate", "Blinks per minute", lambda v: f"{v:.0f}"),
)


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _scores_html(scores: FusionResult) -> str:
    cells = []
    for label, sub in (("Engagement", scores.engagement), ("Fluency", scores.fluency),
                       ("Confidence", scores.confidence)):
        if sub.value is None:
            n, c, low = "—", "no data", True
        else:
            n = f"{sub.value:.0f}"
            c = f"built from {sub.coverage * 100:.0f}% of usual signals"
            low = sub.coverage < 0.5
        cells.append(f'<div class="score"><div class="n">{n}</div><div class="l">{label}</div>'
                     f'<div class="c{" low" if low else ""}">{"⚠️ " if low and sub.value is not None else ""}{c}</div></div>')
    return f'<div class="scores">{"".join(cells)}</div>'


def _metrics_table_html(sec: Section) -> str:
    rows = []
    for name, label, fmt in _METRIC_ROWS:
        v = getattr(sec.inputs, name)
        rows.append(f"<tr><td>{label}</td><td>{fmt(v) if v is not None else '<span class=muted>no data</span>'}</td></tr>")
    if sec.sentiment is not None:
        rows.append(f"<tr><td>Sentiment (−1 negative … +1 positive)</td><td>{sec.sentiment:+.2f}</td></tr>")
    if sec.question_id is not None:
        rel = f"{sec.relevance * 100:.0f}% of the question's key words" if sec.relevance is not None else \
              "<span class=muted>not scored (needs typed question wording and an answer)</span>"
        rows.append(f"<tr><td>Answer relevance</td><td>{rel}</td></tr>")
    return f"<table>{''.join(rows)}</table>"


def _tips_html(tips: list[Tip]) -> str:
    items = "".join(f"<li>{_TIP_ICONS.get(t.severity, _TIP_ICONS['info'])} {_e(t.text)}</li>" for t in tips)
    return f"<ul>{items}</ul>"


def _section_html(sec: Section, flagged: list[FlaggedMoment], combined: list[CombinedFlag] = ()) -> str:
    head = f"<h3>{_e(sec.title)} <span class=muted>({format_seconds(sec.t_start)}–{format_seconds(sec.t_end)})</span></h3>"
    if sec.question_id is not None:
        head += (f'<p class="q">{_e(sec.question_text)}</p>' if sec.question_text
                 else '<p class="muted">Question wording was not entered.</p>')
    body = _scores_html(sec.scores) + _metrics_table_html(sec)
    mine_combined = [f for f in combined if f.question_id == sec.question_id] if sec.question_id is not None else []
    if mine_combined:
        body += ("<p><b>Several signals together during this answer</b></p><ul>" +
                 "".join(f"<li>{_e(f.text)}</li>" for f in mine_combined) + "</ul>")
    mine = [f for f in flagged if f.question_id == sec.question_id] if sec.question_id is not None else []
    if mine:
        body += "<p><b>Flagged during this answer</b></p><ul>" + "".join(f"<li>{_e(f.text)}</li>" for f in mine) + "</ul>"
    body += "<p><b>Coaching tips</b></p>" + _tips_html(sec.tips)
    if sec.question_id is not None:
        body += ('<p><b>Answer transcript</b></p>' +
                 (f'<p class="ans">{_e(sec.answer_text)}</p>' if sec.answer_text
                  else '<p class="muted">No speech was transcribed during this question.</p>'))
    return f'<div class="card">{head}{body}</div>'


def render_html(report: Report) -> str:
    """One self-contained HTML page (inline CSS, no scripts, no external
    requests). Every piece of user-supplied text -- question wording,
    transcript -- is escaped."""
    parts = [f"<h1>Interview practice report</h1>",
             f'<p class="sub">Session {_e(report.session_name) or "—"} · '
             f"length {format_seconds(report.duration)} · {len(report.questions)} question(s) marked</p>"]

    parts.append("<h2>Whole session</h2>")
    parts.append(_section_html(report.overall, report.flagged))

    if report.questions:
        parts.append("<h2>By question</h2>")
        parts += [_section_html(q, report.flagged, report.combined_flags) for q in report.questions]
    else:
        parts.append('<p class="muted">No questions were marked during this session, so there is no '
                     "per-question breakdown. Use “Mark next question” during the interview to get one.</p>")

    parts.append("<h2>Best moments to replay: several signals together</h2>")
    if report.combined_flags:
        items = "".join(
            f"<li>{_e(f.text)}" + (f' <span class="muted">— question {f.question_id}</span>' if f.question_id else "") + "</li>"
            for f in report.combined_flags)
        parts.append(f'<div class="card"><ul>{items}</ul>'
                     '<p class="muted">A stretch is listed only when at least two signals were clearly worse than '
                     "this speaker's own typical level at the same time. One-off glances or pauses are not listed here.</p></div>")
    else:
        parts.append('<div class="card muted">No stretch had several signals worse than usual at once.</div>')

    parts.append("<h2>Every individual flag</h2>")
    if report.flagged:
        items = "".join(
            f"<li>{_e(f.text)}" + (f' <span class="muted">— during question {f.question_id}</span>' if f.question_id else "") + "</li>"
            for f in report.flagged)
        parts.append(f'<div class="card"><ul>{items}</ul></div>')
    else:
        parts.append('<div class="card muted">Nothing was flagged.</div>')

    parts.append(
        '<div class="foot">These are behavioural proxies measured from the video and audio, not measurements of '
        "true confidence or engagement. Gaze, blinking and pacing vary with glasses, lighting, camera position, "
        "accent and individual differences. Scores use fixed reference ranges that have not yet been validated "
        "against human ratings. “Built from X% of usual signals” shows how many of a score's configured inputs "
        "actually had data — treat low values with caution. Numbers are plain averages of the interval readings; "
        "audio readings are stamped when a transcribed chunk ends, so the first seconds of an answer can be counted "
        "toward the previous question. “Worse than usual” is measured against this session's own median for each "
        "signal, computed after the fact; a session that is uniformly weak will therefore show few combined flags — "
        "read the coaching tips for that. This report is for practice and coaching only, not for hiring, grading or "
        "screening decisions.</div>")

    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Interview practice report — {_e(report.session_name)}</title>"
            f'<style>{_CSS}</style></head><body><div class="wrap">{"".join(parts)}</div></body></html>')
