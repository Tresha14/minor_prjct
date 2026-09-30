"""Question markers and per-question segmentation.

The interviewer marks each new question (optionally typing its wording) and
this module turns those markers plus the transcript into per-question
segments -- the unit the post-session report and answer-relevance scoring
work on.

Words are assigned to a question by *timestamp*, never by "whichever
question was current when the word was written to the log". Transcription
lags speech by several seconds (chunks are cut at silence, then sent to
Whisper), so a word spoken just before the interviewer pressed "Mark next
question" is often logged just after it. Tagging at write time would put it
in the wrong question; comparing the word's own start time against the
marker times cannot.

Pure functions over plain rows (same shape EventLog returns) so they're
fast to test with no Streamlit/camera/mic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

MARKER_KIND = "question_marker"


@dataclass
class QuestionMarker:
    question_id: int
    text: str          # "" if the interviewer didn't type the wording
    t_start: float


@dataclass
class QuestionSegment:
    question_id: int
    text: str
    t_start: float
    t_end: float | None        # None only for the last question of a still-running session
    answer_text: str
    word_count: int


def _detail(detail_json) -> dict:
    if not detail_json:
        return {}
    try:
        d = json.loads(detail_json)
    except (json.JSONDecodeError, TypeError):
        return {}
    return d if isinstance(d, dict) else {}


def read_markers(event_rows: list[tuple]) -> list[QuestionMarker]:
    """`event_rows`: (t_start, t_end, source, kind, detail_json) rows as
    returned by EventLog.recent_events(), in any order. Markers written
    before question text/ids existed (bare timestamps) are still handled:
    they get an empty text and an id from their position."""
    raw = sorted((r for r in event_rows if r[3] == MARKER_KIND), key=lambda r: r[0])
    markers = []
    for i, row in enumerate(raw):
        d = _detail(row[4])
        qid = d.get("question_id")
        markers.append(QuestionMarker(
            question_id=qid if isinstance(qid, int) else i + 1,
            text=str(d.get("text") or "").strip(),
            t_start=row[0],
        ))
    return markers


def mark_question(log, text: str, t: float) -> int:
    """Writes a question marker and returns its 1-based question number."""
    n = len(read_markers(log.recent_events(limit=100_000))) + 1
    log.event("session", MARKER_KIND, t_start=t, detail={"question_id": n, "text": text.strip()})
    return n


def build_segments(markers: list[QuestionMarker], words: list[tuple],
                   t_end: float | None = None) -> list[QuestionSegment]:
    """`words`: (t_start, t_end, word, question_id) rows from
    EventLog.all_transcript(). Question i spans [marker_i.t_start,
    marker_{i+1}.t_start); the last one runs to `t_end` (or to the end of
    the transcript if `t_end` is None). Speech before the first marker
    belongs to no question and is left out."""
    ordered = sorted(markers, key=lambda m: m.t_start)
    segments = []
    for i, m in enumerate(ordered):
        end = ordered[i + 1].t_start if i + 1 < len(ordered) else t_end
        in_q = [w for w in words if w[0] >= m.t_start and (end is None or w[0] < end)]
        segments.append(QuestionSegment(
            question_id=m.question_id,
            text=m.text,
            t_start=m.t_start,
            t_end=end,
            answer_text=" ".join(w[2] for w in in_q),
            word_count=len(in_q),
        ))
    return segments


def score_answer_relevance(log, text_worker, t_end: float) -> int:
    """Scores every question that has both typed wording and at least one
    answer word, writing one `answer_relevance` metric per question (at that
    question's end time). Questions with no wording or no answer are
    skipped -- no metric is better than a fabricated 0. Returns how many
    were scored."""
    segments = build_segments(read_markers(log.recent_events(limit=100_000)),
                              log.all_transcript(), t_end=t_end)
    scored = 0
    for seg in segments:
        if seg.text and seg.word_count > 0:
            text_worker.score_answer_relevance(seg.text, seg.answer_text,
                                               t=seg.t_end if seg.t_end is not None else t_end)
            scored += 1
    return scored
