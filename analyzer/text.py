"""Pure text-signal computation: hedging rate, lexicon sentiment, and
question/answer relevance. Week 6's other open item alongside
analyzer/feedback.py.

Deliberately rule-based/lexicon-based like every other scoring layer in
this project -- see analyzer/fusion.py's module docstring for the reason:
the dataset this project can realistically collect is far too small to
train or fine-tune a model responsibly, and a transparent, word-list score
is both easier to defend in a viva and easier to act on than "a model said
so." No dependency on a transcript/microphone/Hugging Face model here --
takes plain strings/word counts in, returns numbers out, so it stays fast
and easy to unit test (mirrors the audio_features.py / video_features.py
split: pure math here, a thin stateful worker below plugs it into the
shared event log).

hedge_rate feeds directly into analyzer/fusion.py's confidence score
(FUSION.confidence_weights["hedge_rate"]) with NO changes needed there --
that weight has been configured but always producing None (dropped) until
this module exists to actually write a "hedge_rate" metric.

Hedge detection itself reuses analyzer.audio_features.count_hedges (same
TEXT.hedge_words lexicon) rather than re-implementing it, so there is one
single definition of "what counts as a hedge phrase" in the whole project.

Answer relevance needs the literal text of the question, which the
interviewer types into app.py's question box when pressing "Mark next
question" (see analyzer/questions.py, which splits the transcript into
per-question answers and calls TextWorker.score_answer_relevance()).
Questions marked without wording simply get no relevance score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from analyzer.audio_features import count_hedges
from analyzer.config import TEXT
from analyzer.events import EventLog

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace. Shared by
    sentiment_score and answer_relevance so both agree on what "a word" is."""
    return _WORD_RE.findall(text.lower())


def hedge_rate_per_100_words(hedge_count: int, word_count: int) -> float:
    """Same shape as audio_features.SegmentStats.filler_rate: hedges per
    100 words, 0.0 for an empty segment rather than dividing by zero."""
    if word_count <= 0:
        return 0.0
    return hedge_count / word_count * 100


@dataclass
class SentimentResult:
    """`score`: -1.0 (all lexicon hits negative) .. +1.0 (all positive),
    0.0 for a segment with no lexicon hits either way -- distinct in
    meaning from "very negative" and "very positive" cancelling out to
    0.0, but not distinguishable from the score alone, which is why
    `positive_hits`/`negative_hits` are kept alongside it (same
    traceability reasoning as fusion.py's SubScore.drivers)."""
    score: float
    positive_hits: int
    negative_hits: int


def sentiment_score(text: str) -> SentimentResult:
    """Counts TEXT.positive_words / TEXT.negative_words lexicon hits (exact
    token match, not substring -- "strongly" must not match "strong") and
    returns (pos - neg) / (pos + neg), the simplest transparent formula
    that's still symmetric and bounded in [-1, 1]."""
    tokens = _tokenize(text)
    positive_set = set(TEXT.positive_words)
    negative_set = set(TEXT.negative_words)
    pos = sum(1 for t in tokens if t in positive_set)
    neg = sum(1 for t in tokens if t in negative_set)
    total = pos + neg
    score = (pos - neg) / total if total > 0 else 0.0
    return SentimentResult(score=score, positive_hits=pos, negative_hits=neg)


@dataclass
class RelevanceResult:
    """`score`: the fraction (0-1) of the question's significant
    (non-stopword) words that also appear in the answer -- 1.0 means the
    answer touches every key word of the question, 0.0 means none.
    `matched_words` is kept for the same reason SubScore.drivers is: a
    caller (or a human sanity-checking the metric) can see exactly *which*
    words drove the number instead of trusting a black box.

    This deliberately measures *coverage of the question*, not similarity
    between the two texts. The first version used Jaccard overlap (shared /
    all distinct words in either text), which shrinks as the answer gets
    longer -- a 3-keyword question against a 150-word answer scored ~0.02
    even when the answer was completely on topic, so it couldn't tell good
    answers from bad. Coverage doesn't have that problem. Its limit, stated
    honestly: it's topical keyword coverage, so an answer that merely
    repeats the question's words would score high; it says nothing about
    the answer's quality or structure.
    """
    score: float
    matched_words: frozenset[str]
    question_words: frozenset[str]
    answer_words: frozenset[str]


def _significant_words(text: str) -> frozenset[str]:
    stop = set(TEXT.stopwords)
    return frozenset(t for t in _tokenize(text) if t not in stop)


def _stem(word: str) -> str:
    """Very light suffix stripping so "database"/"databases" and
    "manage"/"managing"/"managed" compare equal. Not a real stemmer (no
    Porter/NLTK dependency on purpose) -- just enough to stop trivial
    inflection from hiding an obviously on-topic answer."""
    for suffix in ("ing", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[:-len(suffix)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def answer_relevance(question: str, answer: str) -> RelevanceResult:
    """Fraction of the question's significant words found in the answer
    (after light stemming). See RelevanceResult for why this replaced
    Jaccard overlap.

    Returns score=0.0 (not None) when the question has no significant words
    or the answer is empty -- a real (if unhelpful) answer, not a missing
    metric the way a session with no speech yet is; None is reserved in
    this project for "hasn't been observed", not "was observed and turned
    out empty".
    """
    q_words = _significant_words(question)
    a_words = _significant_words(answer)
    answer_stems = {_stem(w) for w in a_words}
    matched = frozenset(w for w in q_words if _stem(w) in answer_stems)
    score = (len(matched) / len(q_words)) if q_words else 0.0
    return RelevanceResult(score=score, matched_words=matched, question_words=q_words, answer_words=a_words)


class TextWorker:
    """Turns already-transcribed text into "text"-source metrics on the
    shared event log. Deliberately does not transcribe or listen to audio
    itself -- it's fed plain text segments (the same text
    analyzer.audio_worker.AudioWorker already assembles internally via
    " ".join(w.text for w in words), see analyzer.audio_features.segment_stats),
    so it can be wired into the live pipeline by whoever integrates it
    without this module needing to know anything about audio capture,
    VAD, or Whisper -- matching every other worker's shape in this
    project (a thin stateful driver around pure functions, event-log-only
    coupling to the rest of the pipeline).
    """

    def __init__(self, event_log: EventLog, source: str = "text"):
        self.log = event_log
        self.source = source

    def process_segment(self, text: str, word_count: int, t: float) -> None:
        """Call once per transcribed segment (mirrors AudioWorker._emit_words'
        cadence) with the segment's full text and word count."""
        hedges = count_hedges(text)
        rate = hedge_rate_per_100_words(hedges, word_count)
        self.log.metric(self.source, "hedge_rate", rate, t=t)

        sentiment = sentiment_score(text)
        self.log.metric(self.source, "sentiment", sentiment.score, t=t)

    def score_answer_relevance(self, question: str, answer: str, t: float) -> RelevanceResult:
        """Call once question text is available (analyzer/questions.py does
        this at the end of a session). Writes
        "answer_relevance" to the log and also returns the full
        RelevanceResult so a caller can show the matched words without a
        second read of the log."""
        result = answer_relevance(question, answer)
        self.log.metric(self.source, "answer_relevance", result.score, t=t)
        return result
