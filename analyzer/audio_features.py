"""Pure feature computation from a word-level transcript.

Deliberately has no dependency on faster-whisper, the microphone, or SQLite:
it takes plain (word, start, end) tuples in and returns numbers out, which
makes it fast and easy to unit test. `transcribe.py` and `audio_worker.py`
are the thin layers that plug real audio into this.
"""
from dataclasses import dataclass, field

from analyzer.config import AUDIO, TEXT


@dataclass
class Word:
    text: str
    start: float  # seconds
    end: float    # seconds


@dataclass
class Pause:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_long(self) -> bool:
        return self.duration >= AUDIO.long_pause_threshold_s


def find_pauses(words: list[Word], segment_start: float | None = None,
                 segment_end: float | None = None) -> list[Pause]:
    """Gaps between consecutive words at or above the pause threshold.

    If segment_start/segment_end are given, a leading or trailing silence
    within the segment also counts as a pause (useful once questions are
    segmented; harmless to omit for a quick whole-clip check).
    """
    if not words:
        return []
    ordered = sorted(words, key=lambda w: w.start)
    pauses = []

    if segment_start is not None and ordered[0].start - segment_start >= AUDIO.pause_threshold_s:
        pauses.append(Pause(segment_start, ordered[0].start))

    for a, b in zip(ordered, ordered[1:]):
        gap = b.start - a.end
        if gap >= AUDIO.pause_threshold_s:
            pauses.append(Pause(a.end, b.start))

    if segment_end is not None and segment_end - ordered[-1].end >= AUDIO.pause_threshold_s:
        pauses.append(Pause(ordered[-1].end, segment_end))

    return pauses


def words_per_minute(words: list[Word], duration_s: float) -> float:
    """Speaking rate over the given duration. Returns 0.0 for zero/negative duration."""
    if duration_s <= 0 or not words:
        return 0.0
    return len(words) / (duration_s / 60.0)


def _normalise(text: str) -> str:
    return text.strip().lower().strip(".,!?;:\"'")


def count_fillers(words: list[Word]) -> int:
    """Counts single-word fillers (um, uh, like, ...). Multi-word fillers
    such as "you know" are matched against consecutive-word pairs too."""
    tokens = [_normalise(w.text) for w in words]
    count = 0
    i = 0
    multi = [f for f in TEXT.filler_words if " " in f]
    single = set(f for f in TEXT.filler_words if " " not in f)
    while i < len(tokens):
        matched_multi = False
        for phrase in multi:
            phrase_tokens = phrase.split()
            n = len(phrase_tokens)
            if tokens[i:i + n] == phrase_tokens:
                count += 1
                i += n
                matched_multi = True
                break
        if matched_multi:
            continue
        if tokens[i] in single:
            count += 1
        i += 1
    return count


def count_hedges(text: str) -> int:
    """Counts hedge phrases in a block of transcript text (case-insensitive,
    substring match — simple and transparent, good enough for a rule-based
    layer; see synopsis Sec 5.4)."""
    lowered = text.lower()
    return sum(lowered.count(phrase) for phrase in TEXT.hedge_words)


@dataclass
class SegmentStats:
    word_count: int
    duration_s: float
    wpm: float
    pauses: list[Pause]
    pause_rate: float          # seconds paused / seconds total
    longest_pause_s: float
    filler_count: int
    filler_rate: float         # fillers per 100 words
    hedge_count: int


def segment_stats(words: list[Word], segment_start: float, segment_end: float,
                   full_text: str | None = None) -> SegmentStats:
    duration = max(segment_end - segment_start, 1e-9)
    pauses = find_pauses(words, segment_start, segment_end)
    paused_time = sum(p.duration for p in pauses)
    fillers = count_fillers(words)
    text = full_text if full_text is not None else " ".join(w.text for w in words)
    return SegmentStats(
        word_count=len(words),
        duration_s=duration,
        wpm=words_per_minute(words, duration),
        pauses=pauses,
        pause_rate=paused_time / duration,
        longest_pause_s=max((p.duration for p in pauses), default=0.0),
        filler_count=fillers,
        filler_rate=(fillers / len(words) * 100) if words else 0.0,
        hedge_count=count_hedges(text),
    )
