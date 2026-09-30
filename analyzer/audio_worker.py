"""Live audio pipeline: silence-aware chunked transcription + feature math,
writing continuously to the shared EventLog.

Chunk boundaries are chosen at detected silence (via Silero VAD, analyzer/vad.py),
not at a fixed time offset. That choice came from a real bug, not a guess: an
earlier fixed-8-second-with-1-second-overlap version was tested against real
GPU transcription of a spoken clip and silently dropped several words at
every chunk boundary. The cause: a chunk that starts mid-word, with no
leading silence for acoustic context, doesn't get mis-transcribed by
Whisper -- it gets skipped outright. Cutting only at silence means every
chunk starts right after a pause, so this never happens. See
tests/test_audio_worker.py for the regression test and README.md's
"Known gotchas" section.

Because chunks are cut at silence, they don't overlap, and there's no need
for the overlap/de-duplication logic a fixed-window design would need.

A chunk is also skipped entirely (never sent to Whisper) if the VAD finds
almost no speech in it. This was added after live testing against a real
microphone picking up nothing but ambient room noise: Whisper reliably
hallucinates short phrases -- "You", "Thank you" -- on near-silent audio,
which would otherwise show up in the live transcript and skew WPM/pause
metrics as if the candidate had actually spoken. See config.AUDIO.min_speech_fraction.
"""
from __future__ import annotations

import logging

import numpy as np

from analyzer.audio_features import Word, segment_stats
from analyzer.config import AUDIO
from analyzer.events import EventLog

logger = logging.getLogger(__name__)


class AudioWorker:
    """Feed it audio with `push_audio()`; call `flush()` once at the end of
    a session to process whatever's left in the buffer.

    `transcriber` needs one method:
        transcribe_chunk(audio_array, offset_s=...) -> list[Word]
    (matches analyzer.transcribe.Transcriber -- inject a fake for tests).

    `vad` needs one method:
        speech_spans(audio_array) -> list of objects with .start/.end (seconds,
        relative to the start of audio_array)
    (matches analyzer.vad.VoiceActivityDetector -- inject a fake for tests,
    since loading the real Silero model is comparatively slow to do in every
    unit test).
    """

    def __init__(self, event_log: EventLog, transcriber, vad,
                 source: str = "audio",
                 sample_rate: int = AUDIO.sample_rate,
                 target_chunk_s: float = AUDIO.target_chunk_s,
                 max_chunk_s: float = AUDIO.max_chunk_s,
                 min_chunk_s: float = AUDIO.min_chunk_s,
                 min_speech_fraction: float = AUDIO.min_speech_fraction,
                 text_worker=None):
        """`text_worker` (optional) needs one method:
            process_segment(text, word_count, t)
        (matches analyzer.text.TextWorker). Called once per transcribed
        segment so hedge_rate/sentiment are written alongside wpm etc."""
        if not (0 < min_chunk_s <= target_chunk_s <= max_chunk_s):
            raise ValueError("expected 0 < min_chunk_s <= target_chunk_s <= max_chunk_s")
        self.log = event_log
        self.transcriber = transcriber
        self.vad = vad
        self.source = source
        self.sample_rate = sample_rate
        self.target_chunk_s = target_chunk_s
        self.max_chunk_s = max_chunk_s
        self.min_chunk_s = min_chunk_s
        self.min_speech_fraction = min_speech_fraction
        self.text_worker = text_worker

        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start_t = 0.0
        self._started = False
        self.all_words: list[Word] = []  # kept for convenience/tests

    def push_audio(self, chunk: np.ndarray, t: float | None = None) -> None:
        """Append newly captured audio.

        `t` is the session-clock time of the first sample in `chunk` -- pass
        it on the very first call; omit it on later calls and the buffer is
        just treated as contiguous with what came before.
        """
        if not self._started:
            self._buffer_start_t = t if t is not None else 0.0
            self._started = True
        self._buffer = np.concatenate([self._buffer, np.asarray(chunk, dtype=np.float32)])
        self._drain()

    def _drain(self) -> None:
        while True:
            duration = len(self._buffer) / self.sample_rate
            if duration < self.min_chunk_s:
                return
            cut_sample = self._find_cut_point(duration)
            if cut_sample is None:
                if duration < self.max_chunk_s:
                    return  # wait for more audio, or for a silence to appear
                cut_sample = len(self._buffer)  # forced cut: rare, only for one long monologue
            self._process_window(cut_sample)

    # How much confirmed silence must follow a candidate cut point before we
    # trust it. Without this, a span whose end simply coincides with the end
    # of the buffered-so-far audio looks identical to a real pause -- the
    # VAD has no way to tell "speech stopped here" from "we just haven't
    # received the next audio yet". Requiring a bit of trailing silence
    # inside the buffer rules out the second case.
    _cut_margin_s = 0.2

    def _find_cut_point(self, buffer_duration: float) -> int | None:
        """Returns a sample index to cut the buffer at, chosen to land in a
        silence at or after target_chunk_s -- or None if it's not time to
        cut yet (audio hasn't reached target_chunk_s, or no *confirmed*
        silence has appeared yet and we haven't hit max_chunk_s).
        """
        if buffer_duration < self.target_chunk_s:
            return None

        spans = self.vad.speech_spans(self._buffer)
        if not spans:
            # No detected speech at all (e.g. pure silence): nothing to
            # protect by waiting, so cut at the target length.
            return int(self.target_chunk_s * self.sample_rate)

        # Each span.end is a candidate cut point, but only trust it if the
        # buffer actually contains silence after it -- otherwise it may just
        # be that speech is ongoing and we haven't received the rest yet.
        confirmed_ends = sorted(
            s.end for s in spans
            if s.end >= self.target_chunk_s and (buffer_duration - s.end) >= self._cut_margin_s
        )
        if confirmed_ends:
            return int(confirmed_ends[0] * self.sample_rate)
        return None  # still mid-speech (or unconfirmed) at target_chunk_s; let _drain() decide whether to wait

    def _has_enough_speech(self, window: np.ndarray) -> bool:
        """False if `window` is almost entirely silence/noise -- used to
        skip calling Whisper at all on such chunks, since it otherwise
        tends to hallucinate short phrases rather than return nothing."""
        duration = len(window) / self.sample_rate
        if duration <= 0:
            return False
        spans = self.vad.speech_spans(window)
        speech_time = sum(s.end - s.start for s in spans)
        return (speech_time / duration) >= self.min_speech_fraction

    def _process_window(self, cut_sample: int) -> None:
        window = self._buffer[:cut_sample]
        window_start_t = self._buffer_start_t
        window_end_t = window_start_t + len(window) / self.sample_rate

        if self._has_enough_speech(window):
            words = self.transcriber.transcribe_chunk(window, offset_s=window_start_t)
            if words:
                self._emit_words(words, window_start_t, window_end_t)

        self._buffer = self._buffer[cut_sample:]
        self._buffer_start_t = window_end_t

    def flush(self) -> None:
        """Process whatever's left in the buffer. Call once at session end."""
        if not self._started or len(self._buffer) / self.sample_rate < 0.3:
            return
        window_start_t = self._buffer_start_t
        window_end_t = window_start_t + len(self._buffer) / self.sample_rate
        if self._has_enough_speech(self._buffer):
            words = self.transcriber.transcribe_chunk(self._buffer, offset_s=window_start_t)
            if words:
                self._emit_words(words, window_start_t, window_end_t)
        self._buffer = np.zeros(0, dtype=np.float32)

    def _emit_words(self, words: list[Word], segment_start: float, segment_end: float) -> None:
        for w in words:
            self.log.word(w.text, w.start, w.end)
        self.all_words.extend(words)

        stats = segment_stats(words, segment_start, segment_end)
        self.log.metric(self.source, "wpm", stats.wpm, t=segment_end)
        self.log.metric(self.source, "pause_rate", stats.pause_rate, t=segment_end)
        self.log.metric(self.source, "filler_rate", stats.filler_rate, t=segment_end)

        if self.text_worker is not None:
            self.text_worker.process_segment(" ".join(w.text for w in words), len(words), t=segment_end)

        for pause in stats.pauses:
            if pause.is_long:
                self.log.event(
                    self.source, "long_pause",
                    t_start=pause.start, t_end=pause.end,
                    detail={"duration": round(pause.duration, 2)},
                )
