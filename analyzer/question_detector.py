"""Marks each new question automatically, from the interviewer's microphone.

Instead of the interviewer typing the question and pressing "Mark next
question", this listens to their mic, splits it into speaking turns (a turn
ends after a stretch of silence), transcribes each turn, and -- if it reads as
a question (`questions.looks_like_question`) -- writes the same question marker
the button would, with the wording filled in. The wording is what enables the
answer-relevance score.

The interviewer's words are only used for this: they are never written to the
transcript, so they can't be mistaken for the candidate's answer or skew the
candidate's speech metrics.

It also passes every chunk on to the `InterviewerMonitor` so the existing
speaking-fraction metric keeps being written.

Limits: it needs the interviewer's voice on the microphone channel, so use
headphones (otherwise the candidate's voice leaks into the mic and their
answers get transcribed as "interviewer speech"; the phrasing check limits
the damage but can't remove it). A question asked in two halves with a long
pause in between may be missed; the manual button always still works.
"""
from __future__ import annotations

import logging

import numpy as np

from analyzer.config import AUDIO
from analyzer.questions import looks_like_question, mark_question

logger = logging.getLogger(__name__)


class InterviewerQuestionDetector:
    """Feed it the interviewer's mic audio with `push_audio(chunk, t)` (`t` =
    session-clock time at the *end* of the chunk, as `AudioSource` passes it).
    Run it behind a `QueuedConsumer`: turn detection and transcription are
    too slow for the audio-capture thread."""

    def __init__(self, log, vad, transcriber, monitor=None,
                 sample_rate: int = AUDIO.sample_rate,
                 end_silence_s: float = 1.5, min_speech_s: float = 1.0,
                 max_turn_s: float = 30.0, check_every_s: float = 0.75,
                 preroll_s: float = 0.3):
        self.log = log
        self.vad = vad
        self.transcriber = transcriber
        self.monitor = monitor
        self.sample_rate = sample_rate
        self.end_silence_s = end_silence_s
        self.min_speech_s = min_speech_s
        self.max_turn_s = max_turn_s
        self.check_every_s = check_every_s
        self.preroll_s = preroll_s

        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start_t: float | None = None
        self._since_check = 0.0
        self.questions_marked = 0

    def push_audio(self, chunk: np.ndarray, t: float) -> None:
        if self.monitor is not None:
            self.monitor.push_audio(chunk, t)
        chunk_s = len(chunk) / self.sample_rate
        if self._buffer_start_t is None:
            self._buffer_start_t = t - chunk_s
        self._buffer = np.concatenate([self._buffer, np.asarray(chunk, dtype=np.float32)])
        self._since_check += chunk_s
        if self._since_check >= self.check_every_s:
            self._since_check = 0.0
            self._check()

    def _consume(self, seconds: float) -> None:
        """Drop the first `seconds` of the buffer."""
        n = min(len(self._buffer), int(seconds * self.sample_rate))
        self._buffer = self._buffer[n:]
        self._buffer_start_t += n / self.sample_rate

    def _check(self) -> None:
        duration = len(self._buffer) / self.sample_rate
        spans = self.vad.speech_spans(self._buffer)
        if not spans:
            if duration > 2.0:
                self._consume(duration - 1.0)  # forget old silence, keep a short lead-in
            return
        last_end = spans[-1].end
        if duration - last_end < self.end_silence_s and duration < self.max_turn_s:
            return  # still speaking, or the pause hasn't lasted long enough to end the turn
        speech_s = sum(s.end - s.start for s in spans)
        if speech_s >= self.min_speech_s:
            a = max(0.0, spans[0].start - self.preroll_s)
            b = min(duration, last_end + 0.3)
            self._handle_turn(self._buffer[int(a * self.sample_rate):int(b * self.sample_rate)],
                              self._buffer_start_t + a)
        self._consume(duration)

    def _handle_turn(self, audio: np.ndarray, t_start: float) -> None:
        words = self.transcriber.transcribe_chunk(audio, offset_s=t_start)
        text = " ".join(w.text for w in words).strip()
        if text and looks_like_question(text):
            n = mark_question(self.log, text[:300], t_start)
            self.questions_marked += 1
            logger.info("Auto-marked question %d at t=%.1fs: %s", n, t_start, text[:80])
