"""Voice-activity detection (Silero VAD).

Runs on the CPU and is cheap, so it can mark speaking/silent stretches in
real time even before the (slower) transcript for that stretch is ready.
This gives the live dashboard something to show immediately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SpeechSpan:
    start: float
    end: float


class VoiceActivityDetector:
    def __init__(self, sample_rate: int = 16_000):
        from silero_vad import load_silero_vad, get_speech_timestamps
        self._get_speech_timestamps = get_speech_timestamps
        self.model = load_silero_vad()
        self.sample_rate = sample_rate

    def speech_spans(self, audio: np.ndarray, offset_s: float = 0.0) -> list[SpeechSpan]:
        """audio: float32 mono samples at self.sample_rate, range [-1, 1]."""
        timestamps = self._get_speech_timestamps(audio, self.model, sampling_rate=self.sample_rate)
        return [
            SpeechSpan(start=t["start"] / self.sample_rate + offset_s,
                       end=t["end"] / self.sample_rate + offset_s)
            for t in timestamps
        ]

    def speaking_fraction(self, audio: np.ndarray) -> float:
        spans = self.speech_spans(audio)
        if len(audio) == 0:
            return 0.0
        speaking = sum(s.end - s.start for s in spans)
        total = len(audio) / self.sample_rate
        return speaking / total if total > 0 else 0.0
