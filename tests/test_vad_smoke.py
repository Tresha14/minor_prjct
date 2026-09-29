"""Smoke test only: checks the Silero VAD wrapper loads and runs without
crashing. It does not assert on detection accuracy -- that needs real
speech audio, which is checked manually in the week-1 feasibility test.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_vad_loads_and_runs_on_silence():
    pytest.importorskip("silero_vad")
    from analyzer.vad import VoiceActivityDetector

    vad = VoiceActivityDetector()
    silence = np.zeros(16_000 * 2, dtype=np.float32)  # 2s of silence
    fraction = vad.speaking_fraction(silence)
    assert 0.0 <= fraction <= 1.0
    assert fraction == 0.0  # pure silence should never be flagged as speech
