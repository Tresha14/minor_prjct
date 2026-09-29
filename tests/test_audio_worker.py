"""Tests the silence-aware chunking logic in AudioWorker using fake VAD and
transcriber implementations, so these run fast with no GPU, no model
download, and no real audio device.

The real Transcriber + VAD are checked by hand against actual audio (see
README.md, "Known gotchas") -- that's how the bug these tests guard against
was originally found: a fixed-time chunk boundary was silently dropping
words because Whisper skips speech that starts a chunk with no leading
silence for context. These tests exist so that bug can't come back
unnoticed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.audio_features import Word
from analyzer.audio_worker import AudioWorker
from analyzer.events import EventLog
from analyzer.vad import SpeechSpan

SR = 16_000


class FakeTranscriber:
    """Given a master list of (absolute-time) words, returns whichever fall
    inside the window passed to transcribe_chunk -- mimics a real
    transcriber without touching a GPU or a model.
    """

    def __init__(self, master_words: list[Word]):
        self.master_words = master_words
        self.calls = []  # (offset_s, window_duration) for assertions

    def transcribe_chunk(self, audio_array, offset_s: float = 0.0, language: str = "en"):
        window_duration = len(audio_array) / SR
        self.calls.append((offset_s, window_duration))
        window_end = offset_s + window_duration
        return [w for w in self.master_words if offset_s <= w.start < window_end]


class FakeVAD:
    """Treats any run of samples with amplitude above a threshold as
    speech -- a simple, deterministic stand-in for Silero VAD so the
    cut-point logic can be tested without loading a real model.
    """

    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold

    def speech_spans(self, audio: np.ndarray) -> list[SpeechSpan]:
        is_speech = np.abs(audio) > self.threshold
        spans = []
        start = None
        for i, v in enumerate(is_speech):
            if v and start is None:
                start = i
            elif not v and start is not None:
                spans.append(SpeechSpan(start / SR, i / SR))
                start = None
        if start is not None:
            spans.append(SpeechSpan(start / SR, len(audio) / SR))
        return spans


def synth_audio(speech_spans_s: list[tuple[float, float]], total_seconds: float) -> np.ndarray:
    """Builds an array that's silent (zeros) except during the given
    [start, end) second ranges, where it's 0.5 amplitude -- "speech" as far
    as FakeVAD is concerned.
    """
    audio = np.zeros(int(total_seconds * SR), dtype=np.float32)
    for start, end in speech_spans_s:
        audio[int(start * SR):int(end * SR)] = 0.5
    return audio


def make_log(tmp_path) -> EventLog:
    log = EventLog(tmp_path)
    log.session_start(t0=0.0)
    return log


def push_in_pieces(worker: AudioWorker, audio: np.ndarray, piece_seconds: float = 1.0):
    piece_samples = int(piece_seconds * SR)
    first = True
    for i in range(0, len(audio), piece_samples):
        chunk = audio[i:i + piece_samples]
        worker.push_audio(chunk, t=0.0 if first else None)
        first = False


def test_cuts_at_silence_at_or_after_target_not_mid_speech(tmp_path):
    # continuous speech 0-10s (crosses target_chunk_s=8), silence 10-10.5s,
    # more speech 10.5-14s.
    audio = synth_audio([(0.0, 10.0), (10.5, 14.0)], total_seconds=15.0)
    log = make_log(tmp_path)
    transcriber = FakeTranscriber([])  # word content doesn't matter for this test
    worker = AudioWorker(log, transcriber, FakeVAD(), sample_rate=SR,
                          target_chunk_s=8.0, max_chunk_s=20.0, min_chunk_s=2.0)

    push_in_pieces(worker, audio, piece_seconds=1.0)
    worker.flush()

    # the first cut must land at the silence (t=10.0), never mid-speech
    # (i.e. never strictly between 0 and 10, and never inside 10.5-14)
    first_offset, first_duration = transcriber.calls[0]
    first_cut = first_offset + first_duration
    assert first_cut == pytest.approx(10.0, abs=0.05), (
        f"expected the first cut at the silence (t=10.0), got t={first_cut}"
    )


def test_forced_cut_at_max_chunk_when_no_silence_appears(tmp_path):
    # 20 seconds of continuous speech, no silence anywhere.
    audio = synth_audio([(0.0, 20.0)], total_seconds=20.0)
    log = make_log(tmp_path)
    transcriber = FakeTranscriber([])
    worker = AudioWorker(log, transcriber, FakeVAD(), sample_rate=SR,
                          target_chunk_s=8.0, max_chunk_s=15.0, min_chunk_s=2.0)

    push_in_pieces(worker, audio, piece_seconds=1.0)
    worker.flush()

    first_offset, first_duration = transcriber.calls[0]
    assert first_offset + first_duration == pytest.approx(15.0, abs=0.05), (
        "with no silence available, the worker must still force a cut at max_chunk_s "
        "so one long monologue can't stall transcription indefinitely"
    )


def test_no_words_lost_across_a_silence_based_cut(tmp_path):
    # words on either side of a natural pause; the cut should land in that
    # pause, and every word should still come out exactly once.
    master = [
        Word("hello", 0.2, 0.6), Word("there", 0.7, 1.1),           # first utterance
        Word("world", 9.0, 9.4), Word("again", 9.5, 9.9),           # second utterance
    ]
    audio = synth_audio([(0.0, 1.2), (8.8, 10.0)], total_seconds=12.0)
    log = make_log(tmp_path)
    transcriber = FakeTranscriber(master)
    worker = AudioWorker(log, transcriber, FakeVAD(), sample_rate=SR,
                          target_chunk_s=8.0, max_chunk_s=15.0, min_chunk_s=2.0)

    push_in_pieces(worker, audio, piece_seconds=1.0)
    worker.flush()

    got = [w.text for w in worker.all_words]
    assert got == ["hello", "there", "world", "again"]


def test_words_written_to_event_log(tmp_path):
    master = [Word("hello", 0.2, 0.6), Word("world", 0.7, 1.1)]
    audio = synth_audio([(0.0, 1.2)], total_seconds=3.0)
    log = make_log(tmp_path)
    worker = AudioWorker(log, FakeTranscriber(master), FakeVAD(), sample_rate=SR,
                          target_chunk_s=8.0, max_chunk_s=15.0, min_chunk_s=2.0)
    push_in_pieces(worker, audio)
    worker.flush()

    rows = log.all_transcript()
    assert [r[2] for r in rows] == ["hello", "world"]


def test_long_pause_is_logged_as_event(tmp_path):
    # a 4-second gap between "hello" and "world" should trigger a long_pause event
    master = [Word("hello", 0.0, 0.4), Word("world", 4.4, 4.8)]
    audio = synth_audio([(0.0, 0.4), (4.4, 4.8)], total_seconds=6.0)
    log = make_log(tmp_path)
    worker = AudioWorker(log, FakeTranscriber(master), FakeVAD(), sample_rate=SR,
                          target_chunk_s=8.0, max_chunk_s=15.0, min_chunk_s=2.0)
    push_in_pieces(worker, audio)
    worker.flush()

    events = log.recent_events()
    long_pauses = [e for e in events if e[3] == "long_pause"]
    assert len(long_pauses) == 1


def test_metrics_written_for_each_confirmed_segment(tmp_path):
    master = [Word(f"w{i}", i * 0.3, i * 0.3 + 0.2) for i in range(15)]
    audio = synth_audio([(0.0, 4.7)], total_seconds=6.0)
    log = make_log(tmp_path)
    worker = AudioWorker(log, FakeTranscriber(master), FakeVAD(), sample_rate=SR,
                          target_chunk_s=8.0, max_chunk_s=15.0, min_chunk_s=2.0)
    push_in_pieces(worker, audio)
    worker.flush()

    wpm_rows = log.recent_metrics("wpm")
    assert len(wpm_rows) >= 1


def test_near_silent_chunk_is_never_sent_to_the_transcriber(tmp_path):
    # 20 seconds with only a faint, sub-threshold hum -- FakeVAD (threshold
    # 0.01) detects no speech at all in it, mimicking the real bug found by
    # testing against a live mic picking up nothing but room noise: Whisper
    # hallucinates short phrases ("You", "Thank you") on audio like this if
    # it's sent at all.
    audio = np.full(20 * SR, 0.001, dtype=np.float32)
    log = make_log(tmp_path)
    transcriber = FakeTranscriber([])
    worker = AudioWorker(log, transcriber, FakeVAD(), sample_rate=SR,
                          target_chunk_s=8.0, max_chunk_s=15.0, min_chunk_s=2.0)

    push_in_pieces(worker, audio, piece_seconds=1.0)
    worker.flush()

    assert transcriber.calls == [], "a near-silent chunk should never reach the transcriber"
    assert log.recent_metrics("wpm") == []


def test_chunk_with_real_speech_is_still_sent_even_with_some_silence(tmp_path):
    # sanity check the gate isn't overly aggressive: a normal chunk (mostly
    # speech, some genuine pauses) must still reach the transcriber.
    audio = synth_audio([(0.0, 3.0), (4.0, 7.0)], total_seconds=9.0)  # ~67% speech
    log = make_log(tmp_path)
    transcriber = FakeTranscriber([])
    worker = AudioWorker(log, transcriber, FakeVAD(), sample_rate=SR,
                          target_chunk_s=8.0, max_chunk_s=15.0, min_chunk_s=2.0)

    push_in_pieces(worker, audio, piece_seconds=1.0)
    worker.flush()

    assert len(transcriber.calls) >= 1


def test_invalid_chunk_bounds_raise():
    with pytest.raises(ValueError):
        AudioWorker(event_log=None, transcriber=None, vad=None,
                    target_chunk_s=5.0, max_chunk_s=3.0)  # max < target
    with pytest.raises(ValueError):
        AudioWorker(event_log=None, transcriber=None, vad=None,
                    min_chunk_s=10.0, target_chunk_s=5.0)  # min > target
