"""Tests the parts of capture.py that don't need a real camera or audio
device: the mono-conversion helper and InterviewerMonitor's interval
logic. VideoSource / AudioSource / SessionCapture themselves are threading
+ hardware glue and are verified by manual smoke test instead (see
README.md status section) -- there's little a fake-hardware unit test would
catch that the (already well-tested) VideoWorker/AudioWorker underneath
wouldn't.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.capture import AudioSource, InterviewerMonitor, QueuedConsumer, to_mono_float32
from analyzer.events import EventLog


def test_to_mono_float32_averages_channels():
    stereo = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
    mono = to_mono_float32(stereo)
    assert mono.shape == (2,)
    assert mono[0] == pytest.approx(0.0)
    assert mono[1] == pytest.approx(0.0)


def test_to_mono_float32_passes_through_already_mono():
    mono_in = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    mono_out = to_mono_float32(mono_in)
    np.testing.assert_allclose(mono_out, mono_in)


def test_to_mono_float32_handles_single_channel_2d():
    single_channel = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
    mono = to_mono_float32(single_channel)
    np.testing.assert_allclose(mono, [0.1, 0.2, 0.3])


class FakeVAD:
    """Returns a fixed speaking fraction regardless of input -- enough to
    test InterviewerMonitor's interval-batching logic without loading a
    real Silero model."""
    def __init__(self, fraction: float):
        self.fraction = fraction

    def speaking_fraction(self, audio: np.ndarray) -> float:
        return self.fraction


def test_interviewer_monitor_writes_metric_once_per_interval(tmp_path):
    log = EventLog(tmp_path)
    log.session_start(t0=0.0)
    monitor = InterviewerMonitor(log, FakeVAD(0.4), interval_s=1.0)

    # push small chunks for 2.5 "seconds" of session time
    for i in range(25):
        monitor.push_audio(np.zeros(100, dtype=np.float32), t=i * 0.1)

    rows = log.recent_metrics("speaking_fraction")
    # roughly one metric per second over 2.5s -> 2 flushes
    assert 2 <= len(rows) <= 3
    assert all(v == pytest.approx(0.4) for (_t, v) in rows)


def test_interviewer_monitor_no_metric_before_first_interval(tmp_path):
    log = EventLog(tmp_path)
    log.session_start(t0=0.0)
    monitor = InterviewerMonitor(log, FakeVAD(0.5), interval_s=5.0)

    monitor.push_audio(np.zeros(100, dtype=np.float32), t=0.5)
    monitor.push_audio(np.zeros(100, dtype=np.float32), t=1.0)

    assert log.recent_metrics("speaking_fraction") == []


def test_queued_consumer_processes_items_in_order():
    results = []
    consumer = QueuedConsumer(lambda x: results.append(x))
    consumer.start()
    for i in range(10):
        consumer.put(i)
    consumer.stop(timeout=2.0)
    assert results == list(range(10))


def test_queued_consumer_decouples_slow_handler_from_producer():
    """The whole point of QueuedConsumer: put() must return immediately
    even if the handler itself is slow -- a producer thread (e.g. audio
    capture) must never block waiting for a slow consumer (e.g. GPU
    transcription). See the module docstring for the real bug this fixes.
    """
    def slow_handler(x):
        time.sleep(0.05)

    consumer = QueuedConsumer(slow_handler)
    consumer.start()
    t0 = time.time()
    for i in range(20):
        consumer.put(i)
    put_duration = time.time() - t0
    consumer.stop(timeout=5.0)

    # 20 puts should take milliseconds, not the ~1s the handler alone would need
    assert put_duration < 0.2


def test_queued_consumer_drains_remaining_items_on_stop():
    results = []
    consumer = QueuedConsumer(lambda x: results.append(x))
    consumer.start()
    for i in range(5):
        consumer.put(i)
    consumer.stop(timeout=2.0)  # must wait for the queue to fully drain
    assert results == [0, 1, 2, 3, 4]


def test_audio_source_records_why_it_stopped_instead_of_dying_silently():
    class BrokenDevice:
        def recorder(self, samplerate):
            raise RuntimeError("Error 0x8889000a")

    src = AudioSource(BrokenDevice())
    src.start(lambda chunk, t: None, time.time())
    src._thread.join(timeout=2)
    assert src.error is not None and "0x8889000a" in src.error
