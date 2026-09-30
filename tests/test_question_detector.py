"""Tests for automatic question detection from the interviewer's mic.
A fake VAD (loud samples = speech) and a fake transcriber stand in for the
real models, so these need no audio hardware or GPU."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.audio_features import Word
from analyzer.events import EventLog
from analyzer.question_detector import InterviewerQuestionDetector
from analyzer.questions import looks_like_question, read_markers
from analyzer.vad import SpeechSpan

SR = 16_000


class FakeVAD:
    """Speech = samples louder than 0.1, in 0.1 s steps; adjacent steps merge."""
    def speech_spans(self, audio, offset_s=0.0):
        step = SR // 10
        loud = [bool(np.abs(audio[i:i + step]).max() > 0.1) for i in range(0, len(audio), step)]
        spans, start = [], None
        for i, l in enumerate(loud + [False]):
            if l and start is None:
                start = i
            elif not l and start is not None:
                spans.append(SpeechSpan(start / 10 + offset_s, i / 10 + offset_s))
                start = None
        return spans


class FakeTranscriber:
    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = 0

    def transcribe_chunk(self, audio, offset_s=0.0):
        self.calls += 1
        text = self.texts.pop(0) if self.texts else "hmm"
        return [Word(text=w, start=offset_s, end=offset_s + 0.1) for w in text.split()]


def _feed(det, seconds_of_speech_then_silence):
    """`[(speech_s, silence_s), ...]` streamed in 0.25 s chunks; returns end time."""
    t = 0.0
    for speech_s, silence_s in seconds_of_speech_then_silence:
        for amp, secs in ((0.5, speech_s), (0.0, silence_s)):
            for _ in range(int(secs / 0.25)):
                t += 0.25
                det.push_audio(np.full(int(0.25 * SR), amp, dtype=np.float32), t)
    return t


def _detector(tmp_path, transcriber):
    log = EventLog(tmp_path)
    return log, InterviewerQuestionDetector(log, FakeVAD(), transcriber)


def test_a_spoken_question_is_marked_with_its_wording_and_start_time(tmp_path):
    log, det = _detector(tmp_path, FakeTranscriber("Tell me about yourself?"))
    _feed(det, [(1.0, 0.0), (3.0, 2.0)])  # 1 s of silence-less lead is fine; then a 3 s turn, 2 s pause
    markers = read_markers(log.recent_events())
    assert [m.text for m in markers] == ["Tell me about yourself?"]
    assert markers[0].question_id == 1
    assert markers[0].t_start < 1.5  # near where the speech began, not when detection fired


def test_a_statement_is_not_marked(tmp_path):
    log, det = _detector(tmp_path, FakeTranscriber("Thanks for coming in today."))
    _feed(det, [(3.0, 2.0)])
    assert read_markers(log.recent_events()) == []


def test_a_short_blip_is_ignored_without_transcribing(tmp_path):
    tr = FakeTranscriber("Can you hear me?")
    log, det = _detector(tmp_path, tr)
    _feed(det, [(0.5, 3.0)])  # under the 1 s minimum
    assert tr.calls == 0 and read_markers(log.recent_events()) == []


def test_consecutive_questions_are_numbered_in_order(tmp_path):
    log, det = _detector(tmp_path, FakeTranscriber("What is your biggest strength?", "Okay.",
                                                   "Describe a time you led a team"))
    _feed(det, [(3.0, 2.0), (2.0, 2.0), (3.0, 2.0)])
    markers = read_markers(log.recent_events())
    assert [(m.question_id, m.text) for m in markers] == [
        (1, "What is your biggest strength?"), (2, "Describe a time you led a team")]


def test_nothing_is_transcribed_before_the_pause_ends_the_turn(tmp_path):
    tr = FakeTranscriber("Why do you want this job?")
    log, det = _detector(tmp_path, tr)
    _feed(det, [(3.0, 0.5)])  # only 0.5 s of quiet: the interviewer may still be mid-question
    assert tr.calls == 0


def test_interviewer_words_are_never_written_to_the_transcript(tmp_path):
    log, det = _detector(tmp_path, FakeTranscriber("Tell me about your last job"))
    _feed(det, [(3.0, 2.0)])
    assert log.all_transcript() == []


def test_looks_like_question():
    assert looks_like_question("So how did that go?")
    assert looks_like_question("Tell me about a time you failed")
    assert looks_like_question("Could you walk me through your resume")
    assert not looks_like_question("Great, thanks for that answer.")
    assert not looks_like_question("What I did was refactor the code")  # candidate-style, no '?'
    assert not looks_like_question("Okay?")  # too short to be a real question
