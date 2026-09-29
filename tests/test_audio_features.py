import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.audio_features import (
    Word, find_pauses, words_per_minute, count_fillers, count_hedges, segment_stats,
)


def words_from(specs):
    """specs: list of (text, start, end)"""
    return [Word(t, s, e) for t, s, e in specs]


def test_find_pauses_detects_gap_above_threshold():
    words = words_from([("hello", 0.0, 0.4), ("world", 1.5, 1.9)])  # 1.1s gap
    pauses = find_pauses(words)
    assert len(pauses) == 1
    assert pauses[0].start == 0.4
    assert pauses[0].end == 1.5
    assert round(pauses[0].duration, 2) == 1.1


def test_find_pauses_ignores_short_gap():
    words = words_from([("hello", 0.0, 0.4), ("world", 0.6, 1.0)])  # 0.2s gap, below 0.7 threshold
    assert find_pauses(words) == []


def test_find_pauses_includes_leading_and_trailing_silence():
    words = words_from([("hi", 2.0, 2.3)])
    pauses = find_pauses(words, segment_start=0.0, segment_end=5.0)
    assert len(pauses) == 2  # leading 0->2.0, trailing 2.3->5.0


def test_pause_is_long_flag():
    words = words_from([("a", 0.0, 0.2), ("b", 5.0, 5.2)])  # 4.8s gap
    pauses = find_pauses(words)
    assert pauses[0].is_long


def test_words_per_minute():
    words = words_from([(f"w{i}", i, i + 0.3) for i in range(10)])  # 10 words
    assert words_per_minute(words, 60.0) == 10.0  # 10 words in 60s = 10 wpm
    assert words_per_minute(words, 30.0) == 20.0
    assert words_per_minute([], 30.0) == 0.0
    assert words_per_minute(words, 0.0) == 0.0


def test_count_fillers_single_word():
    words = words_from([("so,", 0, 1), ("um,", 1, 2), ("I", 2, 3), ("think", 3, 4)])
    assert count_fillers(words) == 1  # only "um"


def test_count_fillers_multi_word():
    words = words_from([("you", 0, 1), ("know,", 1, 2), ("it's", 2, 3), ("fine", 3, 4)])
    assert count_fillers(words) == 1  # "you know" as one filler


def test_count_hedges_in_text():
    text = "I think maybe it could be right, I guess."
    n = count_hedges(text)
    assert n >= 3  # "i think", "maybe", "could be", "i guess" -> at least 3 matches


def test_segment_stats_basic():
    words = words_from([("um", 0.0, 0.3), ("hello", 2.0, 2.4), ("world", 2.4, 2.8)])
    stats = segment_stats(words, segment_start=0.0, segment_end=5.0)
    assert stats.word_count == 3
    assert stats.filler_count == 1
    assert stats.longest_pause_s > 0
    assert 0 < stats.pause_rate <= 1
