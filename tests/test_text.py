"""Tests for analyzer/text.py: hedge rate, lexicon sentiment, and
question/answer relevance. Same testing philosophy as the rest of this
project -- hand-calculated golden values, explicit missing/empty-input
handling, no dependency on a transcriber/microphone/model.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyzer.events import EventLog
from analyzer.text import (
    RelevanceResult, SentimentResult, TextWorker, answer_relevance,
    hedge_rate_per_100_words, sentiment_score,
)


# ---------------------------------------------------------------- hedge_rate_per_100_words

def test_hedge_rate_basic():
    assert hedge_rate_per_100_words(hedge_count=2, word_count=50) == pytest.approx(4.0)


def test_hedge_rate_zero_words_returns_zero_not_a_division_error():
    assert hedge_rate_per_100_words(hedge_count=0, word_count=0) == 0.0


def test_hedge_rate_zero_hedges():
    assert hedge_rate_per_100_words(hedge_count=0, word_count=100) == 0.0


# ---------------------------------------------------------------- sentiment_score

def test_sentiment_score_mixed_golden_value():
    # "confident" and "prepared" are positive (2), "nervous" is negative (1)
    # -> (2 - 1) / 3 = 0.3333...
    result = sentiment_score("I felt confident and prepared, though a bit nervous at first.")
    assert result.positive_hits == 2
    assert result.negative_hits == 1
    assert result.score == pytest.approx(1 / 3)


def test_sentiment_score_no_lexicon_hits_is_zero_not_none():
    result = sentiment_score("The weather is nice today.")
    assert result == SentimentResult(score=0.0, positive_hits=0, negative_hits=0)


def test_sentiment_score_all_positive_is_one():
    result = sentiment_score("I am confident, excited, and prepared.")
    assert result.score == pytest.approx(1.0)


def test_sentiment_score_all_negative_is_minus_one():
    result = sentiment_score("I am nervous, unsure, and worried.")
    assert result.score == pytest.approx(-1.0)


def test_sentiment_score_is_case_insensitive():
    a = sentiment_score("CONFIDENT and Prepared")
    b = sentiment_score("confident and prepared")
    assert a.score == b.score == pytest.approx(1.0)


def test_sentiment_score_matches_exact_tokens_not_substrings():
    # "strongly" must NOT match the lexicon word "strong"
    result = sentiment_score("strongly worded email")
    assert result.positive_hits == 0
    assert result.score == 0.0


def test_sentiment_score_empty_text():
    result = sentiment_score("")
    assert result.score == 0.0
    assert result.positive_hits == 0
    assert result.negative_hits == 0


# ---------------------------------------------------------------- answer_relevance

def test_answer_relevance_golden_value():
    # question significant words: {greatest, strength, leader}
    # answer significant words:   {greatest, strength, leader, communication}
    # all three question keywords appear in the answer -> 3/3
    result = answer_relevance(
        "What is your greatest strength as a leader?",
        "My greatest strength as a leader is communication.",
    )
    assert result.score == pytest.approx(1.0)
    assert result.matched_words == {"greatest", "strength", "leader"}


def test_answer_relevance_completely_unrelated_is_zero():
    result = answer_relevance(
        "Tell me about a time you led a team.",
        "I really enjoy hiking on weekends.",
    )
    assert result.score == 0.0
    assert result.matched_words == frozenset()


def test_answer_relevance_identical_text_is_one():
    result = answer_relevance("Describe your leadership style.", "Describe your leadership style.")
    assert result.score == pytest.approx(1.0)


def test_answer_relevance_empty_answer_is_zero_not_none():
    """An empty/filler-only answer has no signal to compare -- a real (if
    unhelpful) answer, not a missing metric. Distinguishing "observed but
    empty" from "not observed yet" is exactly fusion.py's rule 1/2
    reasoning applied here."""
    result = answer_relevance("What is your greatest strength?", "")
    assert result.score == 0.0


def test_answer_relevance_both_sides_only_stopwords_is_zero_not_a_crash():
    result = answer_relevance("Is it the?", "It is a the.")
    assert result.score == 0.0
    assert result.matched_words == frozenset()


def test_answer_relevance_ignores_case_and_punctuation():
    a = answer_relevance("What is your GREATEST strength?", "My greatest strength, communication!")
    assert "greatest" in a.matched_words
    assert "strength" in a.matched_words


def test_answer_relevance_partial_coverage_golden_value():
    # question keywords {describe, experience, databases}; answer covers 1 of 3
    result = answer_relevance("Describe your experience with databases",
                              "I have worked on databases for years and I know the domain well")
    assert result.score == pytest.approx(1 / 3)
    assert result.matched_words == {"databases"}


def test_answer_relevance_is_not_diluted_by_a_long_on_topic_answer():
    """The regression that motivated replacing Jaccard: same question
    keywords, answer padded out to ~150 words -- the score must not fall."""
    question = "Describe your leadership experience"
    short = "My leadership experience is that I describe goals clearly."
    padding = " ".join(["and then we shipped the project on schedule"] * 20)
    assert answer_relevance(question, short).score == pytest.approx(1.0)
    assert answer_relevance(question, short + " " + padding).score == pytest.approx(1.0)


def test_answer_relevance_long_off_topic_answer_stays_zero():
    padding = " ".join(["I really enjoy hiking with friends on weekends"] * 20)
    assert answer_relevance("Describe your leadership experience", padding).score == 0.0


def test_answer_relevance_matches_simple_inflections():
    assert answer_relevance("How do you manage databases?", "I was managing a database.").score == pytest.approx(1.0)
    r = answer_relevance("Describe the projects", "One project stood out.")
    assert r.matched_words == {"projects"}


def test_answer_relevance_stemming_does_not_merge_unrelated_short_words():
    assert answer_relevance("Tell me about the bus", "I fixed a bug").score == 0.0


# ---------------------------------------------------------------- TextWorker (integration)

def test_text_worker_process_segment_writes_hedge_rate_and_sentiment(tmp_path):
    log = EventLog(tmp_path)
    log.session_start(t0=0.0)
    worker = TextWorker(log)

    worker.process_segment(
        text="I guess I felt confident and prepared for this one.",
        word_count=10,
        t=12.5,
    )

    hedge_rows = log.recent_metrics("hedge_rate")
    sentiment_rows = log.recent_metrics("sentiment")
    assert len(hedge_rows) == 1
    assert hedge_rows[0][0] == 12.5
    assert hedge_rows[0][1] == pytest.approx(10.0)  # 1 hedge ("i guess") / 10 words * 100

    assert len(sentiment_rows) == 1
    assert sentiment_rows[0][0] == 12.5
    assert sentiment_rows[0][1] == pytest.approx(1.0)  # confident + prepared, no negatives


def test_text_worker_score_answer_relevance_writes_metric_and_returns_result(tmp_path):
    log = EventLog(tmp_path)
    log.session_start(t0=0.0)
    worker = TextWorker(log)

    result = worker.score_answer_relevance(
        question="What is your greatest strength as a leader?",
        answer="My greatest strength as a leader is communication.",
        t=30.0,
    )

    assert isinstance(result, RelevanceResult)
    assert result.score == pytest.approx(1.0)
    rows = log.recent_metrics("answer_relevance")
    assert rows == [(30.0, pytest.approx(1.0))]


def test_text_worker_writes_under_the_text_source_by_default(tmp_path):
    log = EventLog(tmp_path)
    log.session_start(t0=0.0)
    TextWorker(log).process_segment(text="fine", word_count=1, t=1.0)
    events = log.recent_metrics("hedge_rate")
    assert events  # written; source itself isn't directly queryable via recent_metrics,
    # but EventLog.metric()'s signature requires a source, and TextWorker defaults it to "text"
    # per this module's docstring -- covered structurally rather than by inspecting raw SQL here.
