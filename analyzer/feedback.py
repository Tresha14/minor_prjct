"""Turns fusion scores' raw inputs into specific, worded coaching tips.

Week 6 of the plan. Deliberately takes the same FusionInputs fusion.py
scores (not a second, separate reading of the event log) so a tip's raw
number is always the exact figure that fed the score it explains.

Same rule fusion.py enforces, applied here to tips instead of numbers: a
metric that hasn't been observed yet produces no tip, never a guessed or
generic one. If literally nothing has been observed yet, a single honest
"not enough data yet" tip is returned instead of an empty list, so a
caller always has something to render -- but that placeholder tip is
itself just a true statement about the data, not a fabricated observation.

If a language model is ever used to *phrase* tips more naturally, it must
only see the already-computed metrics (the same inputs this module uses)
as input, never raw video/audio -- the scoring and the decision of *which*
signals are worth mentioning stay rule-based here, per the synopsis
Sec 5.7. This module currently does the phrasing itself with fixed
templates; nothing here calls out to an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from analyzer.config import FEEDBACK, FUSION
from analyzer.fusion import FusionInputs, band_score, identity01, linear_lower_is_better

F = FUSION


@dataclass
class Tip:
    """One coaching tip.

    `metric` is the FusionInputs field name this tip is grounded in (None
    only for the "not enough data" / "nothing needs flagging" placeholder
    tips, which aren't about any single metric).
    `severity` is "needs_work", "praise", or "info" (a placeholder, not a
    judgement about a real reading).
    `raw_value` is the exact metric value the tip's wording was computed
    from -- kept alongside `text` so a caller (or a test) can verify the
    wording matches the number without re-parsing the sentence.
    """
    metric: str | None
    severity: str
    text: str
    raw_value: float | None = None


TipFormatter = Callable[[float], str]


@dataclass
class _MetricSpec:
    normalise: Callable[[float], float]
    format_praise: TipFormatter
    # One-directional metrics (pause_rate, filler_rate, hedge_rate,
    # pose_stability, camera_facing) only ever need one of these two --
    # leave the other None. Band-shaped metrics (wpm, blink_rate) set both,
    # plus band_good_low/high, to tell "too low" and "too high" apart.
    format_too_low: TipFormatter | None = None
    format_too_high: TipFormatter | None = None
    band_good_low: float | None = None
    band_good_high: float | None = None


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _rate(x: float) -> str:
    return f"{x:.1f}"


_SPECS: dict[str, _MetricSpec] = {
    "camera_facing": _MetricSpec(
        normalise=lambda v: identity01(v),
        format_too_low=lambda v: (
            f"Camera-facing time was {_pct(v)} -- try holding your gaze on the lens "
            f"for 3-5 seconds at a time instead of glancing away."
        ),
        format_praise=lambda v: f"Camera-facing time was {_pct(v)} -- steady eye contact, keep it up.",
    ),
    "wpm": _MetricSpec(
        normalise=lambda v: band_score(v, F.wpm_floor, F.wpm_good_low, F.wpm_good_high, F.wpm_ceiling),
        format_too_low=lambda v: (
            f"Speaking pace was {v:.0f} words/min, on the slow side -- a comfortable "
            f"conversational pace is roughly {F.wpm_good_low:.0f}-{F.wpm_good_high:.0f} wpm."
        ),
        format_too_high=lambda v: (
            f"Speaking pace was {v:.0f} words/min, quite fast -- slowing down slightly "
            f"(aim for roughly {F.wpm_good_low:.0f}-{F.wpm_good_high:.0f} wpm) can make answers easier to follow."
        ),
        format_praise=lambda v: f"Speaking pace was {v:.0f} words/min -- comfortable and easy to follow.",
        band_good_low=F.wpm_good_low,
        band_good_high=F.wpm_good_high,
    ),
    "pause_rate": _MetricSpec(
        normalise=lambda v: linear_lower_is_better(v, F.pause_rate_good, F.pause_rate_bad),
        format_too_high=lambda v: (
            f"{_pct(v)} of speaking time was spent in pauses -- try outlining an answer "
            f"in your head for a second before starting, to cut down on long silences."
        ),
        format_praise=lambda v: f"Only {_pct(v)} of speaking time was spent pausing -- fluent, confident delivery.",
    ),
    "filler_rate": _MetricSpec(
        normalise=lambda v: linear_lower_is_better(v, F.filler_rate_good, F.filler_rate_bad),
        format_too_high=lambda v: (
            f'{_rate(v)} filler words per 100 (e.g. "um", "like") -- a silent pause '
            f"instead of a filler usually reads as more composed."
        ),
        format_praise=lambda v: f"Filler word usage was low ({_rate(v)} per 100 words) -- clean, confident speech.",
    ),
    "hedge_rate": _MetricSpec(
        normalise=lambda v: linear_lower_is_better(v, F.hedge_rate_good, F.hedge_rate_bad),
        format_too_high=lambda v: (
            f'{_rate(v)} hedging phrases per 100 words (e.g. "I guess", "maybe") -- '
            f"stating the answer more directly can come across as more confident."
        ),
        format_praise=lambda v: f"Hedging language was rare ({_rate(v)} per 100 words) -- direct, confident phrasing.",
    ),
    "pose_stability": _MetricSpec(
        normalise=lambda v: linear_lower_is_better(v, F.pose_stability_good, F.pose_stability_bad),
        format_too_high=lambda v: (
            f"Head movement was noticeably unsteady (~{v:.1f} degrees of variation) -- "
            f"try keeping your head still and centred while answering."
        ),
        format_praise=lambda v: f"Head position was steady (~{v:.1f} degrees of variation) -- good, controlled posture.",
    ),
    "blink_rate": _MetricSpec(
        normalise=lambda v: band_score(v, F.blink_floor, F.blink_good_low, F.blink_good_high, F.blink_ceiling),
        format_too_low=lambda v: (
            f"Blink rate was unusually low ({v:.0f}/min) -- a normal resting rate is "
            f"roughly {F.blink_good_low:.0f}-{F.blink_good_high:.0f}/min; often a sign of "
            f"staring or tension rather than something to consciously fix."
        ),
        format_too_high=lambda v: (
            f"Blink rate was quite high ({v:.0f}/min) -- a normal resting rate is roughly "
            f"{F.blink_good_low:.0f}-{F.blink_good_high:.0f}/min; can be a sign of eye strain or nerves."
        ),
        format_praise=lambda v: f"Blink rate was in a normal, relaxed range ({v:.0f}/min).",
        band_good_low=F.blink_good_low,
        band_good_high=F.blink_good_high,
    ),
}

# Every FusionInputs field must have a tip spec, so a metric added to
# fusion.py's NORMALISABLE_METRICS without also being wired up here fails a
# test instead of silently never producing a tip -- see tests/test_feedback.py.
METRICS_WITH_TIPS = set(_SPECS.keys())


def _needs_work_text(spec: _MetricSpec, raw: float) -> str:
    if spec.band_good_low is not None:
        # Band-shaped metric: figure out which side of the "good" range the
        # low score came from. (A raw value strictly inside [good_low,
        # good_high] always normalises to 1.0, i.e. never reaches here.)
        if raw < spec.band_good_low and spec.format_too_low is not None:
            return spec.format_too_low(raw)
        if spec.format_too_high is not None:
            return spec.format_too_high(raw)
        return spec.format_too_low(raw)  # pragma: no cover - both directions are always defined for band metrics
    formatter = spec.format_too_high or spec.format_too_low
    return formatter(raw)


def generate_tips(inputs: FusionInputs) -> list[Tip]:
    """The core, pure tip-generation function -- no I/O, so it's fast and
    easy to test with hand-picked FusionInputs (mirrors
    fusion.compute_scores()). generate_tips_from_log() below is the thin
    wrapper that feeds it real data.

    Returns up to FEEDBACK.max_tips tips, worst (lowest normalised score)
    first among needs_work tips, followed by praise tips. A metric whose
    normalised score falls between FEEDBACK.needs_work_max and
    FEEDBACK.praise_min is "fine" and gets no tip either way -- silence on
    a middling reading is intentional, not a gap.

    Returns a list with a single informational Tip (metric=None,
    severity="info") instead of an empty list in two honest, distinct
    cases: no metric has been observed at all yet, or every observed
    metric is within a comfortable range. Neither of those is a fabricated
    coaching observation -- both are true statements about the data.
    """
    scored: list[tuple[str, float, float]] = []  # (metric, normalised_score, raw)
    any_data = False
    for name, spec in _SPECS.items():
        raw = getattr(inputs, name)
        if raw is None:
            continue
        any_data = True
        scored.append((name, spec.normalise(raw), raw))

    if not any_data:
        return [Tip(metric=None, severity="info",
                     text="Not enough data yet to generate coaching tips.")]

    needs_work = sorted((s for s in scored if s[1] <= FEEDBACK.needs_work_max), key=lambda s: s[1])
    praise = sorted((s for s in scored if s[1] >= FEEDBACK.praise_min), key=lambda s: -s[1])

    tips = [
        Tip(metric=name, severity="needs_work", text=_needs_work_text(_SPECS[name], raw), raw_value=raw)
        for name, _, raw in needs_work
    ] + [
        Tip(metric=name, severity="praise", text=_SPECS[name].format_praise(raw), raw_value=raw)
        for name, _, raw in praise
    ]

    if not tips:
        return [Tip(metric=None, severity="info",
                     text="All observed signals were within a comfortable range -- no specific coaching tips right now.")]

    return tips[:FEEDBACK.max_tips]


def generate_tips_from_log(log) -> list[Tip]:
    """Builds FusionInputs from an EventLog's most recent readings and
    generates tips from them -- same pattern as fusion.compute_from_log().
    `log` needs the same recent_metrics(name) method (inject a fake, or a
    real EventLog, for tests).
    """
    from analyzer.dashboard_data import latest_value

    inputs = FusionInputs(
        camera_facing=latest_value(log.recent_metrics("camera_facing")),
        pose_stability=latest_value(log.recent_metrics("pose_stability")),
        blink_rate=latest_value(log.recent_metrics("blink_rate")),
        wpm=latest_value(log.recent_metrics("wpm")),
        pause_rate=latest_value(log.recent_metrics("pause_rate")),
        filler_rate=latest_value(log.recent_metrics("filler_rate")),
        hedge_rate=latest_value(log.recent_metrics("hedge_rate")),
    )
    return generate_tips(inputs)
