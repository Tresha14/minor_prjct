"""Rule-based fusion layer: turns raw metrics from the event log into three
interpretable sub-scores -- engagement, fluency, confidence -- each on a
0-100 scale and each traceable back to the named metrics that produced it.

Why rule-based rather than a trained model: the dataset this project can
realistically collect (a handful of mock interviews) is far too small to
train a model responsibly. A transparent, additive score is easier to
defend in a viva, easier to act on ("your pause rate was high" is a useful
sentence; "the model said 62" is not), and its weights can be tuned later
with ridge regression if the data ever supports it -- see the synopsis,
Sec 5.5 and Sec 8.

Two rules are enforced everywhere in this module, because a scoring layer
that quietly fabricates data is worse than one that says nothing:

  1. A raw metric is never invented. If a metric hasn't been observed yet
     (session just started, that modality isn't wired up, the candidate
     hasn't spoken yet this segment...), its weight is dropped from the
     average rather than filled in with a guess.
  2. If every input to a sub-score is missing, that sub-score is None, not
     a fabricated default like 50. Callers must handle None explicitly --
     a missing score has to look different from a bad score, always.

tests/test_fusion.py enforces both rules as regression tests, plus
hand-calculated "golden value" checks and monotonicity sweeps (e.g. more
camera-facing time must never lower engagement) -- this module is the part
of the project a wrong answer would matter most in, so it's tested more
heavily than most.
"""
from __future__ import annotations

from dataclasses import dataclass

from analyzer.config import FUSION


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def linear_lower_is_better(value: float, good: float, bad: float) -> float:
    """1.0 at/below `good`, 0.0 at/above `bad`, linear in between. Use for
    metrics where less is better (pause rate, filler rate, pose
    instability). Requires good < bad.
    """
    if good >= bad:
        raise ValueError(f"good ({good}) must be less than bad ({bad})")
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return _clamp01((bad - value) / (bad - good))


def band_score(value: float, floor: float, good_low: float, good_high: float, ceiling: float) -> float:
    """0.0 at/below `floor`, ramps up to 1.0 over [floor, good_low], holds
    at 1.0 over [good_low, good_high], ramps back down to 0.0 over
    [good_high, ceiling], 0.0 at/above `ceiling`. Use for metrics with a
    comfortable middle where too little AND too much are both worse
    (speaking rate, blink rate). Requires floor < good_low <= good_high < ceiling.
    """
    if not (floor < good_low <= good_high < ceiling):
        raise ValueError(
            f"expected floor < good_low <= good_high < ceiling, got "
            f"{floor}, {good_low}, {good_high}, {ceiling}"
        )
    if value <= floor or value >= ceiling:
        return 0.0
    if value < good_low:
        return _clamp01((value - floor) / (good_low - floor))
    if value <= good_high:
        return 1.0
    return _clamp01((ceiling - value) / (ceiling - good_high))


def identity01(value: float) -> float:
    """For metrics already expressed as a 0-1 "higher is better" fraction
    (camera_facing) -- clamps defensively against out-of-range input."""
    return _clamp01(value)


@dataclass
class FusionInputs:
    """The latest known value of each raw metric, or None if it hasn't
    been observed yet. Field names match the metric names AudioWorker /
    VideoWorker write to the event log (see analyzer/events.py), so
    compute_from_log() can build this directly from
    EventLog.recent_metrics() + dashboard_data.latest_value().

    hedge_rate exists for forward-compatibility with analyzer/text.py
    (Week 6, not built yet) -- it is simply None until that module writes
    a "hedge_rate" metric, at which point compute_from_log() picks it up
    with no changes needed here.
    """
    camera_facing: float | None = None      # 0-1 fraction
    pose_stability: float | None = None     # degrees, std-dev; lower = steadier
    blink_rate: float | None = None         # blinks per minute
    wpm: float | None = None                # words per minute
    pause_rate: float | None = None         # 0-1 fraction of time paused
    filler_rate: float | None = None        # fillers per 100 words
    hedge_rate: float | None = None         # hedge phrases per 100 words (not produced yet)


# Every FusionInputs field name that has a normalisation rule below. Used
# to sanity-check the config weight dicts (see tests/test_fusion.py) so a
# typo'd key in FUSION.*_weights fails loudly instead of silently
# contributing nothing.
NORMALISABLE_METRICS = {
    "camera_facing", "pose_stability", "blink_rate", "wpm", "pause_rate", "filler_rate", "hedge_rate",
}


def _normalised(inputs: FusionInputs) -> dict[str, float | None]:
    """Runs every present raw metric through its 0-1 "how good is this"
    transform. A metric that's None in `inputs` stays None here -- never
    guessed (see this module's docstring, rule 1)."""
    F = FUSION
    return {
        "camera_facing": identity01(inputs.camera_facing) if inputs.camera_facing is not None else None,
        "pose_stability": (
            linear_lower_is_better(inputs.pose_stability, F.pose_stability_good, F.pose_stability_bad)
            if inputs.pose_stability is not None else None
        ),
        "blink_rate": (
            band_score(inputs.blink_rate, F.blink_floor, F.blink_good_low, F.blink_good_high, F.blink_ceiling)
            if inputs.blink_rate is not None else None
        ),
        "wpm": (
            band_score(inputs.wpm, F.wpm_floor, F.wpm_good_low, F.wpm_good_high, F.wpm_ceiling)
            if inputs.wpm is not None else None
        ),
        "pause_rate": (
            linear_lower_is_better(inputs.pause_rate, F.pause_rate_good, F.pause_rate_bad)
            if inputs.pause_rate is not None else None
        ),
        "filler_rate": (
            linear_lower_is_better(inputs.filler_rate, F.filler_rate_good, F.filler_rate_bad)
            if inputs.filler_rate is not None else None
        ),
        "hedge_rate": (
            linear_lower_is_better(inputs.hedge_rate, F.hedge_rate_good, F.hedge_rate_bad)
            if inputs.hedge_rate is not None else None
        ),
    }


def weighted_average(weighted_scores: dict[str, tuple[float, float | None]]) -> tuple[float | None, dict[str, float]]:
    """`weighted_scores`: {metric_name: (weight, normalised_score_or_None)}.

    Returns (weighted average over just the present entries, {name:
    normalised_score} for just those present entries). Returns (None, {})
    if nothing was present, or if the present entries' weights sum to zero
    or less -- see this module's docstring, rule 2.
    """
    present = {name: (w, s) for name, (w, s) in weighted_scores.items() if s is not None}
    if not present:
        return None, {}
    total_weight = sum(w for w, _ in present.values())
    if total_weight <= 0:
        return None, {}
    avg = sum(w * s for w, s in present.values()) / total_weight
    drivers = {name: s for name, (_, s) in present.items()}
    return _clamp01(avg), drivers


@dataclass
class SubScore:
    """One sub-score: 0-100, or None if nothing fed it. `drivers` maps
    each metric that *was* available to its normalised (0-1) contribution
    -- this is what makes a score explainable rather than a black box.

    `coverage` is the fraction (0-1) of this score's *configured* weight
    that actually had data behind it. This exists because of a real,
    important failure mode found by testing: a score built from half its
    normal inputs (e.g. only camera_facing and blink_rate, with pause_rate
    and hedge_rate both missing) renders as a plain number -- 67 -- that
    looks exactly as confident as a fully-informed score. It isn't. A
    caller (the dashboard) should always show `coverage` next to `value`,
    not just when it happens to be low, so a thin score can never be
    mistaken for a complete one.
    """
    value: float | None
    drivers: dict[str, float]
    coverage: float  # 0-1; 0.0 when value is None (nothing was present at all)


def _sub_score(normalised: dict[str, float | None], weights: dict[str, float]) -> SubScore:
    weighted = {name: (weight, normalised.get(name)) for name, weight in weights.items()}
    avg01, drivers = weighted_average(weighted)
    value = round(avg01 * 100, 1) if avg01 is not None else None

    total_configured_weight = sum(weights.values())
    present_weight = sum(weight for name, (weight, score) in weighted.items() if score is not None)
    coverage = (present_weight / total_configured_weight) if total_configured_weight > 0 else 0.0

    return SubScore(value=value, drivers=drivers, coverage=coverage)


@dataclass
class FusionResult:
    engagement: SubScore
    fluency: SubScore
    confidence: SubScore


def compute_scores(inputs: FusionInputs) -> FusionResult:
    """The core, pure scoring function -- see this module's docstring for
    the two rules it enforces. Deliberately has no I/O so it's trivial and
    fast to test; compute_from_log() is the thin layer that feeds it real
    data.
    """
    norm = _normalised(inputs)
    return FusionResult(
        engagement=_sub_score(norm, FUSION.engagement_weights),
        fluency=_sub_score(norm, FUSION.fluency_weights),
        confidence=_sub_score(norm, FUSION.confidence_weights),
    )


def compute_from_log(log) -> FusionResult:
    """Builds FusionInputs from an EventLog's most recent readings and
    scores them. `log` needs one method: recent_metrics(name) -> [(t,
    value), ...] (matches analyzer.events.EventLog -- inject a fake, or a
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
    return compute_scores(inputs)


def write_scores(log, result: FusionResult, t: float) -> None:
    """Persists a FusionResult back to the event log (source="fusion") so
    the dashboard and the post-session report can read it like any other
    metric. Sub-scores that are None (nothing fed them) are simply not
    written -- never written as a fabricated placeholder value.
    """
    for name, sub in (
        ("engagement_score", result.engagement),
        ("fluency_score", result.fluency),
        ("confidence_score", result.confidence),
    ):
        if sub.value is not None:
            log.metric("fusion", name, sub.value, t=t)
