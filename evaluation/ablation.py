"""Leave-one-signal-out ablation for the fusion sub-scores (Week 7,
synopsis Sec 8): quantifies how much each individual metric actually
contributes to a sub-score.

Deliberately reuses analyzer.fusion.weighted_average() directly rather
than reimplementing the weighted-average math here, so an ablation result
can never silently drift out of sync with how the real score is computed
-- and deliberately operates on already-normalised driver scores (a
FusionResult sub-score's `.drivers`, batched from one or more real/logged
sessions) rather than raw metrics, so this module needs zero changes to
analyzer/fusion.py and re-derives no normalisation logic that already
exists and is already tested there.
"""
from __future__ import annotations

from dataclasses import dataclass

from analyzer.fusion import weighted_average


@dataclass
class AblationResult:
    metric: str
    baseline_score: float | None        # sub-score with every signal this session had present
    without_metric_score: float | None  # the same sub-score with `metric` removed
    impact: float | None                # baseline - without_metric; None if either side is None


def ablate_one_metric(drivers: dict[str, float], weights: dict[str, float], metric: str) -> AblationResult:
    """`drivers`: {metric_name: normalised_0_1_score} for ONE session's
    sub-score (e.g. a FusionResult.engagement.drivers). `weights`: the
    same weight dict fusion.py used to produce it (e.g.
    FUSION.engagement_weights). `metric` must be a key actually present in
    `drivers` -- ablating a metric that was never observed this session
    would just reproduce the baseline unchanged and silently look like
    "this metric doesn't matter", which would be a fabricated conclusion
    about a signal that was never even tested; this function raises
    instead of allowing that to happen quietly.
    """
    if metric not in drivers:
        raise ValueError(
            f"cannot ablate {metric!r}: it wasn't present in this session's drivers "
            f"({sorted(drivers)}), so removing it wouldn't measure anything real"
        )
    baseline, _ = weighted_average({name: (weights.get(name, 0.0), score) for name, score in drivers.items()})
    reduced = {name: score for name, score in drivers.items() if name != metric}
    without, _ = weighted_average({name: (weights.get(name, 0.0), score) for name, score in reduced.items()})
    impact = (baseline - without) if (baseline is not None and without is not None) else None
    return AblationResult(metric=metric, baseline_score=baseline, without_metric_score=without, impact=impact)


def ablate_across_sessions(sessions_drivers: list[dict[str, float]], weights: dict[str, float],
                            metric: str) -> list[AblationResult]:
    """Runs ablate_one_metric() over multiple sessions' driver dicts,
    skipping (never fabricating a zero-impact result for) any session
    where `metric` wasn't actually present."""
    return [
        ablate_one_metric(drivers, weights, metric)
        for drivers in sessions_drivers
        if metric in drivers
    ]


def average_impact(results: list[AblationResult]) -> float | None:
    """Mean impact across sessions, or None if there's nothing to average
    (no sessions had this metric present, or every computed impact was
    itself None)."""
    impacts = [r.impact for r in results if r.impact is not None]
    if not impacts:
        return None
    return sum(impacts) / len(impacts)
