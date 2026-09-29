"""Ties evaluation/statistics.py together into the two concrete checks
Week 7 needs (synopsis Sec 8):

  1. Are independent human raters consistent enough with each other to
     trust an averaged rating at all? (rater_agreement)
  2. Do the fusion sub-scores actually correlate with (averaged, trusted)
     human ratings? (correlate_with_human_ratings)

Both are thin, honest wrappers -- they do no data collection themselves.
Real numbers only exist once real sessions have been recorded and rated;
see evaluation/README.md for the collection plan and
evaluation/ratings_template.csv for the exact shape real data should be
entered in.
"""
from __future__ import annotations

from dataclasses import dataclass

from evaluation.statistics import cohens_kappa, pearson_correlation, spearman_correlation


@dataclass
class AgreementResult:
    kappa: float | None
    n_items: int
    reliable: bool | None  # None if kappa is None (undefined); otherwise kappa >= threshold


def rater_agreement(ratings_a: list, ratings_b: list, threshold: float = 0.6) -> AgreementResult:
    """`threshold` follows the common Landis & Koch rule-of-thumb cutoff
    for "substantial agreement" (kappa >= 0.6) -- reasonable as a
    documented starting point for a minor project, not a claim this
    project has independently validated that specific cutoff.

    `reliable` being False (or the human-rating correlation step being run
    at all) is itself a real, useful finding -- it means the rating
    process needs fixing (clearer rubric, rater training) before trusting
    any correlation computed from those ratings, not a project failure to
    hide.
    """
    kappa = cohens_kappa(ratings_a, ratings_b)
    reliable = None if kappa is None else (kappa >= threshold)
    return AgreementResult(kappa=kappa, n_items=len(ratings_a), reliable=reliable)


@dataclass
class CorrelationResult:
    pearson_r: float | None
    spearman_rho: float | None
    n: int
    note: str | None = None  # set when n is too small to trust the numbers above, even if they computed


_MIN_N_FOR_A_TRUSTWORTHY_CORRELATION = 10  # engineering judgement, not a statistical law -- documented, not hidden


def correlate_with_human_ratings(system_scores: list[float], human_ratings: list[float]) -> CorrelationResult:
    """`system_scores` and `human_ratings` must be the same length and
    paired by session (system_scores[i] and human_ratings[i] describe the
    same session). Computes both Pearson and Spearman so a reader can see
    whether a relationship, if any, looks linear or just monotonic.

    A correlation coefficient CAN be computed from very few sessions and
    will still look like a real, precise number -- exactly the kind of
    misleadingly-confident-looking-number analyzer/fusion.py's `coverage`
    field exists to prevent for fusion scores. `note` plays the same role
    here: it's set (not the numbers suppressed) whenever n is below
    _MIN_N_FOR_A_TRUSTWORTHY_CORRELATION, so a caller sees both the number
    and an explicit warning not to over-read it, rather than a bare
    number that looks equally solid at n=3 and n=300.
    """
    if len(system_scores) != len(human_ratings):
        raise ValueError(
            f"system_scores and human_ratings must be the same length, "
            f"got {len(system_scores)} and {len(human_ratings)}"
        )
    n = len(system_scores)
    r = pearson_correlation(system_scores, human_ratings)
    rho = spearman_correlation(system_scores, human_ratings)
    note = None
    if n < _MIN_N_FOR_A_TRUSTWORTHY_CORRELATION:
        note = (
            f"n={n} is below the {_MIN_N_FOR_A_TRUSTWORTHY_CORRELATION}-session minimum this "
            f"project treats as a trustworthy sample -- treat any correlation value above as "
            f"preliminary, not a validated result."
        )
    return CorrelationResult(pearson_r=r, spearman_rho=rho, n=n, note=note)
