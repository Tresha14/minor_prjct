# Week 7: Evaluation

**Status: tooling is built and tested. No real evaluation has been run yet.**
Everything below the "Methodology" section that looks like a number is either
a worked example with synthetic data (clearly labelled) or a template with no
rows filled in. Validating the system against real human judgement requires
actually recording interviews and getting people to rate them — that can't be
rushed, approximated, or generated to look more finished before that happens.
If you're reading this expecting rater-agreement/correlation/ablation
*results*, they don't exist yet; what exists is the code to produce them once
real ratings are collected.

## What's built (`evaluation/`)

- `statistics.py` — `pearson_correlation`, `spearman_correlation`,
  `cohens_kappa`. Pure functions, each returns `None` (never a fabricated
  0.0 or 1.0) when the input can't actually support the statistic — see each
  function's docstring for the specific case.
- `ablation.py` — leave-one-signal-out ablation for `analyzer/fusion.py`'s
  sub-scores. Reuses `fusion.weighted_average()` directly so an ablation
  result can never drift out of sync with the real scoring math.
- `validation.py` — `rater_agreement()` (wraps Cohen's kappa with a
  documented reliability threshold) and `correlate_with_human_ratings()`
  (wraps Pearson/Spearman, and explicitly flags — not hides — when the
  sample size is too small to trust).
- `ratings_template.csv` — the exact shape real ratings should be entered in
  once collected (see "How to actually run this" below).
- `tests/test_evaluation_*.py` — hand-calculated/textbook golden-value
  checks proving the statistics themselves are computed correctly. These use
  synthetic example numbers only; they are not, and must never be read as,
  project results.

## Methodology

1. **Collect sessions.** Record N mock interviews through the actual live
   pipeline (`app.py`) so the event log/fusion scores being evaluated are the
   real ones a candidate would see, not a hand-crafted example.
2. **Rate them independently.** At least two raters, each blind to the
   system's own scores, rate each session on engagement / fluency /
   confidence — suggested a 1–5 ordinal scale to match `cohens_kappa`'s
   categorical assumption, though `pearson_correlation`/`spearman_correlation`
   work with a continuous scale too if that's preferred instead.
3. **Check rater agreement first**, via `validation.rater_agreement()`. If
   agreement is weak (kappa below the documented threshold, default 0.6),
   the ratings themselves aren't trustworthy yet — fix the rubric or rater
   training before step 4, rather than correlating the system against noisy
   ground truth and drawing conclusions from it.
4. **Average each session's (trusted) ratings** and correlate against the
   corresponding fusion sub-score with `validation.correlate_with_human_ratings()`.
5. **Run the ablation** (`ablation.ablate_across_sessions()`) per sub-score to
   see which configured metric actually moves the score the most across the
   collected sessions — useful both as a report finding and as a sanity
   check on `analyzer/config.py`'s `FUSION.*_weights` (a metric with real
   data but near-zero measured impact may be mis-weighted).
6. **Recalibrate `FUSION`'s reference ranges only if the data supports it**
   (synopsis Sec 8) — this is optional and should not happen from a single
   small batch of sessions.

## How to actually run this once data exists

1. Fill in `ratings_template.csv` (or a copy of it) with real session
   ratings — one row per (session, rater) pair.
2. Load it (plain `csv`/`pandas`, whichever the rest of the project ends up
   using) and group by `session_id` to get each rater's list of ratings in
   session order for `rater_agreement()`.
3. Average each session's per-rater ratings (once agreement is confirmed
   acceptable) and pull the matching `engagement_score` / `fluency_score` /
   `confidence_score` for that session from its `events.db` (see
   `analyzer.events.EventLog.recent_metrics`) for `correlate_with_human_ratings()`.
4. For ablation, pull each session's `FusionResult.<subscore>.drivers` (see
   `analyzer.fusion.compute_from_log`) into a list and pass it to
   `ablation.ablate_across_sessions()` alongside the relevant
   `FUSION.*_weights` dict.

No loader script for step 2/3 exists yet — deliberately left out tonight
since there's no real CSV to load and building one against imagined data
risks baking in wrong assumptions about the real data's shape. Write it
once the first real ratings come in, against `ratings_template.csv`'s
actual columns.
