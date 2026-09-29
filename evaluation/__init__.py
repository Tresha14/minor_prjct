"""Week 7 evaluation tooling (synopsis Sec 8): rater-agreement statistics,
system-vs-human correlation, and leave-one-signal-out ablation for the
fusion sub-scores.

Everything in this package is real, tested code that operates on whatever
data you give it -- but no real evaluation data has been collected yet
(that requires actually recording interviews and getting them rated by
people, which cannot be rushed or faked). See evaluation/README.md for
the full methodology and current status. Tests in tests/test_evaluation_*
use only synthetic, clearly-labelled example data to prove the statistics
themselves are computed correctly -- never treat those numbers as project
results.
"""
