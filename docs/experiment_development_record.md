# Complete development and implementation record

Updated 11 October 2026. This record preserves comparisons omitted from the main experimental narrative. It does not add experiments or revise any result. The report instead leads with testing, recovery, prediction and computation questions. The earlier statistical study, the four development stages, the repaired integration replay and the independent confirmation remain separate evidence cohorts.

## Four development stages

Every stage uses the same 26 paired development configurations, but each has its own immutable source snapshots and timing batch. Their times are not pooled to construct a benchmark. Counts include unsuccessful partial work. A dash means the measurement was not recorded; it does not mean zero.

| Stage | Search | Evaluated subsets | Continuous solves | Masters | Total seconds | Optimal runs |
|---|---|---:|---:|---:|---:|---:|
| 1 | Mean minus SD | 475 | — | 209 | 5.188 | 26/26 |
| 1 | Expected improvement | 475 | — | 206 | 4.771 | 26/26 |
| 1 | Mean minus SD, master priority | 477 | — | 218 | 4.859 | 26/26 |
| 1 | Improvement, master priority | 475 | — | 217 | 4.870 | 26/26 |
| 1 | Tangent master at every step | 467 | — | 467 | 6.056 | 26/26 |
| 2 | Original GP search | 475 | 950 | 209 | 5.072 | 26/26 |
| 2 | Ranked proposals | 480 | 960 | 212 | 4.884 | 26/26 |
| 2 | Dual-face tangent | 467 | 1419 | 206 | 17.270 | 26/26 |
| 2 | Tangent master at every step | 467 | 934 | 467 | 5.817 | 26/26 |
| 3 | Original GP search | 475 | 950 | 209 | 4.987 | 26/26 |
| 3 | Fractions at every step | 382 | 764 | 382 | 7.722 | 26/26 |
| 3 | Tangent master at every step | 467 | 934 | 467 | 5.840 | 26/26 |
| 4 | Original GP search | 475 | 950 | 209 | 5.132 | 26/26 |
| 4 | Stored fractions | 398 | 796 | 168 | 4.985 | 25/26 |
| 4 | Fractions at every step | 382 | 764 | 382 | 7.837 | 26/26 |
| 4 | Tangent master at every step | 467 | 934 | 467 | 5.988 | 26/26 |

Stage 1 changes the prediction score and master priority. Stage 2 changes candidate construction or selects a different dual tangent on the same evaluated subset; the latter costs an additional continuous solve. Stage 3 stores dual fractions and solves their master at every iteration. Stage 4 keeps the fractions while retaining the original periodic refresh.

The stage-4 stored-fraction variant has one solver-reported infeasible master and only 25/26 optimal results. Its partial cost is not an all-success improvement claim. A feasible incumbent satisfied the stored constraints. The integrated implementation verifies such a witness and, within the remaining time allowance, retries the same master without presolve. The repair is shared by all compared methods.

## Repaired integration replay

These are the same 26 development configurations, with one run per method. They test the fixed implementation after repair; they are not independent confirmation data.

| Search | Evaluated subsets | Continuous solves | Masters | Master seconds | Total seconds | Optimal runs |
|---|---:|---:|---:|---:|---:|---:|
| Tangent master at every step | 467 | 934 | 467 | 3.948 | 7.558 | 26/26 |
| Original GP search | 475 | 950 | 209 | 2.205 | 5.822 | 26/26 |
| Stored fractions | 398 | 796 | 168 | 2.944 | 6.135 | 26/26 |

## Earlier complete-fraction comparison

This earlier 60-record supplement uses different seeds from the primary optimization study. It is not a timing comparison of the current implementation. Its time difference is paired fraction minus tangent seconds, with a 95% training-seed bootstrap interval. A positive difference means the fraction version is slower. Limits remain in the denominator.

| Mechanism | Dimension | Tangent optimal | Fraction optimal | Paired time difference [95% interval], seconds |
|---|---:|---:|---:|---:|
| mean | 8 | 5/5 | 5/5 | 0.19 [0.12, 0.25] |
| mean | 12 | 5/5 | 3/5 | 9.45 [7.38, 11.39] |
| mean | 20 | 0/5 | 0/5 | 0.00 [-0.00, 0.00] |
| mixture | 8 | 5/5 | 5/5 | 0.31 [0.22, 0.40] |
| mixture | 12 | 5/5 | 4/5 | 10.57 [7.36, 13.79] |
| mixture | 20 | 0/5 | 0/5 | 0.00 [0.00, 0.00] |

## Complete earlier outcome inventory

Optimality refers to the robust objective at its numerical tolerance. Other outcomes include completed baselines and diagnostic supports; they do not mean a robust global optimum. All original and retry work stays charged.

| Suite | Records | Optimal | Time limit | Failure | Other |
|---|---:|---:|---:|---:|---:|
| statistical | 960 | 114 | 6 | 0 | 840 |
| computational | 320 | 199 | 120 | 1 | 0 |
| real | 240 | 0 | 40 | 0 | 200 |
| sensitivity | 60 | 58 | 0 | 2 | 0 |
| ratio | 60 | 37 | 23 | 0 | 0 |

The original 12 archived failures were retried only within unused original allowance: 5 ended optimal, 6 time-limited and 1 failed. Later sensitivity failures are not part of that original replay cohort. Final records contain 3 failures.

The finite source-atom translation check has 1260 recorded evaluations, 0 violations above 1e-7, and maximum recorded nonnegative violation 0. These have an explicitly feasible empirical coupling; they do not establish coverage of the unknown population.

Charged selection and end-to-end sums are 2.94303387 and 2.95917699 hours, including unsuccessful work. Sums across concurrent runs are not wall-calendar time.

## Parameters, sources and reproducibility paths

- Statistical, real-data, sensitivity, optimization and earlier fraction configurations: `configs/statistical.json`, `real.json`, `sensitivity.json`, `computational.json`, `ratio_cuts.json`. The current study does not sweep statistical sample size, dimension or budget as in the source paper.
- Baseline fidelity and adaptations: `docs/baseline_reproduction.md`. Sparse additive MMD implements only the paper's linear criterion; its variance coefficient is fixed at 0.1, not selected by the original cross-validation protocol. The inspected source revision uses a different variance sign and label-dependent test-bandwidth fitting. Neither is silently inherited.
- Full earlier records, every metric and support: `results/research_20261003/per_trial.csv`, `summary.json`, and suite `records/*.json`. Original failure provenance is `remediation/initial_failures.json` with individual original/retry records.
- Development phases: `reports/verification/20261010/next_selection/development/phase1` through `phase4`; source/raw fingerprints are in `development/evidence_manifest.json`.
- Repaired integration: `reports/verification/20261010/next_selection/current/development`. Its `protocol.json`, raw records, summary, source copies and `seal.json` specify the one-run evidence.
- Independent confirmation: `reports/verification/20261010/next_selection/current/confirmation`. Its 16 configurations reuse four generated datasets and include three rotated-order timings per method; the report sums within-configuration medians. All 144 outcomes pass. These are small cases, not a large-scale or statistical-utility confirmation.
- The first instrumentation-keyword failure is retained under `current_instrumentation_failure`. It produced no usable logistic comparison and is not counted as an algorithm failure benchmark.
- Environment versions for the current validation are retained in `reports/verification/20261010/next_selection/current_environment.json`. Immutable source fingerprints identify the exact implementation, in addition to Git history.

The statistical study uses selection-fitted standardization; the current optimization confirmation uses generated coordinates directly. The current rule keeps the original Gaussian-process prediction, at most eight proposals and a three-evaluation master refresh. Its change is retaining the safe fraction already produced by the same complete inner solve. The confirmation set was not used to retune this rule.
