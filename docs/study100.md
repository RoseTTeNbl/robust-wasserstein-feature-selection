# A small 100-dimensional study

This experiment tests the current outer-approximation implementation with the
standard hinge generating loss `ell(t) = max(0, 1+t)`, for which the zero detector
has robust risk 2. It does not reuse results from the earlier half-hinge code.
The theory remains general; this numerical implementation uses its linear-program
special case.

## Design and sources

The design combines two established experimental choices, without reproducing
either paper's complete experiment:

- Xie, Gao and Xie (2021), [Section 5.1](https://arxiv.org/pdf/2105.14348), use
  100-dimensional Gaussian mixtures, few training observations, 1,000 test
  observations per class and validation-selected Wasserstein radii. We retain
  the dimension, small-sample setting and independent large test sample. We use
  a sparse Gaussian mean difference so that selected coordinates have an explicit
  ground truth, and use unequal class sample sizes.
- Bertsimas, Pauphilet and Van Parys, [Section 3.1](https://arxiv.org/pdf/1710.01352),
  study Gaussian features, a known sparse support, prediction and support
  recovery. We use the same separation of optimization quality, predictive error
  and support recovery. Their label-generation model and algorithm are not
  claimed as reproduced baselines.

For a randomly located support `T` of size 5 in 100 dimensions, let `s` be its
indicator. Generate independent training and validation observations from

```
class 0: N(+0.6 s, I_100)
class 1: N(-0.6 s, I_100).
```

There are 20 training observations in class 0 and 25 in class 1, with 200
validation and 1,000 test observations per class. Every method selects exactly
5 coordinates. The support is selected from its own reproducible random stream
and is never supplied to a selector. All features are centered and scaled using
the pooled training observations only; validation and test data use that same
transformation.

The same trained model is evaluated on a second test set with the same class
means and noise standard deviation 1.25. Its Gaussian noise draws are paired
with those of the nominal test set. This is a prespecified variance-shift stress
test, not a claim that the population distribution lies in an empirical
Wasserstein ball. No model is retrained for this evaluation.

All choices are in `configs/study100.json`. Development seed 2026102500 is
separate from confirmation seeds 2026102501, 2026102502 and 2026102503.
The development results are not included in the confirmation summaries.
Confirmation validation draws are retained for reproducibility but do not alter
the frozen parameter choices.

## Parameter selection and comparisons

On the development sample only, select a common radius scale from
`{0.05, 0.15, 0.4}` by validation error, breaking ties by validation hinge loss
and then the smaller radius. Calibration uses the fixed training Fisher support.
For class `k`, the actual radius is `scale * sqrt(20 / n_k)` in standardized
coordinate units. This is a declared calibration rule, not a coverage theorem.
The scale is frozen before running any confirmation seed, and is identical
across all robust methods. Calibrating on the Fisher support is a method-dependent
choice; sharing the resulting radius ensures that every robust selector is
compared on the same optimization objective.

Choose the L1 logistic inverse penalty `C` from `{0.1, 1, 10}` on the same
development validation sample. For each candidate, rank training coefficients,
keep 5 coordinates, and evaluate an ordinary L2 logistic refit with fixed `C=1`.
Ties select the smaller L1 `C`. Both choices remain frozen for confirmation.
Class weights in logistic models give each class equal total training weight.

The methods are:

| Method | Coordinate selection | Detector |
| --- | --- | --- |
| `outer_approximation` | The current dual-cut method, initialized by Fisher selection | Current robust hinge detector |
| `fisher_robust` | Largest squared mean differences divided by within-class variance | Same robust hinge detector |
| `l1_robust` | Largest absolute L1 logistic coefficients | Same robust hinge detector |
| `random_robust` | Uniform random subset with an independent fixed seed | Same robust hinge detector |
| `l1_logistic` | Same L1-selected support as `l1_robust` | Ordinary L2 logistic refit |

L1 ranking returns exactly five coordinates, including zero-coefficient ties if
necessary; their number is recorded. It is not an exact cardinality-constrained
logistic optimizer. The robust baselines use the same fixed-support fitting
routine, radii, score extension and tolerances as the outer-approximation method.
The final comparison changes the detector while retaining the same support.
Positive scores mean class 0 throughout; the sklearn logistic decision score is
negated to obey this convention.

The outer method has a 120-second limit per confirmation sample, each inner fit
has a 30-second limit, and all solvers use one thread. The declared absolute
outer tolerance is 0.001. Initial Fisher selection is included in its reported
training time. Baseline selection time is included in each baseline's training
time even where a support is computed once for consistency.

## Measures and interpretation

- `error0` and `error1` are the two class-conditional hard-decision errors.
  `error_sum = error0 + error1`; balanced error is half this value. A zero score
  is assigned to class 0, so a constant zero score has error sum 1.
- `empirical_hinge` is the test average of `max(0,1-g)` in class 0 plus
  `max(0,1+g)` in class 1. It is a measured test quantity, not a robust certificate.
  It is not reported for ordinary logistic margins, whose scale is not the
  bounded hinge-score convention.
- `robust_upper` is the feasible training-problem risk bound. Only the outer
  method has a `global_lower` and `global_gap` for variable selection.
  Baseline inner-fit gaps are not global selection gaps.
- `signal_hits` counts selected coordinates in the known generating support.
  Solver status, time, fit count and master count are retained, including
  timeouts and failed runs. An unfinished run is not reported as optimal.

Three confirmation samples make this an implementation-scale comparison, not
evidence of statistical superiority. A lower training robust objective need
not produce lower held-out classification error. The chosen sparse mean-shift
model is favorable to linear classification; the stress evaluation does not
change that fact.

## Recorded results

The frozen run in `outputs/study100-20261025` selected radius scale 0.4 and
L1 logistic `C=10` using the development sample. Its class radii are 0.4 and
0.3577708764. The following are means over all three confirmation seeds;
BER is `(error0 + error1) / 2`.

| Method | Training robust upper | Test BER | Stress BER | True coordinates / 5 | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Outer approximation | 0.569241 | 37.85% | 42.77% | 3.00 | 120.038 |
| Fisher + robust | 0.605873 | 32.30% | 38.48% | 4.33 | 0.016 |
| L1 + robust | 0.614520 | 37.20% | 41.13% | 3.00 | 0.017 |
| Random + robust | 0.744730 | 49.97% | 50.38% | 0.00 | 0.017 |
| L1 + logistic | N/A | 20.58% | 25.23% | 3.00 | 0.004 |

The outer method reduced the training objective relative to its Fisher
initialization in every seed, but had worse average test error and support
recovery than Fisher selection. The ordinary logistic refit on the L1 support
performed better than the robust detector on that same support. These results
do not support a predictive advantage for the proposed method in this design.
The robust predictions also showed a marked class imbalance: mean class-0 /
class-1 test errors were 68.83% / 6.87% for outer approximation, 59.43% / 5.17%
for Fisher + robust, and 66.43% / 7.97% for L1 + robust. The corresponding
L1 + logistic errors were 22.33% / 18.83%. The run measures this behavior but
does not establish its cause.

Every outer run reached the time limit; none established global optimality.
Here `U` is the returned feasible risk, `L` is the global master lower bound,
and the gap is `U-L`.

| Seed | U | L | Gap | Status | Fits / masters | Seconds |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| 2026102501 | 0.615762 | 0 | 0.615762 | `time_limit` | 718 / 717 | 120.026 |
| 2026102502 | 0.580804 | 0 | 0.580804 | `time_limit` | 701 / 700 | 120.041 |
| 2026102503 | 0.511156 | 0 | 0.511156 | `time_limit` | 760 / 760 | 120.047 |

All 15 method runs returned predictions. An independent finite transport LP
verified the fitted lower/upper intervals for all 12 returned robust supports,
including the three outer incumbents, at tolerance `1e-7`. This verifies the
fixed-support values only. Recomputing 60,000 saved scores and their metrics
reproduced the individual results and summary; every source hash still matched
the frozen protocol. Details are in `outputs/study100-20261025/verification.json`.
The compact versioned record `docs/study100_results.json` retains the protocol
identity, source hashes, frozen parameters, all 15 rows and verification summary.

## Reproduction and files

From the repository root:

```powershell
.venv-research/Scripts/python.exe -m robusthw.experiments.study100 --output outputs/study100-20261025
```

The program writes its configuration, source hashes and environment to
`protocol.json` before calibration or confirmation. `calibration.json` stores
every candidate and the frozen choices. Each `seed-*` directory contains the
standardized data and transformation, per-method fits and predictions, and
`result.json`. `results.csv` retains individual results and `summary.json` gives
simple averages over available predictions, with the number of evaluated runs
and every status reported. Valid predictions from timed-out optimization runs
remain in the averages; exceptions have no prediction to average and stay in
the status list and individual records.

Rerunning the identical command resumes completed seeds. A code or configuration
change requires a new output directory, preserving previous results. Interrupted
seeds are rerun; completed seed results are reused only under the same source
and configuration hashes. Nothing under the historical study or report folders
is overwritten.

Code responsibilities are limited to:

- `core/`: mathematical fitting, certificates and selection (no data generation).
- `experiments/study100_baselines.py`: transparent comparison selectors and
  ordinary logistic fitting (no robust optimization).
- `experiments/study100.py`: sampling, train-only transformation, development
  calibration, evaluation and recording (no LP or cut algebra).
