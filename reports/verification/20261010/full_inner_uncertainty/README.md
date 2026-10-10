# Matched validation of full inner solves and uncertainty-guided selection

This is a new algorithm validation, separate from the frozen historical study. It measures robust optimization accuracy and computational effort; no testing-power or variable-recovery conclusion is inferred from it.

Independent exhaustive search supplies the reference interval. Methods share generated data, fixed radii, loss, initial support, numerical tolerance and wall-time limit. Both use the same current oracle, including its model reuse, so the comparison isolates the search policy.

The adaptive method fully solves the inner problem at every selected binary support. It then adds one tangent owned by that evaluation. A fixed Hamming-kernel Gaussian predictor trained on completed actual risk evaluations only ranks candidates by mean minus standard deviation: there is no transferred solution, candidate-centered tangent or fractional inner solve. All incumbent upper bounds come from actual inner evaluations.

## Primary cases

| Method | Certified stops | Median full calls | Median seconds |
|---|---:|---:|---:|
| master | 24/24 | 15.0 | 0.204 |
| adaptive | 24/24 | 15.0 | 0.184 |

Adaptive uses fewer/equal/more full calls in 4/12/8 matched cases. It is faster in 23 and slower in 1. The median paired call ratio is 1.000; the median runtime ratio is 0.786.

## Larger cases

| Method | Certified stops | Median full calls | Median seconds |
|---|---:|---:|---:|
| master | 2/2 | 28.5 | 0.697 |
| adaptive | 2/2 | 29.5 | 0.527 |

Adaptive uses fewer/equal/more full calls in 0/1/1 matched cases. It is faster in 2 and slower in 0. The median paired call ratio is 1.023; the median runtime ratio is 0.790.

Candidate screening, predictive means and model standard deviations, and global certification are included in runtime. A reduction in expensive oracle calls does not imply a runtime reduction. These are single wall-clock measurements for each matched case, without timing-repeat confidence intervals. This seeded validation is too small to establish universal performance superiority.

`summary.json` retains every status, reference consistency check, attained gap, repricing/refinement count, nonconstant-uncertainty and changed-order counts, and model/workspace reuse count. Individual raw JSON files include the generated data, full traces and returned test certificates. The source snapshot and manifest identify the code used.
