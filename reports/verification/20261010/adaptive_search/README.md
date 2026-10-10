# Matched validation of history-guided search

This is a new algorithm validation, separate from the frozen historical study. It measures robust optimization accuracy and computational effort; no testing-power or variable-recovery conclusion is inferred from it.

Independent exhaustive search supplies the reference interval. Methods share generated data, fixed radii, loss, initial support, numerical tolerance and wall-time limit. Both use the same current oracle, including its model reuse, so the comparison isolates the search policy.

## Primary cases

| Method | Certified stops | Median full calls | Median seconds |
|---|---:|---:|---:|
| master | 24/24 | 15.0 | 0.207 |
| adaptive | 24/24 | 12.5 | 0.240 |

Adaptive uses fewer/equal/more full calls in 20/2/2 matched cases. It is faster in 10 and slower in 14. The median paired call ratio is 0.879; the median runtime ratio is 1.105.

## Larger cases

| Method | Certified stops | Median full calls | Median seconds |
|---|---:|---:|---:|
| master | 2/2 | 28.5 | 0.710 |
| adaptive | 2/2 | 21.0 | 0.589 |

Adaptive uses fewer/equal/more full calls in 2/0/0 matched cases. It is faster in 1 and slower in 1. The median paired call ratio is 0.802; the median runtime ratio is 1.057.

The extra repricing and cut-densification work is included in runtime. A reduction in expensive oracle calls does not imply a runtime reduction. These are single wall-clock measurements for each matched case, without timing-repeat confidence intervals. This seeded validation is too small to establish universal performance superiority.

`summary.json` retains every status, reference consistency check, attained gap, repricing/refinement count and model/workspace reuse count. Individual raw JSON files include the generated data, full traces and returned test certificates. The source snapshot and manifest identify the code used.
