# Matched validation of history-guided search

This is a new algorithm validation, separate from the frozen historical study. It measures robust optimization accuracy and computational effort; no testing-power or variable-recovery conclusion is inferred from it.

Independent exhaustive search supplies the reference interval. Methods share generated data, fixed radii, loss, initial support, numerical tolerance and wall-time limit. Both use the same current oracle, including its model reuse, so the comparison isolates the search policy.

## Primary cases

| Method | Certified stops | Median full calls | Median seconds |
|---|---:|---:|---:|
| master | 24/24 | 15.0 | 0.239 |
| adaptive | 24/24 | 13.0 | 0.240 |

Adaptive uses fewer/equal/more full calls in 19/3/2 matched cases. It is faster in 10 and slower in 14. The median paired call ratio is 0.871; the median runtime ratio is 1.089.

## Larger cases

| Method | Certified stops | Median full calls | Median seconds |
|---|---:|---:|---:|
| master | 2/2 | 28.5 | 0.909 |
| adaptive | 2/2 | 20.5 | 0.719 |

Adaptive uses fewer/equal/more full calls in 2/0/0 matched cases. It is faster in 1 and slower in 1. The median paired call ratio is 0.764; the median runtime ratio is 1.095.

The extra repricing and cut-densification work is included in runtime. A reduction in expensive oracle calls does not imply a runtime reduction. This seeded validation is too small to establish universal performance superiority.

`summary.json` retains every status, reference consistency check, attained gap, repricing/refinement count and model/workspace reuse count. Individual raw JSON files include the generated data, full traces and returned test certificates. The source snapshot and manifest identify the code used.
