# Structure of the bilingual theory note

Revised on 9 October 2026. The note develops one argument from joint feature selection and test design to a globally valid search. It contains no literature discussion or bibliography. The two languages share the same mathematical structure and numbered formulas.

## Main argument

1. **Selection objective.** Introduce the pooled observations, class labels and classwise weights; define the uncertainty sets, the selected coordinates, the score convention and a general loss. Explain the optimization target before its numerical instances.
2. **Finite test design.** Use the two-index distance vector and finite scores, source allowances and class prices. State exactness and the whole-space upper-bound guarantee; explain the convex-interpolation idea without specialized loss inverses.
3. **Risk-dependent coordinate prices.** Derive the price contribution bound from nonnegative source allowances. Write the complete continuous model explicitly, explain every line and establish binary value equivalence.
4. **Lower bounds from transported masses.** Explain the target loss before introducing its function. Keep only the transport matrix and correction vector in the final certificate. Derive the fractional lower bound directly from the original objective, then distinguish this inequality from the exact dual representation.
5. **Global search.** Differentiate a fixed fractional bound, obtain its global affine cut, define the discrete master and track feasible-test upper bounds and global lower bounds. Explain touching and finite termination under exact solves.
6. **Instances and scope.** Only here introduce half hinge, logistic, exponential and squared hinge. Separate general mathematical coverage from the two implemented losses and from population calibration.

## Appendices

- **A:** projection/lifting; the finite adversary; an explicit bounded, measurable convex combination of endpoint scores; the complete two-way finite/full-space equality.
- **B:** existence, strict feasibility after removing fixed coordinates, and continuity and mass scaling of the minimum target loss, including boundary infima.
- **C:** the ordinary and risk-dependent Lagrangians; free-variable and nonnegative-variable infimum rules; the score-by-score derivation of the dual objective; exact elimination of allocation multipliers.
- **D:** binary equivalence, normalization, positive and zero scales, fraction convexity, valid cuts, numerical lower evaluations, direct risk bounds and exact termination.
- **E:** explicit fixed coordinate and shared bounds, their own dual derivations, and the comparison with the risk-dependent model.
- **F:** loss formulas, a scalar example, the optional complete-fraction master and a counterexample for squared transport cost.

## Notation and maintenance

- All observations share one index. Class indicator and signed label are data; target labels do not restrict where either class can move mass.
- The coordinate-distance vector has only source and target indices. Class movement is a vector; no four-index distance tensor is displayed.
- The main text uses V for original binary risk and v for its convex extension. Fixed-bound comparison functions remain in the appendix.
- Source allowances and scores are free primal variables. Their nonnegativity or finite bounds follow from constraints; do not change their dual coefficient conditions by adding unstated domains.
- The final main-text lower certificate has two unknown objects. Allocation multipliers and normalization variables appear only where they are derived in the appendix.
- Number main equations consecutively; appendices use A.1, B.1, etc. Use only labels that are needed for the argument.
- Keep both TeX sources and the English Markdown reading copy synchronized, and follow the reviewed paired build in AGENTS.md. Preserve prior versions.
- This revision generalizes the mathematical proof but leaves the original study records and core implementations unchanged. The explicit whole-space construction is sufficient, not unique; the implemented half-hinge/logistic extensions satisfy the same loss bounds.