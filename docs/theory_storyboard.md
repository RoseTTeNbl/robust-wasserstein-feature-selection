# Structure of the bilingual theory note

Revised on 10 October 2026. The note develops one argument from joint feature selection and test design to a globally valid search. It contains no literature discussion or bibliography. The two languages share the same mathematical structure and numbered formulas.

## Main argument

1. **Selection objective.** Introduce the pooled observations, class labels and classwise weights; define the uncertainty sets, the selected coordinates, the score convention and a general loss. Explain the optimization target before its numerical instances.
2. **Finite test design.** Use the two-index distance vector and finite scores, source allowances and class prices. State exactness and the whole-space upper-bound guarantee; explain the convex-interpolation idea without specialized loss inverses.
3. **Risk-dependent coordinate prices.** Derive the price contribution bound from nonnegative source allowances. Write the complete continuous model explicitly, explain every line and establish binary value equivalence.
4. **Lower bounds from transported masses.** Explain the target loss before introducing its function. Keep only the transport matrix and correction vector in the final certificate. Derive the fractional lower bound directly from the original objective, then distinguish this inequality from the exact dual representation.
5. **Adaptive global search.** Retain full dual fractions and feasible scores. Recenter old fractions without an oracle call and reprice old scores to obtain genuine feasible-test upper bounds at new subsets. Use possible improvement and interval width for acquisition, with a global master guard and a required information threshold. State the history-width, guarded-gap, packing and directly computable feature-tail evaluation bounds with explicit oracle/master accuracy conditions.
6. **Instances and scope.** Only here introduce half hinge, logistic, exponential and squared hinge. Separate general mathematical coverage from the two implemented losses and from population calibration.

## Appendices

- **A → Theorem 1:** three connected parts: finite transports and their LP dual give the lower value inequality; whole-space interpolation and source-marginal integration give the reverse inequality; compact primal sublevel sets give attainment. The selected distance/projection/lifting geometry is merged, and test construction is immediately followed by risk integration. All endpoint, convexity, zero-distance and attainment steps remain explicit.
- **B → Theorem 2:** specify every multiplier and its domain; expand the Lagrangian through explicit regrouping identities; minimize separately over free and nonnegative variables; derive the target-loss objective; eliminate allocation multipliers in both directions. Then establish strong duality, exclude zero scale at an optimum, check every rescaled constraint and objective, verify binary equivalence, and prove convexity by the lower functions' Hessians.
- **C → Theorem 3:** prove repriced-test feasibility and the source-weighted correction cap; bound one-sided changes in transferred upper bounds and retained cuts; derive the width estimate with coefficient B; prove that a current-model master guard meets the information threshold; obtain strict query separation, packing complexity and the explicit sorted-feature-tail count. A further power-law rate is conditional on a separately justified packing estimate. Accurate binary evaluation stages, internal refinements and wall time are distinguished.
- **D → Section 6:** define the scalar loss and minimizing score before use; derive clipping, derivatives, substitution and square completion explicitly, with the boundary score limits and nonnegativity bounds shown.

The appendices retain the same four topics after the length reduction, but include the intermediate steps needed to follow each argument. Symbols and domains precede their use, and equations show both directions of claimed equivalences. Fixed-bound comparisons, alternative master formulations, long examples and a separate continuity proof remain omitted; earlier versions remain archived.

## Notation and maintenance

- All observations share one index. Class indicator and signed label are data; target labels do not restrict where either class can move mass.
- The coordinate-distance vector has only source and target indices. Class movement is a vector; no four-index distance tensor is displayed.
- The main text uses V for original binary risk and v for its convex extension.
- Source allowances and scores are free primal variables. Their nonnegativity or finite bounds follow from constraints; do not change their dual coefficient conditions by adding unstated domains.
- The final main-text lower certificate has two unknown objects. Allocation multipliers and normalization variables appear only where they are derived in the appendix.
- Number main equations consecutively; appendices use A.1, B.1, etc. Use only labels that are needed for the argument.
- Keep both TeX sources and the English Markdown reading copy synchronized, and follow the reviewed paired build in AGENTS.md. Preserve prior versions.
- The 10 October revision changes the search implementation and its quantitative guarantee. The original 1,640 study records and their empirical report remain unchanged; new matched computational validation has a separate method version and directory. The whole-space construction is sufficient, not unique; the half-hinge/logistic implementations satisfy the same loss bounds.
