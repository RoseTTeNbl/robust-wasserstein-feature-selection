# Structure of the bilingual theory note

Revised on 10 October 2026. The note develops one argument from joint feature selection and test design to a globally valid search. It contains no literature discussion or bibliography. The two languages share the same mathematical structure and numbered formulas.

## Main argument

1. **Selection objective.** Introduce the pooled observations, class labels and classwise weights; define the uncertainty sets, the selected coordinates, the score convention and a general loss. Explain the optimization target before its numerical instances.
2. **Finite test design.** Use the two-index distance vector and finite scores, source allowances and class prices. State exactness and the whole-space upper-bound guarantee; explain the convex-interpolation idea without specialized loss inverses.
3. **Risk-dependent coordinate prices.** Derive the price contribution bound from nonnegative source allowances. Write the complete continuous model explicitly, explain every line and establish binary value equivalence.
4. **Lower bounds from transported masses.** Explain the target loss before introducing its function. Keep only the transport matrix and correction vector in the final certificate. Derive the fractional lower bound directly from the original objective, then distinguish this inequality from the exact dual representation.
5. **Full inner solves and prediction-guided selection.** Define the actual feasible risk and dual lower value returned by each binary evaluation, then form its one owned tangent and retain the safe fraction from the same dual result. Their maximum screens proposals. A fixed Hamming-kernel Gaussian predictor uses only completed accurate evaluations; rank eligible proposals by mean minus standard deviation. Only full solves update the actual incumbent. Shared products turn the stronger binary master into an exact linear model with D additional continuous variables. Padding coordinates does not change either minimum. Define all accuracy and counting symbols before the quantitative theorem; prediction accuracy is not an assumption of that theorem.
6. **Instances and scope.** Only here introduce half hinge, logistic, exponential and squared hinge. Separate general mathematical coverage from the two implemented losses and from population calibration.

## Appendices

- **A → Theorem 1:** explain the two directions first. Restricted transports give the lower value bound; their LP dual identifies the finite objective. To reverse the bound, state the whole-space source-loss inequalities to be preserved, derive the interpolation weight from the endpoint allowances, and integrate against an admissible coupling. Finish with a compact-sublevel attainment argument. All endpoint, convexity, zero-distance and attainment steps remain explicit.
- **B → Theorem 2:** four connected goals: eliminate design variables to obtain the normalized dual; establish equality and attainment; normalize its source masses to derive the fraction denominator; return to binary selection and global cuts. Domains, scalar infimum rules, score elimination, both directions of allocation elimination and rescaling, strict feasibility, positive scale, binary maps and Hessian derivatives remain explicit.
- **C → Theorem 3:** a full solve supplies a reliable cut and an actual test; the global master certifies stopping or supplies an eligible candidate. The filter then forces a fixed drop in an earlier cut. Introduce movement and gradient bounds at this point to convert that drop to subset separation, and count distinct patterns on the largest-weight coordinates. The omitted-weight tail is strict. Prediction accuracy is not a condition of the optimization guarantee.
- **D → Section 6:** define the scalar loss and minimizing score before use; derive clipping, derivatives, substitution and square completion explicitly, with the boundary score limits and nonnegativity bounds shown.
- **E → Section 5:** the verified inner gap makes completed risk bounds usable approximate observations. Introduce similarities and their matrix roles, subtract the part linearly explained by the observations, and derive the conditional mean and remaining variance. Verify the kernel after this main calculation. Obtain the incremental factor from the augmented matrix by block multiplication; triangular solves compute predictions with quadratic work.

Appendices A–D retain the mathematical argument and intermediate steps; Appendix E describes the candidate predictor separately. Symbols and domains precede their use, and equations show both directions of claimed equivalences. Appendix C now also proves safe-fraction validity, full-budget equivalence and the shared-product master. `docs/ratio_cut_derivation.md` preserves the earlier exploratory derivation; earlier versions remain archived.

## Notation and maintenance

- All observations share one index. Class indicator and signed label are data; target labels do not restrict where either class can move mass.
- The coordinate-distance vector has only source and target indices. Class movement is a vector; no four-index distance tensor is displayed.
- The main text uses V for original binary risk and v for its convex extension.
- Section 5 uses a for the maximum of certified tangents and fractions, m for predicted risk mean, sigma for predictive standard deviation, and b for the shared target-times-selection products. Predictions rank candidates passing a <= U - epsilon/2; only actual completed tests update U, and only valid global master bounds update LB.
- Source allowances and scores are free primal variables. Their nonnegativity or finite bounds follow from constraints; do not change their dual coefficient conditions by adding unstated domains.
- The final main-text lower certificate has two unknown objects. Allocation multipliers and normalization variables appear only where they are derived in the appendix.
- Number main equations consecutively; appendices use A.1, B.1, etc. Use only labels that are needed for the argument.
- Keep both TeX sources and the English Markdown reading copy synchronized, and follow the reviewed paired build in AGENTS.md. Preserve prior versions.
- The 10 October revision changes the search implementation and its quantitative guarantee. The original 1,640 study records and their empirical report remain unchanged; new matched computational validation has a separate method version and directory. The whole-space construction is sufficient, not unique; the half-hinge/logistic implementations satisfy the same loss bounds.
