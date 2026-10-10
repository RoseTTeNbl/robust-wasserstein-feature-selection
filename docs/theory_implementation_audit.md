# Independent audit of the theory and implementation

Updated 10 October 2026 for retained-fraction adaptive search (`full_inner_fraction_v4`). The current path keeps the previous full binary evaluations, fixed-kernel ranking, pool of at most eight proposals and three-evaluation master interval. It also retains a safe fraction from the same completed dual certificate and uses the shared-product binary master. Previous `full_inner_uncertainty_v3` verification is identified separately below; its numbers are not results for the new path.

The note assumes two classes, positive radii, separately normalized positive source masses, full ambient observation space, L1 transport cost, and a finite, convex, nonnegative, nondecreasing loss with positive value at zero. The implementation supplies half-hinge (`overlap`) and logistic losses; it does not expose an arbitrary-loss oracle.

## Current theory and code mapping

| Component | Implementation | Note location |
|---|---|---|
| Original risk, exact finite formulation, whole-space test | `Problem`, fixed-support primal, `RobustTest` | Sections 1–2; Appendix A |
| Risk-scaled primal, normalized dual, certificate repair | `GateOracle`, `_repair_primal`, `_repair_dual` | Sections 3–4; Appendix B |
| Source-weighted multiplier bounds, one owned tangent per complete binary evaluation | `Problem.price_bounds`, `GateOracle.mu_bounds`, `solve_adaptive.evaluate`, `EvaluationHistory.add` | Section 5; Appendix C |
| Lower model from owned tangents and safe fractions; global binary master | `EvaluationHistory.lower`, `solver._ratio_master`, `solve_adaptive.master` | Section 5; Appendix C; `docs/ratio_cut_derivation.md` |
| Fixed-kernel ranking from actual evaluated risks | `FixedKernelPredictor.append` and `.predict` | Section 5; Appendix E |

Appendix D derives scalar loss formulas; Appendix E derives the fixed predictor. The alternative master-only search, box/triangle formulations and legacy `cut_mode="ratio"` remain comparison paths. The default adaptive path now uses the same exact shared-product representation for its retained safe fractions. `use_fractions=False` restores the earlier tangent-only adaptive model without changing prediction or complete-evaluation requirements.

## Exact finite model and repaired certificates

Appendix A proves both directions without an infinite-dimensional minimax exchange. Projection and coupling lifting preserve the selected-coordinate transport problem. Restricting targets to pooled observations gives the finite adversarial LP; source-marginal and budget multipliers yield its potentials and distance prices. Conversely, distance envelopes and convex interpolation between opposite-class sources construct a bounded measurable whole-space score satisfying the finite loss allowances. Integrating these inequalities over admissible couplings supplies the reverse bound. The construction extends feasible loss bounds; it need not interpolate every finite score, including scores at duplicate projected observations.

The code's loss-specific envelope intervals supply feasible tests for its two supported losses. For overlap, risk is the sum of the randomized class errors; its balanced average is half that sum. Hard-threshold error is a separate metric. Logistic probabilities use the corresponding exponential-envelope interval.

Eliminating the free risk epigraph variable gives the normalized dual relations

\[
\tau+\bar\mu^Tz=1,\qquad \bar\Gamma\mathbf1=\tau q.
\]

Positive scale gives the fixed-domain fraction \(H(\Pi)/(1+\mu^Tz)\); zero scale has zero row mass and objective. Convexity and cuts follow from this fraction, not from joint convexity of the primal product of risk and selection.

Primal repairs recompute potentials from actual losses and distances, producing feasible upper bounds. Dual repairs restore row masses, allocation budgets, residual coverage, and normalization, then recompute their own objective. A dual value never becomes a feasible-test upper bound; a primal objective never substitutes into a dual gradient.

The oracle uses the repaired raw-fraction gradient and shifts the affine tangent down by a numerical margin:

\[
\ell_s=f_s(z^s)-e_s,\quad e_s\ge0,\qquad
L_s(z)=\ell_s+\nabla f_s(z^s)^T(z-z^s)\le f_s(z).
\]

This differs from reducing the fraction numerator and recomputing its gradient. The actual evaluation gap \(R_s-\ell_s\le\delta\) includes the margin. The source-weighted bounds \(A_{kj}\le\theta_kM_j\) justify multiplier clipping before objective/gradient calculation and imply \(-BM_j\le g_{s,j}\le0\). `Problem.price_bounds` calculates these read-only bounds once; oracle clipping and search diagnostics share the same cached array.

## Retaining the same complete certificate

For repaired normalized values \(h,\tau,\bar\mu\), the raw certificate is \(f_s(z)=h/(\tau+\bar\mu^Tz)\le V(z)\). With the same numerical margin, the saved fraction is

\[
n_s=\max(0,h-e_s),\qquad F_s(z)=n_s/(\tau+\bar\mu^Tz).
\]

Positive \(\tau\) and nonnegative \(\bar\mu\) give positive denominators, and \(0\le F_s\le f_s\le V\). A zero or missing usable fraction is skipped; the owned tangent remains. The fraction and tangent must belong to the same retained inner attempt, and their values at the evaluated support must agree within the declared numerical check.

The shifted affine tangent and reduced-numerator fraction need not dominate each other at every point. Consequently `EvaluationHistory.lower` takes the maximum of zero, every tangent and every usable fraction. No new point is solved and no new tangent is generated by these substitutions.

For binary z, shared variables \(y_j=\eta z_j\) and an actual feasible risk bound U give an exact MILP representation. Each fractional row is \(n_s\le\tau_s\eta+\bar\mu_s^Ty\). All rows share D additional variables, retain the affine constraints and use updated product bounds when U changes. This strengthens the candidate lower model but does not by itself imply lower wall time.

## Complete binary evaluations and global convergence

Every selected subset is binary and solves its complete inner problem. Refinements stay at that subset; only its final raw certificate contributes one owned tangent. The active search has no fractional root solves, transferred-test repricing, recentered tangents, or predicted-risk incumbent updates. A reused solver workspace still executes a fresh optimization. Persistent cone paths explicitly reject stale values after `solver_error` or a no-execution `time_limit`; cache-prime regressions cover both primal and dual paths.

The incumbent is \(U_t=\min(B,R_s)\) over actual feasible evaluations. The lower model is \(a_t(z)=\max(0,L_s(z),F_s(z))\) over all retained certificates. A valid previous master bound remains valid as cuts are added, permitting a running maximum lower bound. An interrupted master's incumbent alone cannot establish a global lower bound. Candidate precision is checked against the running bound; an inadequate interrupted solve returns `time_limit`.

For absolute tolerance \(\epsilon\), the theorem requires \(\delta,\xi\le\epsilon/4\) and accepts only

\[
a_t(z)\le U_t-\epsilon/2.
\]

When the global gap exceeds \(\epsilon\), a master candidate with \(a_t(z_M)-LB_t\le\xi\) passes this filter. If bounded proposals supply no eligible subset, the algorithm therefore uses the global master. The first accurate evaluation is checked for immediate stopping.

The separation proof still applies because the stronger lower model retains every owned affine cut. Owned cuts separate distinct complete evaluations by weighted distance at least \(r=(\epsilon/2-\delta)/B\). A sorted-weight prefix whose omitted tail is strictly less than \(r\) gives Appendix C's binomial evaluation bound. The first evaluation counts; same-point refinements are extra solver calls, not extra distinct evaluations. This bounds evaluations under the accuracy assumptions, not wall time. Time or accuracy exits retain valid bounds without invoking convergence. Exactly-budget search is legitimate because a test can ignore added coordinates. Optional relative tolerance uses the declared combined stopping target; the matched validation uses relative tolerance zero.

## Prediction only ranks eligible subsets

The model learns from accurate actual risks \(R_s/B\), with mean \(1/2\), kernel

\[
k(z,z')=\tfrac14\exp(-\|z-z'\|_1/\max(1,d)),
\]

and fixed regularizer \(\nu=\max(10^{-8},(\delta/B)^2)\). Each binary-coordinate factor is positive definite. The full-cube kernel is a tensor product; distinct observed subsets form a principal submatrix. Adding \(\nu I\) gives a strictly positive-definite observation system. Appendix E's auxiliary Gaussian conditioning calculation yields the implemented mean and standard deviation.

Incremental lower Cholesky extension and triangular solves avoid explicit inverses, dense factorization rebuilds, and hyperparameter fitting. Append and a bounded prediction batch require quadratic work in the number of observations, plus kernel evaluation. Mean minus standard deviation ranks candidates only after the certified-cut filter. Neither prediction is a certified risk bound or calibrated statistical confidence interval. Convergence does not assume predictor accuracy.

## Recorded v3 verification and limits

For the previous tangent-only v3 release, the independently run final suite passed **150 tests**; Ruff passed for `code` and `scripts`. The log retains one Clarabel accuracy warning. All **78 method outcomes** in the final **26 matched cases** were optimal; all 26 exhaustive references completed and their intervals were consistent. Source snapshots match the benchmark manifest. All **1640 frozen research records** retain their previous hashes. The final bilingual Appendix C and E mathematical blocks match.

Among 449 acquisitions, 287 eligible pools had nonconstant predicted standard deviations, and 197 selections differed from deterministic minimum-cut ranking. Thus prediction actually affected ranking. Adaptive search used **475 full calls versus 467** for the master baseline: this cohort does not show aggregate call reduction. It used 209 global masters versus 467 and was faster in 25 of 26 single-timing pairs. These timings do not establish universal superiority. Earlier geometric runs remain preserved separately.

That v3 audit checked actual-test feasibility, binary queries, one-cut counts, gradient bounds, and that predictions never update the incumbent. Current retained-fraction performance must be read from the separate development and frozen confirmation records under `reports/verification/20261010/next_selection/`. Repairs and margins operate in ordinary floating-point arithmetic, not a directed-rounding theorem for every input. Optimization certificates do not establish population coverage of uncertainty sets, significance control, or power superiority; these require separate statistical calibration and evaluation.

Evidence: [final audit](../reports/verification/20261010/full_inner_revision/final_gp_audit.json), [tests](../reports/verification/20261010/full_inner_revision/tests_final_gp.log), [Ruff](../reports/verification/20261010/full_inner_revision/ruff_final_gp.log), and [matched validation](../reports/verification/20261010/full_inner_uncertainty/summary.json).


## Refactor and failure-handling checks

The logistic primal and dual models are each constructed in one function; ordinary and persistent solve paths share these definitions. Parameter updates and workspace reuse still execute a new optimization for every selected z. Existing cached-value guards remain in place.

The source-weighted bound was moved without changing its arithmetic. A weighted two-coordinate hand example yields \(M=(21.25,13.75)\); the returned array is cached, read-only and shared with `GateOracle.mu_bounds`. Eighteen before/after actual logistic solves across all gates, core-point settings and binary/fractional inputs gave identical upper bounds, lower bounds and gradients. Earlier canonical-model equivalence checks remain in `next_selection/refactor/`.

`_milp_with_retry` permits one unchanged-model presolve-off retry after a solve error. An infeasible status permits that retry only when `_check_milp_witness` establishes a complete feasible incumbent in the stored linear model using exact rational arithmetic on the floating coefficients. Missing or invalid witnesses do not authorize this retry. All failure fields are cleared, both calls share a wall-time budget and unsuccessful retry statuses remain visible. This policy is independently covered by `test_master_retry.py` and `test_ratio_witness_retry.py`; it does not rewrite historical research records.
