# Independent audit of the theory and implementation

This is the historical 3 October 2026 audit. The current 0.4.0 search revision and code-to-theory mapping are documented in `docs/adaptive_search.md`; its separate checks and matched validation are under `reports/verification/20261010/`. The observations below describe the earlier algorithm version.

Reviewed 3 October 2026. This was a read-only mathematical and source-code audit; it did not rerun the experimental suite or change the algorithm. The parent task separately reported 64 passing tests. That result is reported evidence, not an additional test run performed by this audit.

**Verdict.** No material mathematical error or theory-to-code inconsistency was found under the stated assumptions: two classes, the two specified losses, strictly positive Wasserstein radii, positive normalized source masses after removing zero weights, fixed feature scaling, full ambient space, and L1 transport cost. The implementation reoptimizes the complete continuous problem at each new candidate support. Its master searches globally; historical transport mixtures do not define its search space. Numerical results should be described as certificates to the declared floating-point tolerances, rather than exact arithmetic guarantees.

## 1. The finite model really corresponds to the original objective

The finite target pool is not accepted merely because it contains the training observations. I checked both directions of Appendix A independently.

For a fixed pool score, the adversary is a finite LP with source-row masses and one total transport budget per class. Its ordinary LP dual gives the source potentials and nonnegative distance prices. Restricting the adversary to the pool yields a lower relaxation of the original full-space risk.

The reverse direction is the important part. For any finite feasible potentials, define the two distance envelopes. At each pair of opposite-class source points, the pool constraints imply two endpoint inequalities. For logistic loss, convexity of

\[
e^{-a_0-\lambda_0 r}+e^{-a_1-\lambda_1(d-r)}
\]

and the triangle inequality imply that the two probability-interval endpoints are compatible at every observation. For half hinge, the endpoint constraints and the triangle inequality imply that the two envelopes sum to at least one. Their midpoint therefore defines a measurable test on the whole selected-coordinate space. Integrating its pointwise losses over admissible couplings yields the finite objective as an upper bound. Thus the two inequalities establish equality without an unsupported infinite-dimensional minimax exchange.

This argument remains valid when samples coincide after projection: the extension constructs one consistent test even if the finite program temporarily allows separate scores at duplicate pool entries. The full-space projection/lifting argument also preserves the radii exactly. It would require reconsideration on a restricted observation domain or for a different loss.

Sources checked: `note/main.md`, Appendix A; `oracle.py`, `RobustTest.interval` and the fixed-support primal paths.

## 2. The logical dual and the strengthened dual are consistent

The grouped triangle is justified by the objective itself: nonnegative source potentials give theta·lambda <= J <= B at an optimum. Saturating a selected coordinate's price to lambda can only relax its loss inequalities, because coordinate differences are nonnegative. Both facts are necessary for binary exactness, and both are used correctly.

I checked the conjugate sign for the nonsymmetric nonnegative triangle. Its support function is B times the maximum of zero and the two radius-normalized multipliers. The resulting coverage constraints are A <= beta + theta*mu, with beta's classwise allocation budget <= theta. These are expanded lower-bound certificates; they need not be original budget-feasible adversarial transports. The note explicitly preserves this distinction.

For the risk-scaled gate, the normalization tau + mu_hat·z = 1 comes from eliminating the free epigraph variable. Dividing a positive-scale certificate by tau gives a certificate in a fixed domain, and homogeneity yields H/(1 + mu·z). A zero-scale certificate has zero mass and zero objective; no invalid division is required. Each fixed-certificate fraction is convex, so its tangent is globally valid. The gradient -H(Gamma_hat)*mu_hat used in the code has the correct scale and sign.

Eliminating eta in the implemented primal is valid even at fractional masks. Given J < eta, raising every source potential by (eta-J)/2 makes J equal eta because both classes' weights separately sum to one. It relaxes the loss constraints and leaves the coordinate allowance unchanged. The implementation therefore minimizes the same value function; it is not simply assuming an unexplained active constraint.

The dominance proof compares exact value functions over the same certificate domain. It does not establish dominance between arbitrary finite sets of returned cuts or guarantee universal runtime improvement. The note states these limits.

Sources checked: `note/main.md`, Appendices B and C; `oracle.py`, both LP/cone dual paths and `_repair_dual`.

## 3. Lower certificates and feasible-test upper bounds remain separate

The primal repair recomputes every potential from the actual losses and distances, rather than trusting reported solver objectives. At a fractional risk gate, reducing coordinate prices may increase those recomputed potentials and J. This direction makes the repaired gate remain feasible after the single scaling pass. At an integer support, the returned deployment certificate comes from the original fixed-support problem.

The dual repair clips negative masses, restores source-row masses, scales beta back inside its allocation budget, recomputes coordinate movement and the required nonnegative residual prices, and finally restores the risk normalization. The returned entropy and gradient are recomputed from this repaired certificate. The small margin lowers the cut by a constant; it does not substitute a primal upper objective into its gradient. A repaired certificate can be nonoptimal, but its own tangent remains a valid lower cut.

Core-point selection uses a numerical relaxation of the dual-optimal face. This does not justify calling every resulting cut an exact touching cut. The implementation remains safe because it repairs the returned certificate and uses its own recomputed entropy. Appendix D explicitly distinguishes the ideal exact-face argument from this numerical case. Cut quality and exact face membership can change after repair; validity does not depend on retaining the face.

The deployed class-0 probability agrees with the note's score sign. For half hinge, the certified risk is the sum of randomized errors; its balanced average is half that sum. Thresholded deterministic error is a separate measurement. The code exposes both probabilities and hard labels, and the metric/documentation paths keep these quantities separate. Logistic predictions use the corresponding exponential envelope interval.

Sources checked: `oracle.py`, `_repair_primal`, `_repair_dual`, `solve`, and `RobustTest`; `experiments/metrics.py`; `docs/code_guide.md`.

## 4. Global search and stopping do not assume a local optimum is global

The integer master contains the accumulated global cuts and a full-budget equality. Full-budget restriction is exact because a test can ignore added coordinates. All returned cut coefficients are nonpositive, so padding is also compatible with the continuous master. Neither fixed support-local neighborhoods nor fixed historical transports restrict the candidate set.

The solver updates the global lower bound from the integer master's `mip_dual_bound`. It uses a master objective in its place only when the master has solved optimally. Fractional oracle calls add cuts but never update the deployable integer-test upper bound. The warm support is fully evaluated, and the best integer feasible test supplies the upper bound.

The exact finite-termination proof depends on tight oracle cuts and exact master solutions: revisiting an evaluated support forces equality of the global bounds. The numerical implementation instead reports an accuracy limit if it revisits a cached support while the gap is unresolved. Caching an inexact result can therefore stop progress, but it does not silently produce an exact-optimality claim. The explicit `optimal` status is governed by the declared absolute and relative bound tolerances.

Sources checked: `note/main.md`, Appendix D; `solver.py`, `_master_impl`, `evaluate`, global bound update, and repeated-support handling.

## 5. Deliberate boundary and counterexample checks

These are analytic checks, not additional experiments:

| Case that could invalidate a careless derivation | Result of the check |
|---|---|
| Empty support | All coordinate prices vanish; the best constant test has risk B. The normalized dual permits tau=1. |
| Identical observations from the two classes | The constant-test risk B is attained, and no separation or unique informative support is asserted. |
| Duplicate projected atoms | The whole-space extension supplies a single consistent probability at that projected point. |
| Unequal class sizes or nonuniform source weights | Separate normalized row constraints preserve both classes' total mass at one; no equal-size assumption is hidden in the entropy or eta-elimination step. |
| Unequal positive radii | Each class keeps its own beta budget and radius factor in residual coverage. |
| Nonoptimal or numerically relaxed core-point certificate | Its own entropy gives a valid non-touching tangent; primal risk is not used to make it artificially tight. |
| Interrupted master with a good incumbent | Only the global bound is used as LB, so an attractive incumbent cannot create a false optimality certificate. |
| Zero radius | The public problem object rejects it. The note explains why the compact-price and attainment arguments cannot be extended by direct substitution. |

The hand example in the note also separates integer exactness from fractional strengthening: box, triangle, and risk-scaled gates agree at binary masks while taking different fractional values. That rules out interpreting the new gate as a change to the statistical objective.

## 6. Limits of this audit

The repairs, residual checks, and conservative margins operate in ordinary floating-point arithmetic. They are useful and internally consistent numerical safeguards, but no directed rounding or interval-arithmetic bound proves every displayed decimal rigorously for arbitrarily ill-conditioned inputs. Very small radii, extreme coordinate scales, or tiny source masses can reduce practical numerical reliability; the code already rejects nonfinite data and overflowing coordinate differences. This audit does not establish a formal all-input numerical-error theorem.

The independent transport-LP tests provide stronger evidence than agreement of two gate formulations, because their reference uses only the original source marginals and classwise total-cost budgets. The parent-reported passing suite also covers analytic curves, cross-support cut validity, unequal weights, duplicates, deployment, and stopping behavior. Those finite checks support the implementation; the mathematical claims above rely on the derivations rather than on the tests alone.

Finally, exact optimization of the empirical robust objective does not establish population coverage of a chosen radius, a prescribed significance level, power superiority, or scientific uniqueness of the selected variables. These require the separate frozen evaluation protocol.
