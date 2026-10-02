# Experimental protocol, frozen before implementation

**Protocol date: 3 October 2026.** The experimental sections and appendices of the local robust-testing and variable-selection papers were read before this protocol was written. The reading records are `literature_robust_testing.md` and `literature_variable_selection.md`. No new algorithm or experiment implementation preceded this protocol. Later changes must be recorded below rather than silently chosen after seeing results.

## 1. Questions and claims

There are two distinct questions. First, does the objective-scaled gate give a useful computational method for finding the globally best feature subset? Second, is minimizing a Wasserstein robust testing risk a useful feature-selection principle on independent data? A stronger continuous relaxation does not change the integer optimum. Thus a gate ablation measures optimization efficiency, not an improvement in the statistical objective at exact convergence.

The computational claims to test are integer exactness, primal/dual agreement, validity of cuts away from their generating support, stronger fractional bounds, and practical speed. The statistical claims to test are support recovery, independent two-sample testing power, null calibration, individual classification performance, and sensitivity to distribution shifts. We will report negative and inconclusive findings as well as favorable ones.

This is a reproducible research study with controlled synthetic mechanisms and two modest real datasets. It is not a reproduction of every source-paper table, a claim of clinical validity, or evidence of scalability to tens of thousands of variables.

## 2. Literature decisions

Wang et al., *Variable Selection for Kernel Two-Sample Tests*, motivates separate selection and test samples, power at level 0.05, false-discovery proportion (FDP), and non-discovery proportion (NDP), together with mean, covariance, non-Gaussian, and mixture alternatives. Its additive kernel is linear in coordinate weights; the component kernels are one-dimensional RBFs. The baseline must preserve this distinction.

Xie et al., *Robust Hypothesis Testing with Wasserstein Uncertainty Sets*, motivates small per-class training samples, mixtures with information that cannot be captured by a mean difference, and independent classification error. Gao et al. motivates examining radius sensitivity. The sequential change-detection experiments of these papers are not interchangeable with our fixed-sample feature-selection problem.

Swaroop et al., *Distributionally Robust Feature Selection*, motivates an independent downstream training split and a common downstream predictor. Its multi-population objective does not match our two class-conditional Wasserstein balls. We borrow the evaluation separation, not the misleading identification of class labels with populations. Hanada et al.'s safe screening retains a union of possible supports and is not a fixed-budget selector; it is discussed but not relabeled as a budget-matched baseline.

Bertsimas et al., *A Unified Approach to Mixed-Integer Optimization Problems with Logical Constraints*, motivates comparing incumbent objective and runtime under a common time limit, reporting unsolved cases, and ablating algorithm ingredients. Its experiments also show why a gap alone does not establish useful performance.

## 3. Algorithm and numerical controls

The main loss is half hinge (called `overlap`), so all oracle problems are linear programs. The primary solver stack is open source: SciPy/HiGHS for continuous oracles and the integer cutting-plane master. Logistic loss is a secondary correctness and deployment experiment using an open exponential-cone solver; it does not depend on the size-limited COPT license.

Every candidate support is evaluated by reoptimizing the full continuous oracle. No historical transport-plan mixture or fixed local neighborhood defines the proposed method. The risk-adaptive normalized dual produces a feasible lower certificate and a global tangent. An independently feasible fixed-support test supplies the upper bound. A master incumbent is not a lower bound; the master solver's global bound is used.

Use absolute stopping tolerance 1e-6 and relative tolerance 1e-5. Report achieved bounds and status on time limits. Numerical feasibility is checked and repaired conservatively; unrepaired failures are recorded, not treated as optimal. The simple default has a small continuous-root cutting stage and a common initial support chosen by univariate MMD screening. Core-point dual selection is a separately labeled ablation, with no unsupported claim that it always helps.

## 4. Computational experiments

### 4.1 Independent mathematical verification

For 20 random instances with D=6, three or four source atoms per class, nonuniform positive weights, unequal positive radii, and budgets 1--3, enumerate all supports. Compare the original fixed-support primal against an independent transport-budget dual, the new gate primal, and the normalized gate dual. Evaluate each cut on all binary supports and random fractional points. Include zero budget, duplicate samples, coincident classes, unequal class sizes, constant columns, and a hand-computable one-dimensional case. Logistic verification uses smaller instances and its own feasibility tolerances.

Tests answer mathematical questions rather than merely mirroring a solver's matrix assembly. They check that published bounds bracket independent enumeration, and that the deployed envelope test obeys its interval and selected-coordinate restrictions.

### 4.2 Gate and solver comparison

Compare (i) fixed box gates, (ii) shared triangle gates, (iii) risk-adaptive gates without root cuts, (iv) risk-adaptive gates with root cuts, and (v) enumeration on small problems. All share the same original objective, stopping tolerances, initial support, hardware, and loss. Exact objective equality is expected if every method solves the instance.

Small comparison: D=8 and 12, budget 2 or 3 respectively, n=8 per class, ten independent seeds, mean and mixture alternatives, theta0=theta1=0.25. Give each optimization method a 30-second limit. Enumeration is an independent reference only where affordable.

Scaling comparison: D=20,40,80, budget 3, n=10 per class, five seeds, mean and mixture alternatives, 30-second limits. Record solved fraction, total runtime, time spent in oracles/master, oracle calls, unique integer supports visited, root bound, final feasible objective, absolute/relative gap, and convergence traces. Timeouts remain in the comparison; do not average runtime only over solved instances. Use a log runtime axis and report paired comparisons, not just ratios of unrelated means.

Root-bound comparison uses identical fractional points and the same stopping budget. Pointwise theoretical dominance does not imply that a finite set of tangents yields a better master at every iteration; report both the oracle relaxation and the finite-cut master bounds.

## 5. Statistical data and independence

### 5.1 Synthetic mechanisms

Use four balanced two-class mechanisms in D=12 with a signal set of size 3 and feature budget 3. Shuffle feature positions for each seed, avoiding an implementation advantage from placing signals first.

* **Mean:** standard Gaussian coordinates with a class-1 shift 0.9 on the three signal coordinates.
* **Variance:** zero means, independent coordinates, with class-1 signal SD 1.8 and other SDs 1.
* **Dependence:** both classes have identical standard-normal marginals. Class 1 has equicorrelation 0.65 on the three-coordinate signal block; class 0 has identity covariance. This tests whether selecting coordinates jointly finds information invisible to univariate screening.
* **Mixture:** class 0 is standard Gaussian; class 1 has a symmetric shared-sign shift +/-1.5 on the three signal coordinates plus Gaussian noise. The means coincide and the signal coordinates are dependent through the mixture component.

These are explicit adaptations of source-paper mechanisms, not literal copies of their sample sizes or all parameter values. Use 30 independent training seeds (0--29). Each seed has n=20 selection observations per class, n=20 independent downstream-training observations per class, and 1,000 fresh classification observations per class. Preprocessing uses only pooled selection data: center and divide by pooled sample SD, with a floor for constant columns. This fitted transform is frozen for every later split and every method in the trial.

The Wasserstein ground cost is L1 after this common transformation. Radii theta0=theta1=0.25 are fixed in the primary comparison. They do not change with the selected support. Radius sensitivity uses 0.10,0.25,0.50 on ten predeclared seeds for mean and mixture data. This is sensitivity analysis, not test-set hyperparameter tuning. Baseline hyperparameters are fixed below; the report does not claim that this study identifies an optimal radius.

Each selected support is also evaluated using 20 fresh alternative two-sample datasets and 20 fresh null datasets, n_test=40 per class. Under the null, both samples are independently drawn from the class-0 distribution. The selected support and all preprocessing/kernel parameters stay frozen. Repeat the entire selection procedure across the 30 training seeds. Aggregate test replicates within a seed first; bootstrap training seeds for uncertainty. Do not call the 600 conditional replicates 600 independent selections.

### 5.2 Real data

Use the locally sourced UCI WDBC (569 x 30) and Ionosphere (351 x 34), with their IDs/labels excluded. The source manifest includes URLs, DOIs, licenses, and hashes. Use 20 independent stratified splits. Within each split allocate n=20 per class to selection and another disjoint n=20 per class to downstream training. The remaining records are held out for final classification. Transformations and any constant-column handling are fitted on selection data only. Budget is 3 and theta=0.25. Give robust global selection 30 seconds and report its bound/status; a time-limited incumbent is labeled as such.

Real-data results include balanced error, randomized balanced error, AUC, surrogate loss, worst measured stress error, support frequency and stability. Ground-truth support recovery is undefined and omitted. Overlapping random splits are reported as repeated-split variability, not independent population-level confidence intervals. No clinical interpretation is inferred.

## 6. Selectors and reproduction status

All budget-matched methods observe the same selection data and choose at most d coordinates. Evaluators are identical afterward.

1. **Risk-adaptive global selection (ours):** the newly derived exact objective and certified global cuts. A time-limited run returns its best feasible test/support plus a bound.
2. **Robust greedy:** add the coordinate with the lowest fully refitted original robust risk at each step. Same loss/radius as ours, but no global optimality guarantee. This isolates the value of global search.
3. **Univariate MMD screening:** rank unbiased marginal RBF MMD values. A transparent marginal baseline, not a reproduction of Wang's optimized additive kernel.
4. **Wang additive-MMD selection:** reconstruct the paper's mean/variance sparse trust-region objective from one-dimensional RBF kernels. Use lambda=0.1 and per-coordinate training-only median cross-distance bandwidth, sigma factor 1. Preserve unit-norm sparse weights. For D<=12 use support enumeration with a globally solved trust-region subproblem; for larger D use a documented forward/swap support search for the same objective and label it approximate. Test the coefficients and objective against the official revision source on fixed fixtures. Do not redistribute its unlicensed source. Official repository/commit are recorded. Negative weights, if the objective permits them, are kept for native-statistic diagnostics; common downstream evaluation uses the support only.
5. **L1 logistic ranking:** fit sklearn liblinear L1 logistic with C=1 on selection data and retain the largest absolute coefficients, with deterministic variance-based ties. This follows a source-paper baseline family; its top-d step is an explicit budget adaptation.
6. **Random selection:** a seeded budget-sized negative control.

An all-feature common test and the known signal support are diagnostic references for synthetic data, not budget-matched deployable competitors. The all-feature reference uses D variables; the signal reference is unavailable in practice. The report distinguishes each method's objective from the common downstream evaluator. We do not claim faithful replay of unpublished hyperparameter grids.

## 7. Evaluators and metric definitions

### 7.1 Variable selection

For selected S and known signal T, FDP=|S\T|/max(1,|S|), NDP=|T\S|/|T|, precision=|S intersect T|/max(1,|S|), recall=1-NDP, and F1 is the harmonic mean of precision/recall (zero if both vanish). Stability is average pairwise Jaccard between selections across seeds, calculated in the unshuffled semantic feature indexing so coordinate shuffling does not artificially destroy stability. Real-data coordinates are fixed. Because correlated variables may help condition out noise, signal recovery is evidence to interpret alongside risk and power, not a definition of Bayes optimality.

### 7.2 Equality testing

The common test is unbiased quadratic RBF MMD on the frozen selected support. Its pooled training median distance fixes bandwidth. Use 199 random balanced label permutations, plus the observed assignment, and p=(1+number of permuted statistics >= observed)/(199+1). Reject when p<=0.05. The pooled test Gram matrix and bandwidth do not depend on the test labels. Under the null, exchangeability conditional on the independent selection data justifies this calibration. Null rejection rate and alternative power must be displayed together.

This evaluates how useful the support is for a common two-sample test. It is not a theorem that the robust-risk objective controls level-0.05 power. We separately provide native additive-MMD diagnostic results where implemented; they are labeled differently from the common-support comparison.

### 7.3 Individual robust-test deployment

Refit the same half-hinge robust envelope head on every selected support using the independent downstream-training data. Let p0(x) be its probability of accepting class 0. Randomized Type-I is mean_0(1-p0), Type-II is mean_1(p0), and randomized balanced error is half their sum. Deterministic error thresholds p0 at 0.5, with an explicit tie rule. ROC AUC uses 1-p0 as the class-1 score. Also report the two-class surrogate sum and its balanced average, with the factor of two stated.

Add a common RBF SVM head as an evaluator of support utility, using C=1 and training-only median bandwidth; this prevents conclusions from depending exclusively on our own envelope deployment. Its individual balanced error and AUC are reported separately.

### 7.4 Distribution shift

Use paired deterministic translations of class k toward the other class's downstream-training mean, normalized to L1 magnitude rho=0,0.25,0.50 in the common standardized coordinates. Evaluate fresh held-out observations under these translations; use the same direction and transformed observations across selectors. Report error as a function of rho and the maximum across this finite stress set.

A translation of size rho has a coupling cost rho relative to the unshifted distribution. This does not show that the shifted population lies in a ball around the finite training empirical measure. Accordingly this is explicitly called a population stress experiment. A separate source-atom translation within theta is evaluated to check the fitted certificate, where membership in the empirical Wasserstein ball is established by the source-to-shifted-atom coupling.

## 8. Uncertainty, records, and presentation

Write one machine-readable record per seed/method/dataset, preserving selected coordinates, solver status/bounds, timings, radius, preprocessing, split seed, permutation seed and metrics. Store traces separately. Preserve failure rows. Aggregate synthetic metrics with a 95% bootstrap interval over training seeds (2,000 resamples with a fixed seed). For real data use mean +/- SD across splits and explicitly state their dependence. Support stability is a descriptive aggregate. Pair methods within the same split/seed when reporting differences. No uncorrected scan across many metrics is called statistical significance.

The report order is question -> method/data -> why each metric answers that question -> figure -> interpretation -> limits. Plan figures before plotting: (1) gate mechanism and a one-coordinate analytic example, (2) runtime/visited-support and bound-strength panels, (3) power beside null rejection, (4) support FDP/NDP, (5) stress curves, (6) real-data compact outcome table and support-frequency panel. Use consistent method colors, units, uncertainty bars, readable labels, vector outputs and restrained booktabs tables. Captions state n, repetition unit, uncertainty type, and whether higher/lower is better. Detailed all-metric tables belong in an appendix.

Every reported result is generated from saved records. The final report includes environment/hardware, reproduction commands, baseline provenance, actual completed run counts, timeouts and failures. Paper-level reporting requires transparency and calibration; it does not permit an inflated novelty or performance claim.

## 9. Change log

* 2026-10-03: protocol frozen after literature review, before implementation. Numerical tolerances and practical execution details may be corrected after correctness pilots; the correction and reason will be recorded here before the affected full runs.

* 2026-10-03 supplemental design: primary timing showed that tangent masters can remain weak despite pointwise relaxation dominance. Before new runs, froze `configs/ratio_cuts.json` (new computational seeds 10--14) to compare the same normalized dual oracle with a master retaining complete fractional certificates. Shared product variables and incumbent level cuts are derived in `docs/ratio_cut_derivation.md`. This is an experiment-motivated exploratory extension; it does not replace or relabel the predeclared primary comparison. Primary runs and defaults stay unchanged.

* 2026-10-03 numerical policy: a deterministic same-MILP probe showed a HiGHS status-4 presolve failure that succeeded without presolve. Added one same-model retry, solely for status 4, within the original remaining budget. Subsequent runners use this policy. After the two already-running primary runners finish, every affected failed record is archived and rerun uniformly; original failed time remains charged. The objective, data, seeds, cuts and evaluation protocol are unchanged. See `docs/numerical_remediation.md` for the evidence and preservation rules.
