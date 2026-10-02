# Baseline reproduction and adaptation record

This record was prepared after reading the source experimental sections and freezing `docs/experiment_plan.md`, before full runs. It distinguishes a reproduced mathematical criterion from literal replay of a published table. The experiment modules are independently written; no third-party source is vendored.

## What each method contributes

| Record name | Selection input and objective | Reproduction status |
|---|---|---|
| `risk` | Selection split; exact-budget Wasserstein half-hinge risk, global cuts | Our risk-adaptive method |
| `greedy` | Same loss/radius; fully refit the fixed-support optimum at each forward addition | Transparent greedy comparator, not a globally certified selector |
| `mmd` | Selection split; rank unbiased one-coordinate Gaussian-kernel MMD | Marginal screening control, not Wang's full method |
| `wang` | Selection split; sparse unit-norm additive-kernel MMD minus estimated variance | Independent reproduction of the paper's criterion with the differences below |
| `l1` | L1 logistic fit, then top-budget absolute coefficient ranking | Adaptation of the sparse-logistic baseline family |
| `random` | Uniform budget-sized subset from a fixed independent seed | Negative control |
| `signal` | Known generating signal support | Synthetic oracle diagnostic, unavailable in practice |
| `all` | All coordinates | Diagnostic with a larger acquisition budget |

Every selected support is subsequently evaluated with the same independent robust head, the same independent RBF-SVM head, and the same frozen-bandwidth two-sample RBF-MMD test. This comparison asks whether the selected coordinates are useful; it does not unfairly grant each selector its own preferred downstream prediction architecture. The head-training data are separate from selection data, and final observations are separate from both.

## Wang additive-MMD: mathematical reconstruction

Source: Wang, Dey, and Xie, *Variable Selection for Kernel Two-Sample Tests*, arXiv:2302.07415v4. Official repository: [WalterBabyRudin/MMDVar_Selection](https://github.com/WalterBabyRudin/MMDVar_Selection), inspected commit `031f740e9ca6672751145b16de62cfaf731e98de`. Local reference files are `external/code/MMDVar_Selection/Optimization/Linear_MMD_revision.py` and `MIQP_solver.py`. They are read-only references and are excluded from publication of our source tree.

The additive kernel is

\[
K_z(x,y)=\sum_j z_j\exp[-(x_j-y_j)^2/(2b_j^2)],
\quad \|z\|_2=1,\quad \|z\|_0\le d.
\]

Here "linear" means linear in kernel weights, not a raw linear dot-product kernel. The bandwidth convention follows the revision coefficient function: `b_j^2` is the median of squared cross-class selection distances, with sigma factor one. A floor of `1e-12` prevents division by zero on constant coordinates.

For equal class size n, form coordinate contrasts

\[
h_{il,j}=k_j(x_i,x_l)+k_j(y_i,y_l)-k_j(x_i,y_l)-k_j(y_i,x_l).
\]

The unbiased paired mean coefficient is

\[
t=\frac{1}{n(n-1)}\sum_{i\ne l}h_{il}.
\]

Define row means `r_i = n^{-1} sum_l h_il` and `r_bar = n^{-1} sum_i r_i`. The source variance estimator is

\[
\widehat\Sigma=\frac4n\sum_i(r_i-\bar r)(r_i-\bar r)^\top.
\]

The paper's Eq. (11) says to maximize mean minus variance. The implemented paper criterion is therefore

\[
\max_{\|z\|_2=1,\|z\|_0\le d}
t^\top z-\lambda z^\top\widehat\Sigma z,
\qquad\lambda=0.1.
\]

`wang_coefficients` returns `A=-lambda*Sigma`, `t`, and training bandwidths. It also returns the revision-source coefficient matrix for an explicit comparison fixture.

### A source discrepancy that is not hidden

The inspected revision coefficient source returns **positive** `A=+lambda*Sigma`, followed by an eigenvalue shift. The inspected `MIQP_app_solver` evaluates `t^T z+z^T A z` and takes `np.argmax`; it does not negate A before optimization. Thus that combination rewards variance, whereas the paper's stated objective penalizes it. The primary baseline is labeled **Wang additive-MMD (paper objective)**. It is not called a literal replay of the revision script. The coefficient test uses independent scalar kernel calculations and a row-covariance calculation to check both the paper-correct matrix and the revision-source sign.

The repository's testing routine also recomputes cross-class bandwidths on final testing labels and uses a different paired U-statistic permutation construction. Our native additive diagnostic freezes bandwidths from selection data and uses a standard unbiased two-sample statistic with plus-one Monte Carlo calibration. This is an explicit evaluation-protocol adaptation to preserve the held-out calibration argument. Published source-table numbers are not used as our reproduced outcomes.

### Solving the support and weight problem

For D at most 12, enumerate every size-d candidate support. Supports of smaller effective size are covered because the fitted weight vector may have zero entries. For each support S, solve

\[
\max_{\|x\|_2=1} x^\top A_{SS}x+t_S^\top x
\]

globally. In an eigenbasis of `A_SS`, stationarity gives `x_i=(t_i/2)/(lambda-d_i)`. The secular norm equation determines lambda above the largest eigenvalue. If the linear term vanishes on the top eigenspace, the implementation also handles the hard case by filling the remaining unit norm in that eigenspace. This avoids a local gradient optimizer masquerading as an exact baseline.

For larger D, the same objective is optimized by forward support addition followed by up to ten best-improving one-coordinate swaps. Every candidate's weights are globally refit. The record is marked `approximate_forward_swap`; no global selection guarantee is claimed. Negative optimized weights are kept, as allowed by the paper's coefficient domain. A native weighted-kernel statistic with negative weights is still permutation-calibrated when frozen independently of test labels, but it should not be described as a positive-definite-kernel MMD unless the weights are nonnegative.

Hyperparameter lambda is fixed by the preregistered protocol, rather than tuned to final test outcomes. This preserves the criterion but differs from source cross-validation and source-table parameter sweeps.

## Other selectors

The univariate screening baseline uses the same coordinate kernels and the off-diagonal paired MMD means; only their ranking is used. Ties are broken by feature index. The L1 selector uses scikit-learn `LogisticRegression(solver='liblinear', l1_ratio=1, C=1, max_iter=2000)`, which specifies pure L1 in the pinned current API, and ranks absolute coefficients. Zero/equal coefficient ties use selection-split variance and then index. Top-d ranking is the explicit adaptation from a regularization-selected support to a common acquisition budget.

The random selector uses an RNG stream reserved for random selection. The robust greedy selector evaluates every available addition with a fully reoptimized fixed-support problem. It shares the global method's loss and radius, but forward decisions can miss useful joint combinations. If its shared wall-time limit expires, it keeps its last complete forward stage and fills the remaining budget with deterministic marginal-screening candidates, recording `greedy_time_limit`.

Swaroop et al.'s multi-population DRFS and Hanada et al.'s safe-screening union support are discussed in the literature record but not silently transformed into budget-matched competitors. The former requires genuine deployment populations and the latter can return an arbitrarily large safe superset. The six primary selectors in this report share a well-defined two-class, fixed-budget task.

## Common evaluators and why their metrics differ

The **common equality test** uses a support-specific RBF Gram matrix and a bandwidth fitted only on pooled selection samples. It computes an unbiased MMD statistic, generates 199 random fixed-size label assignments, and uses `p=(1+#permuted>=observed)/200`. Every selector in a given trial receives the same test observations and permutation RNG seed. The null and alternative testing datasets are independently generated. Rejection fractions within a training seed are averaged first; uncertainty across selectors is paired by training seed.

The **common robust prediction head** is refitted on the independent downstream-training split using the same half-hinge Wasserstein risk. It produces probability of accepting class zero. Its randomized balanced error equals half the two-class half-hinge surrogate sum; both names are retained to explain the original objective's normalization, not as two independent pieces of evidence. Deterministic error uses a 0.5 threshold and resolves ties to class zero. AUC uses the class-one score. Predictive log loss is additionally computed with probabilities clipped to `[1e-12,1-1e-12]`; this head was optimized for half hinge, so its predictive cross entropy is an evaluation metric, not an achieved logistic training optimum.

The **common RBF-SVM head** uses C=1 and a median bandwidth fitted only on the downstream-training split. Its balanced error and AUC test whether conclusions depend on our envelope head. It is not used to retune selected supports.

Population stress tests use deterministic, paired class translations in standardized coordinates, always using the same direction for all methods. A separate shifted-source-atom calculation has a known empirical Wasserstein coupling and checks the fitted certificate. The population stress maximum is not mislabeled as a proof of the worst possible population distribution.

## Verification and reproducibility records

`code/tests/test_experiments_new.py` checks data-split disjointness, development-only transformations, stable semantic feature indexing under shuffling, scalar-kernel/variance coefficients, trust-region global solutions against dense independent searches, the trust-region hard case, direct unbiased-MMD statistics, plus-one p-values, and metric normalization.

Each full run writes a configuration snapshot, per-trial preprocessing and split metadata, selected observed and semantic coordinates, status and wall time, global bounds where applicable, per-replicate permutation results, head certificates, stress metrics, and failure records. Global/greedy optimization traces are separate JSON files. Existing per-trial files are resumed only when their configuration digest matches. Native additive diagnostics are named separately from the common-support MMD comparison. Failure and time-limit rows remain part of the record rather than disappearing from aggregated results.
