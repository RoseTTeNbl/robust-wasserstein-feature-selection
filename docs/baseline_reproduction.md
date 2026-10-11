# What was compared, and what was reproduced

Updated 11 October 2026 after rereading the local *Variable Selection for Kernel Two-Sample Tests* (Wang, Dey and Xie, arXiv:2302.07415v4). The experiment sections are PDF pp. 16–24; the learned kernels and test procedure are on pp. 4–6. This document explains the existing records. It does not announce new experiments or change the frozen protocol.

There are two comparisons. The statistical study asks whether the selected coordinates are useful on new data. Its six practical selectors are listed below. The revised-search validation asks how efficiently different solvers minimize the same robust objective. Its master-only, original GP and retained-fraction searches are optimization controls, not additional statistical selectors. Historical statistical results describe their recorded search version; they are not rerun results for the current algorithm.

## The six practical selectors

All six see the same selection sample and use the same feature budget. In the recorded synthetic study all six select three coordinates. Their supports are evaluated by the same independent robust classifier, RBF SVM and joint RBF two-sample test.

| Display name | How it chooses coordinates | Why it is included |
|---|---|---|
| Global robust risk (`risk`) | Minimize the class-conditional Wasserstein robust testing risk by global cuts | Our selection objective; recorded optima or time-limited incumbents |
| Robust greedy (`greedy`) | Add the coordinate with the smallest fully refitted robust risk | Isolate the value of global search under the same objective |
| Marginal MMD screening (`mmd`) | Rank one-coordinate distribution discrepancies | A simple distribution-based selector |
| Wang additive MMD (`wang`) | Optimize sparse, unit-norm coordinate-kernel weights using MMD minus estimated variance | Independent implementation of one mathematical criterion from the reference paper |
| L1 logistic ranking (`l1`) | Fit L1 logistic regression and keep the largest absolute coefficients | A familiar classifier-based selector; not the paper's L0 logistic method |
| Random (`random`) | Draw a uniform budget-sized subset | A low-information control |

Known signal coordinates (`signal`) and all coordinates (`all`) are diagnostic references. The first is unavailable in practice; the second observes more variables. They are kept in the data but are not budget-matched practical competitors.

## Correspondence with the reference paper

The paper compares learned tests as well as their selected variables. Our main comparison holds the evaluator fixed after selection. That is a valid support-quality comparison, but it is a different experiment.

| Reference experiment | Existing records | Accurate interpretation |
|---|---|---|
| Sparse additive, quadratic and Gaussian MMD, plus full-variable MMD versions (pp. 18–20) | Additive MMD criterion only; marginal screening and a common joint RBF evaluator | The common RBF evaluator is not the missing learned Gaussian-MMD selector |
| L0-constrained sparse logistic regression and its linear testing statistic (pp. 6, 18–19) | L1 logistic coefficients, top-budget ranking, then common evaluators | A baseline-family adaptation, not reproduction of the L0 method or its native test |
| Projected Wasserstein test (pp. 18–21) | Not implemented | No result is reported for this baseline |
| Variance penalty chosen from `{0.1, 0.5, 1, 2, 5}` by 50:50 training/validation p-values (p. 6) | Fixed additive penalty 0.1; fixed L1 and SVM `C=1` | A fixed-parameter study, not the paper's tuning protocol |
| Train a weighted kernel, then test with that same kernel on independent samples (p. 6) | Main power uses a joint RBF test on the selected support; additive weights have a separate native diagnostic | Support-only and learned-test results must retain different labels |
| Four source mechanisms, sample-size/dimension sweeps and budget sweep (pp. 16–21) | Four adapted mechanisms at `D=12`, signal size and budget 3, with different sample counts | An adapted small-sample study, not a replay of source figures |
| MNIST and sepsis testing (pp. 22–24) | WDBC and Ionosphere classifier evaluation | Additional real-data prediction experiments, not source real-data replication |

The existing baseline coverage is therefore precise: **only the Wang additive-MMD mathematical criterion was independently implemented**. Learned quadratic/Gaussian MMD, L0 sparse logistic regression and projected Wasserstein were not reproduced. Adding any of these methods, the source tuning grid or source data mechanisms requires new, separately recorded experiments; changing descriptions cannot create those results.

## What the two evaluation tracks measure

The main equality test discards a selector's fitted weights and uses its chosen support. It fits a joint RBF bandwidth on pooled selection data, freezes it, and computes an unbiased squared-MMD statistic on independent test samples. All methods in a trial receive the same batches and permutation seeds. There are 199 fixed-size label permutations, with

\[
p=\frac{1+\#\{T_{\mathrm{perm}}\ge T_{\mathrm{obs}}-10^{-14}\}}{200},
\qquad \text{reject if }p\le0.05.
\]

Alternative rejection measures power; separate null batches measure calibration. Testing repeats are averaged within a training seed before uncertainty is computed across seeds.

The additive-MMD native diagnostic retains its fitted weights and coordinate bandwidths. It answers how that learned additive statistic performs, whereas the common test answers how useful its support is. Neither statistic is relabeled as the other. The paper's native procedure also differs from our diagnostic in its finite-sample statistic and calibration implementation, detailed below.

Individual prediction is a separate evaluation. Each support gets a robust classifier and an RBF SVM fitted on a downstream sample independent of selection. Final observations are independent of both. The robust classifier returns class-zero probability: randomized balanced error is half its two-class half-hinge loss sum, while hard error thresholds at 0.5 with ties assigned to class zero. AUC uses class-one scores. Balanced predictive cross-entropy clips each true-class probability below at `1e-12`; it is an evaluation metric, not an optimized logistic training objective. The SVM uses `C=1` and a downstream-only median bandwidth.

The population shift experiment translates held-out classes in opposite directions along one common, downstream-estimated direction. Its maximum is over a declared finite shift set. A separate translation of source atoms has a known admissible transport coupling and checks the fitted empirical certificate. Neither outcome is called a population coverage guarantee.

## Fixed implementation details

Marginal screening uses unbiased paired coordinate contrasts with selection-only median squared cross-class distances and index ties. L1 ranking uses `LogisticRegression(solver="liblinear", l1_ratio=1, C=1, max_iter=2000)`; coefficient ties use selection variance and then index. Random selection has a reserved RNG stream. Robust greedy fully reoptimizes each available addition; if its common time budget expires, it keeps the last complete stage and fills the remaining budget deterministically from marginal screening, recording that status.

The additive selector uses penalty 0.1 and the same selection-only coordinate bandwidth rule. For `D<=12`, it enumerates supports and globally fits weights on each unit sphere. For larger dimensions it uses forward additions and at most ten improving swaps, refitting weights each time; records are labeled `approximate_forward_swap`. This changes the paper's optimization implementation without changing the declared additive objective. Time-limited or approximate results are not presented as globally certified selections.

## Technical checks retained for inspection

The source PDF is `D:/论文/Generative/jw/Jie_Wang/variable selection for kernel two-sample tests.pdf`. The inspected author code is commit `031f740e9ca6672751145b16de62cfaf731e98de`; read-only files remain under `external/code/MMDVar_Selection/`. No third-party implementation is imported or republished.

The paper's additive kernel is linear in weights, not a raw dot product:

\[
K_w(x,y)=\sum_jw_j\exp[-(x_j-y_j)^2/(2b_j^2)],
\qquad \|w\|_2=1,\quad \|w\|_0\le d.
\]

For equal per-class size n, let

\[
h_{i\ell,j}=k_j(x_i,x_\ell)+k_j(y_i,y_\ell)
-k_j(x_i,y_\ell)-k_j(y_i,x_\ell),
\quad
 t=\frac{\sum_{i\ne\ell}h_{i\ell}}{n(n-1)}.
\]

With row averages \(r_i=n^{-1}\sum_\ell h_{i\ell}\) and their mean \(\bar r\), the implemented variance is

\[
\widehat\Sigma=\frac4n\sum_i(r_i-\bar r)(r_i-\bar r)^T,
\qquad
\max_{\|w\|_2=1,\|w\|_0\le d}
 t^Tw-0.1w^T\widehat\Sigma w.
\]

`wang_coefficients` implements the negative variance penalty stated in paper Eq. (11). The inspected revision coefficient function returns a positive penalty followed by a diagonal shift, and its inspected maximizer does not negate that penalty. That source combination is not silently used as the primary baseline. Independent coefficient tests preserve both expressions for comparison. This is a source-code discrepancy, not evidence of improved experimental performance.

On a fixed support, the weight problem is \(\max_{\|u\|_2=1}u^TA_{SS}u+t_S^Tu\). An eigenbasis and secular norm equation give a global solution, including the hard case with a zero linear component in the top eigenspace. Smaller effective supports are covered by zero coefficients inside size-d supports. Signed weights are retained as permitted by the printed coefficient domain; their native statistic is not called positive-definite-kernel MMD unless the weights are nonnegative.

The paper's estimator excludes paired cross-diagonal terms (Eq. (1), p. 4). Our common and native evaluation use the usual unbiased two-sample form with all cross-pairs. Both target the same population expression but need not give identical finite-sample statistics. The author revision additionally refits cross-class bandwidths using final labels; our testing bandwidths are frozen from selection. The 199-permutation plus-one rule above is our declared calibration, not an invented universal value for the source paper's unspecified permutation count.

`test_experiments_new.py` checks split independence, frozen transformations, stable feature identities, scalar-kernel and variance coefficients, global fixed-support weight solutions, the hard case, direct MMD statistics, permutation p-values and normalization. Records retain selected supports, parameters, seeds, statuses, charged timings, head certificates and per-batch tests. Historical records and their original configuration hashes are preserved.
