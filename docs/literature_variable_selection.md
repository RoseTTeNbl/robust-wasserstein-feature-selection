# What the feature-selection literature actually evaluates

Read on 3 October 2026. This document records the literature before the new experimental protocol and experiment code are written. It is a source audit, not a claim that any of these experiments has already been reproduced. The source PDFs and their supplementary experimental sections were read; representative figures were rendered to check their presentation.

The closest experimental reference is **Variable Selection for Kernel Two-Sample Tests**. The other two papers study useful but different tasks: feature acquisition for several downstream populations, and safe screening before deployment-time adaptation. Their metrics should not be silently combined into a single hypothesis-testing leaderboard.

## 1. Variable Selection for Kernel Two-Sample Tests

Jie Wang, Santanu S. Dey, and Yao Xie. Local version: arXiv:2302.07415v4, dated 20 December 2024. Primary sources: [paper](https://arxiv.org/abs/2302.07415), [author-linked implementation](https://github.com/WalterBabyRudin/MMDVar_Selection). The implementation link is explicit in the [Optimization Online paper version](https://optimization-online.org/wp-content/uploads/2023/02/MMD_1012.pdf). Local PDF: `external/papers/wang_variable_selection_local.pdf`. The supplementary EC.1--EC.5 material is mathematical/algorithmic; the experimental settings are principally in Sections 6 and 7.

### The quantity being selected

The paper selects a sparse **weighted kernel**, maximizing empirical MMD squared minus a variance penalty. Its coefficient vector satisfies `||z||_2 = 1` and `||z||_0 <= d`. Our binary coordinate mask is a different parameterization. A comparison can share the support budget, but should state that Wang's method also optimizes weights on that support.

An easily missed distinction is that the paper's **linear kernel** is linear in these weights:

\[
K_z(x,y)=\sum_{j=1}^D z_j k_j(x_j,y_j),
\qquad k_j(s,t)=\exp[-(s-t)^2/(2b_j^2)].
\]

It is not the raw dot-product kernel `x^T y`. This additive kernel can detect marginal variance and shape changes, but it cannot detect a change purely in dependence when all marginals remain equal. The quadratic kernel squares the weighted sum of coordinate kernels, thereby including coordinate interactions. The Gaussian kernel uses a sparse learned projection. A univariate-MMD ranking is a useful screening baseline, but is not a reproduction of the paper's variance-regularized sparse optimization.

For a faithful additive-kernel reproduction, Eq. (13) supplies one vector of coordinate kernel contrasts `h_ij`. Their mean gives the linear objective coefficient; the variance estimate gives the quadratic coefficient. The resulting sparse trust-region subproblem must be optimized over sparse unit-norm weights. Substituting an arbitrary dot-product kernel or merely selecting the largest marginal MMD values changes the method.

### Synthetic mechanisms (Section 6)

| Mechanism | Two distributions | What it is intended to test |
|---|---|---|
| Correlated Gaussian mean shift | `P=N(0,Sigma)`, `Q=N(m,Sigma)`, `Sigma_ab=rho^|a-b|`, `m_j=tau/j` for the first `s` coordinates and zero otherwise | Mean signal with correlated background and decreasing signal strength |
| Gaussian covariance shift | Same zero means; multiply the first `s x s` covariance block by `tau>1`, leaving the rest as specified in the paper | Equal-mean differences that a linear classification statistic can miss |
| Gaussian versus Laplace | `P=N(0,I)`; Q's first `s` coordinates are independent zero-mean Laplace variables with standard deviation 0.8; all other coordinates stay Gaussian | Distribution-shape and variance differences beyond mean separation |
| Gaussian versus symmetric Gaussian mixture | `P=N(0,I)`; Q's signal block is `0.5 N(-m,I_s)+0.5 N(m,I_s)`, with `m_j=tau/j`; remaining coordinates stay standard Gaussian | Equal-mean multimodal alternatives and shared latent mixture membership |

Default parameters are `rho=0.5`, `tau=1` for mean shift, `tau=2` for covariance shift, and `tau=2` for the mixture. The Gaussian mixture is a joint mixture on the signal block, not independently drawn mixture labels for each coordinate.

The paper sets coordinate Gaussian bandwidths by median pairwise distance, and chooses the global Gaussian bandwidth and quadratic-kernel constant by median heuristics. In a new honest implementation these quantities must be fitted on the selection/training data, then held fixed for final testing.

### Separation of selection and final hypothesis testing

Algorithm 1 has two independent parts. It first learns the sparse selection/kernel on training samples. It then computes a two-sample statistic on held-out samples and permutes the held-out labels while keeping the learned selection/kernel fixed. Under the null, those held-out labels are exchangeable conditional on the learned rule, which is the basis for Type-I control.

The variance penalty is tuned over `{0.1, 0.5, 1, 2, 5}` using a 50:50 training/validation division and the held-out negative p-value. This tuning must be confined to the development pool; a further independent final testing pool is needed if one reports the final test's calibration. Selection on the final testing labels followed by a fixed-selection permutation distribution would invalidate the stated calibration argument.

The source calls its shuffled-label calibration both bootstrap and permutation. Its Algorithm 1 explicitly shuffles the labels. A new report should use the precise term **permutation test** and state the number of permutations. The text does not give a universally fixed `N_p` for every experiment, so it should not be invented.

### Metrics, sample sizes, and repetitions

The primary statistic is **testing power at nominal Type-I level 0.05**, not predictive accuracy. For the main sample-size sweep, `D=100`, true signal size `s=20`, selected budget `d=20`, and the per-class selection and test sample sizes are both `n`. Figure 2 covers `n=50,60,...,100`; its uncertainty is based on 20 independent trials. For the dimension sweep, `n=50`, `s=d=20` are fixed (Figure 3).

Support recovery uses the paper's own metrics:

\[
\operatorname{FDP}(S)=\frac{|S\setminus S_*|}{|S|},
\qquad
\operatorname{NDP}(S)=\frac{|S_*\setminus S|}{|S_*|}.
\]

These are `1 - precision` and `1 - recall` for a nonempty selected set. Figure 4 also plots their average, with ideal-support dashed lines. It uses `D=100`, true support size 20, and budgets `d=1,...,20`. Covariance and Laplace cases use `n=150`; mean-shift and mixture cases strengthen the signal and use `(tau,n)=(2,200)` and `(4,200)`, respectively. The recovery panel text does not explicitly state a separate repetition count or CI construction.

The optimization study is separate from the statistical study. Figure 1 and Table 1 compare feasible STRS objective values and wall-clock time for normalized truncation, SDP relaxation, and convex integer programming. Parameter triples are `(1000,20,5)`, `(1000,40,10)`, `(1000,60,6)`, `(1000,80,8)`, `(1000,100,10)`; Figure 1 uses 100 independent trials. This is an appropriate precedent for reporting optimization quality and time separately from power and support recovery.

The official repository's checked main Gaussian script has 10 training repetitions and 200 fresh testing repetitions per trained rule, with per-class train/test sizes 50. Those defaults are not identical to the local v4 paper's figure parameters. A fresh test replicate conditional on one selected support is not an independent rerun of the selector. Reproduction reports must record both counts and cannot treat the conditional test replicates as independent training seeds.

The source figures use boxplots for distributions of objective/power and lines for support metrics. The text sometimes describes these as error bars, but it does not define them as 95% confidence intervals. A new report should explicitly distinguish across-seed SD, binomial uncertainty, and confidence intervals instead of attaching the label "CI" to an unspecified source error bar.

### Baselines and real datasets

Section 6.2 compares sparse additive/linear, quadratic, and Gaussian MMD against their full-variable MMD counterparts, sparse logistic regression with an L0 support constraint, and projected Wasserstein selection/testing. Sparse logistic regression is a linear classifier-based statistic, so equal-mean alternatives are deliberately difficult for it. Projected Wasserstein selects a projection direction, which is not automatically a hard coordinate subset unless its sparsity is enforced. Both distinctions belong in a fair reproduction.

MNIST (Section 7.1) is principally an interpretability demonstration: digit pairs `(0,6)`, `(8,9)`, `(3,8)`, `(7,9)`; original images are `28 x 28`, processed with a `9 x 9` convolution to `D=169`; per-class selection sample size 20, test sample size 5, budget 20. Additional 1000-sample visualizations show distributions and witness functions in a 2D embedding. The 2D embedding is explanatory, not a substitute for a calibrated test.

The healthcare experiment (Section 7.2) has 39 variables, 20,771 healthy records and 2,891 sepsis records. Selection uses 20,000 healthy and 2,000 sepsis records. Remaining records supply held-out draws of 100 samples per class; power is estimated over 2,000 testing trials at level 0.05. The top-five variables are visualized. Table 2 reports power 0.835/0.915/0.784/0.771/0.749 for additive, quadratic, Gaussian MMD, logistic regression, and projected Wasserstein respectively. These are **source results**, not results of our implementation; the paper does not give a CI in that table. Repeated draws from a finite held-out reservoir can overlap, so they are not equivalent to 2,000 independent new clinical datasets.

### What can transfer to our research

Power, explicit null calibration, and FDP/NDP are directly relevant. However, our robust classification-risk objective alone does not imply a level-0.05 two-sample test. To report power we must first define a scalar two-sample statistic based on a frozen selected support or trained score and calibrate that statistic on independent test samples. A common downstream RBF-MMD test on every selected support evaluates the quality of the support; the paper's native learned-kernel test evaluates the complete learned testing pipeline. Both can be reported, with different labels.

Our native classifier also has useful per-observation false-positive and false-negative rates, balanced error, AUC, and surrogate loss. A classifier's class-0 false-positive rate at threshold 0.5 is not the same object as the Type-I rejection rate of a two-sample equality test. The two must be reported in separate panels.

With correlated nuisance coordinates, the paper's declared changed-coordinate set is not necessarily the unique Bayes-optimal support for our robust risk. Background variables may help condition out correlated noise. Recovery against the declared signal set should therefore accompany risk/power, not replace them or be called proof of optimal selection.

## 2. Distributionally Robust Feature Selection

Maitreyi Swaroop, Tamar Krishnamurti, and Bryan Wilder. Local source: arXiv:2510.21113v1. Read Section 4 and appendices C, D, and E. Primary sources: [paper](https://arxiv.org/abs/2510.21113), [official code](https://github.com/maitreyiswaroop/dro-feature-selection). Local code already exists at `external/code/dro-feature-selection-main/` and another earlier copy at `external/review_20260928/dro-feature-selection/`.

This paper chooses a shared set of observed features, after which a separate prediction model can be trained for each population. Its objective protects several specified subpopulations; it does not define a Wasserstein class-conditional ambiguity set or a two-sample permutation test. Its architecture is useful for disentangling selection quality from the predictor used during selection.

### Data splitting and metrics

Each population is split 60:40 for feature selection versus downstream learning/evaluation. The remaining 40% is split 80:20, giving overall proportions **60% selection, 32% downstream training, 8% final test**. Every selector is evaluated using the same downstream model pipeline on its selected features. The paper uses random forests (100 trees) and a one-hidden-layer MLP (100 units, maximum 1000 epochs with early stopping) rather than evaluating each selector solely with its own in-training predictor.

Regression reports per-population MSE and, on real data, R2. Classification reports per-population log loss and accuracy. Appendix tables report means plus/minus across-run standard deviations, not 95% confidence intervals. Synthetic experiments use three seeds; ACS uses five; Adult uses three due to logistic-regression instability. A robust-performance claim is supported by showing each population, so a large group cannot hide a small group's poor performance in a pooled average.

### Mechanisms and real data

Synthetic linear data have 15 variables and three populations: populations A and B reverse the signs of the coefficients on variables 0--4, while B adds variables 5--6 and C depends on variables 7--10. Budgets 5 and 10 and total sample size 36,000 are used. Appendix E.1 adds nuisance variables to dimension 50, retaining those budgets. This is a cancellation example: pooled linear importance can hide variables that are strongly useful in separate populations.

The nonlinear experiment has 50 variables and four populations: squared effects, products `X5*X6`, `sin(2X7)`, heteroscedastic noise driven by `X3,X4`, and scaled t3 noise. Budget is 8 and total sample size 44,000. The printed population percentages in the main-text v1 are 40/35/25/15, which sum to 115%; the text also labels this nonlinear experiment "Linear model". Those inconsistencies should be resolved against official data generators before claiming literal replication, not copied into a new specification.

Appendix E.2 gives a sparse linear 50-variable setting: each of three populations depends on a different scattered support. Neighboring features follow `X_(i+1)=0.3 X_i+0.7 eta`, and most variables are noise. Budgets 5 and 10 and 36,000 total records are used.

Adult Income uses a 10% subsample after combining the original files; its actual populations are sex groups, although an earlier sentence refers to age. The group-defining feature is removed. One-hot encoding yields 44 variables and budget 5; a random forest classifier is trained separately by population. ACS uses 2018 data from California, Florida, and New York, subsamples 5% of each, preselects 18 pooled-random-forest-important variables, and uses budget 7 for income regression. The text says its preliminary forest sees the entire pooled dataset; in a new leakage-free protocol, any such supervised pre-screening must use the development split only.

### Baselines and implementation

The paper compares pooled Lasso/logistic regression coefficient rankings, XGBoost importance rankings, group-DRO versions of these two methods using exponentiated population-weight updates, and an embedded MLP with a learnable sparse mask. All output a budgeted feature set for a common downstream predictor.

The proposed selector injects coordinate noise, estimates conditional means with a kernel/nearest-neighbor construction, and optimizes population variance losses. Synthetic settings use Adam and cosine scheduling, nearest-neighbor counts 1000, and 10 or 50 Monte Carlo draws. Appendix C gives the baselines, while D and E specify epochs, learning rates, and budgets. The official implementation is available locally and should be isolated from our solver environment because it brings PyTorch/XGBoost dependencies.

### What can transfer

The independent selection/training/test separation and common downstream evaluator transfer directly. Per-environment reporting and a worst-environment summary are useful when we explicitly define deployment perturbations. The source's group-DRO feature selector is a justified baseline only if our comparison supplies explicit groups and states how its task has been adapted. Treating the two class labels themselves as "populations" in a group-DRO predictor can create one-class groups and does not reproduce the paper's setting. An adapted implementation should be called an adaptation, with its changes and validation recorded.

## 3. Safe Distributionally Robust Feature Selection under Covariate Shift

Hiroyuki Hanada, Satoshi Akahane, Noriaki Hashimoto, Shion Takeno, and Ichiro Takeuchi. Local source: arXiv:2603.16062v1. Read Section 4 and Appendix B; Appendix A.7 supplies the algorithm. Primary sources: [paper](https://arxiv.org/abs/2603.16062), [full HTML](https://arxiv.org/html/2603.16062v1). No author-linked official repository was found in the supplied PDF/HTML or the targeted search on the read date.

This method certifies that a variable is inactive in **every** weighted L1-regularized model under bounded covariate importance-weight changes. It retains a superset of the union of all possible optimal sparse supports. That is a screening task, not selection of the best fixed-size subset for robust testing.

### Experimental target

The paper explicitly does not evaluate prediction performance: the claim is that certified deletion preserves the sparse learning optimum within its setting. The metric is the **fraction of safely removed variables** versus distributional uncertainty and regularization. There are no power, Type-I, AUC, support F1, or comparative fixed-budget-selector tables in this experimental section.

Uncertainty is described by `w_i in [1-delta,1+delta]` with `sum_i w_i=n`. The reported horizontal axis is total weight variation `V=max_w ||w-1||_1`, swept over 0 and log-spaced values from `1e-5` to 1. Appendix B.2 shows exactly `V=n*delta` for even n and `(n-1)*delta` for odd n. Regularization takes `lambda_max * {1,10^-0.5,10^-1,10^-1.5,10^-2}`. Larger uncertainty typically permits fewer certified removals.

The eight regression datasets are cpusmall, housing, Facebook comments, triazines, online news, communities-and-crime, superconductivity, and blog feedback. The eight classification datasets are Australian, ionosphere, sonar, splice, mushrooms, a1a, madelon, and colon cancer. Table 1 gives post-processing dimensions ranging from 12 to 2000 and sample sizes from 62 to 52,397.

Appendix B removes constant/uninformative variables, standardizes features to zero mean and sample SD one, removes UCI missing-value columns, and uses only the training file when LIBSVM provides an existing split. Logistic models use LIBLINEAR, regression uses scikit-learn Lasso, and screening itself uses NumPy matrix operations. No repeated-random-split or uncertainty-band protocol is specified for the displayed deterministic screening curves.

### What can transfer

The paper provides a useful distinction between a **safety certificate** and empirical downstream performance. We can emulate the clarity of its uncertainty/regularization sweep and specify exactly what our global optimum certificate guarantees. But forcing its retained superset down to exactly `d` variables destroys its no-false-elimination guarantee. It should therefore not be introduced as an unchanged budget-matched baseline. A separate optional screening experiment or an explicitly labeled screening-then-ranking adaptation is possible.

## 4. What the figures teach about readable reporting

Rendered pages checked: Wang pp. 19 and 21 (power and support figures), Swaroop p. 10 (real-data comparison), Hanada pp. 6--7 (dataset table and screening curves).

Wang separates mechanisms into panels and uses a common legend, consistent metric axes, and a dashed ideal-support reference. Swaroop pairs a loss metric with a familiar prediction metric and exposes each population in its own row; its appendix uses mean/SD tables with explicit up/down arrows and bold best values. Hanada sweeps uncertainty on a log scale and labels sample size, feature count, and regularization in each dataset panel. These are useful conventions. Tiny repeated legends, dense four-row figure grids, and long equations in axis labels are not aspects that should be copied when the goal is a readable group-meeting report.

A readable new report should first explain the question answered by a figure and the direction of improvement. A compact summary panel can show the main effect with consistent method colors and uncertainty intervals; a separate table can provide exact values and repetition counts. Algorithmic time/calls/bounds, statistical calibration/power, support quality, and per-observation deployment risk answer different questions and should have distinct captions.

## 5. Measurement definitions to preserve during protocol design

These are implications of the source audit, not results or an already executed experiment plan:

1. **Testing power:** fraction of independent final two-sample datasets rejecting equality under a specified alternative at alpha 0.05. **Type-I error:** the corresponding rejection fraction under equality. Report binomial uncertainty for genuinely independent test datasets; conditional replicates under one training seed require clustered or hierarchical uncertainty when making across-training claims.
2. **Selection quality:** FDP/NDP reproduce Wang's metrics. Precision, recall, F1, and support Jaccard can be added with their denominators stated. Distinguish Jaccard against ground truth from pairwise Jaccard stability across training seeds. A null case has no true support and must use a defined convention or show the metric as not applicable.
3. **Prediction quality:** per-class error and balanced error, ROC AUC, and average surrogate loss on fresh individual observations. For logistic, the original objective sums two class-average losses; a balanced average log loss is half this sum. Thresholds and loss normalization must be consistent across methods.
4. **Robustness:** a maximum over a finite collection of stress distributions is an empirical stress test. It is not a proof of the worst-case Wasserstein risk. Controlled transport shifts with a known coupling can establish inclusion in the intended ball; arbitrary noise magnitudes do not automatically do so.
5. **Computation:** actual wall time, continuous-oracle calls, master time, solved-support fraction, and lower/upper certificates. Every baseline must receive the same input split, feature budget, scaling, and relevant time policy. Statistical conclusions need more than an optimization gap.

Only the first paper directly defines a selected-feature two-sample testing protocol. The latter two contribute selection/evaluation separation and robust deployment/safety perspectives. This distinction should remain visible in the eventual experiment report and baseline naming.
