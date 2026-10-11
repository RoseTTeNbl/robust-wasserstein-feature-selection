# What the feature-selection literature actually evaluates

Initial reading on 3 October 2026 preceded the frozen experiment protocol and implementation. The Wang experiment comparison was reread and clarified on 11 October 2026. This document records source evidence and the scope of our adaptations. It is a source audit, not a claim that any of these experiments has already been reproduced. The source PDFs and their supplementary experimental sections were read; representative figures were rendered to check their presentation.

The closest experimental reference is **Variable Selection for Kernel Two-Sample Tests**. The other two papers study useful but different tasks: feature acquisition for several downstream populations, and safe screening before deployment-time adaptation. Their metrics should not be silently combined into a single hypothesis-testing leaderboard.

## 1. Variable Selection for Kernel Two-Sample Tests

Jie Wang, Santanu S. Dey and Yao Xie; local arXiv:2302.07415v4, dated 20 December 2024. The current original is `D:/论文/Generative/jw/Jie_Wang/variable selection for kernel two-sample tests.pdf` (42 PDF pages; printed page numbers agree). Reread on 11 October 2026: the full experiment sections on pp. 16–24, kernel definitions on p. 4, and training, tuning and testing on pp. 5–6. Existing author-source checks are described in `baseline_reproduction.md`.

The experimental question is: **can learning a small set of variables yield a more powerful two-sample test than testing with all variables?** The paper measures rejection under a specified alternative at nominal level 0.05, then separately checks which changed coordinates were found. It is not chiefly a classifier-accuracy study.

The learned object includes both a support and weights. The paper's additive kernel sums one-dimensional Gaussian kernels with learned sparse weights. Its quadratic kernel squares that sum plus a constant, allowing coordinate interactions. Its Gaussian kernel learns sparse coordinate scales inside the squared distance. These are three different optimized selectors; a common RBF test on an already selected support does not implement all three.

### Source experiment versus our recorded study

| Item | Source paper | Existing records | What the comparison supports |
|---|---|---|---|
| Main selectors | Sparse additive, quadratic and Gaussian MMD; L0 sparse logistic; projected Wasserstein (pp. 18–20) | Robust global/greedy, marginal MMD, Wang additive MMD, L1 ranking, random | Only the additive-MMD mathematical criterion is independently implemented from this paper |
| Selection objective | Empirical squared MMD minus a variance penalty (p. 5) | Same additive criterion for Wang; robust risk for ours | Objectives differ; a common evaluator isolates usefulness of their supports |
| Testing statistic | Learned kernel retained on independent test data (p. 6) | Main power uses the same joint RBF test on every support; additive weights have a separate native diagnostic | Main support-quality results and native learned-test results must be separate |
| Hyperparameters | Penalty grid `{0.1,0.5,1,2,5}`, 50:50 training/validation, choose by validation p-value (p. 6) | Additive penalty fixed at 0.1; selection-only bandwidths | Existing results are not a cross-validated reproduction |
| Power sweeps | `D=100`, true support and budget 20, `n_train=n_test=50,60,...,100`; 20 trials (p. 19). Dimension sweep `D=50,60,...,100`, `n=50` (pp. 19–20) | `D=12`, true support and budget 3; 20 selection samples/class, 40 samples/class/test batch; 30 training seeds and 20 batches/condition/seed | An adapted small-sample experiment, not the source's sample-size or dimension sweep |
| Support recovery | `D=100`, true support 20, budgets 1–20; FDP, NDP and their average (pp. 20–21) | Fixed true support and budget 3; stored FDP/NDP, precision/recall/F1 | FDP and NDP coincide in the six recorded equal-size synthetic comparisons; reproducing the budget curve requires new runs |
| Prediction data | Two-sample testing is primary | An extra independent fitting sample of 20/class and 1,000/class classification observations | Classifier error and AUC are additional measurements, not substitutes for power |
| Real data | MNIST interpretation and sepsis testing (pp. 22–24) | WDBC and Ionosphere prediction | Different tasks and datasets; no source clinical or image replication is claimed |

The six practical selectors share a well-defined acquisition budget. Known-signal and all-feature supports are diagnostics; all features require D observations rather than d. The paper's L0 logistic method is not our L1 fit followed by coefficient ranking. Its native linear statistic can miss equal-mean changes, whereas our common RBF evaluator can detect nonlinear differences on whatever coordinates L1 happened to select. Thus the two procedures cannot share a reproduction label.

Projected Wasserstein learns a projection and need not observe only d coordinates; paper p. 3 notes the lack of sparsity in those projection methods. An unchanged dense-projection test would be a full-input testing comparator, not a budget-matched coordinate selector. No result for it, learned quadratic MMD or learned Gaussian MMD exists in the frozen study.

### The four data mechanisms are adaptations

In the source, s denotes the true changed-coordinate count and j indexes the changed coordinates. The unmodified coordinates follow the source's stated background distribution.

| Source mechanism (p. 16) | Our existing generator | Degree of alignment |
|---|---|---|
| Gaussian mean shift with covariance `Sigma_ij=0.5^|i-j|`; changed means `1/j` | Independent Gaussian coordinates; constant mean shift 0.9 on three shuffled signals | Same kind of mean difference, different background correlation and signal-strength pattern |
| Gaussian covariance shift: same correlated background, first signal covariance block multiplied by 2 | Independent coordinates; signal SD multiplied by 1.8 | Both change covariance, but ours changes only marginal variances and uses different strength |
| Gaussian versus independent zero-mean Laplace signals with SD 0.8 | Identical standard-normal marginals; signal block gains equicorrelation 0.65 | A different experiment: pure dependence change replaces the source's marginal shape change |
| Gaussian versus a shared-component symmetric mixture, signal means `+/-2/j` | Shared-sign mixture with constant shifts `+/-1.5` | Same joint-mixture idea, different component means |

In particular, the dependence experiment must not be called Gaussian-versus-Laplace. It is an additional challenge for joint selection: unchanged marginals contain no population signal for an additive marginal test. This structural limitation explains why marginal/additive comparators may perform poorly there; it does not establish that the same comparison was reported in the source.

Our preprocessing also differs: pooled selection means and sample SDs standardize the reference study before transport costs and downstream evaluation. The paper's simulated model definitions do not specify this extra standardization. Reproducing its precise data distributions would require generating new records, not renaming the existing cases.

### Which metrics transfer directly

Power is the fraction of independent alternative test batches rejecting equality. Null rejection is measured on separate equal-distribution batches. Both are reported at the same level 0.05. A classifier's class-zero error at a fixed decision threshold is a different event and has a different denominator.

For chosen support S and declared signal support S*, the paper's recovery measures are

\[
\mathrm{FDP}(S)=\frac{|S\setminus S_*|}{|S|},
\qquad
\mathrm{NDP}(S)=\frac{|S_*\setminus S|}{|S_*|}.
\]

For nonempty selected sets these equal one minus precision and one minus recall. In our six synthetic selectors, both sets have size three, so precision, recall and F1 agree, and FDP equals NDP. Showing recall is therefore one readable way to summarize the existing recovery evidence. The source budget sweep breaks that equality and reveals the false-selection/missed-signal tradeoff. A new analogous sweep could use budgets 1–3 with the fixed three-signal generator; it would be a new experiment.

The declared changed-coordinate set need not be the unique optimal support for a different robust-risk objective, especially with correlated backgrounds. Recovery should be read alongside risk and held-out power, not as a definition of robust optimality.

### A minimal, honest alignment

The present report can improve without another run: use source-relevant power, null rejection and recovery terms; explain the six actual selectors; show the all-feature common RBF diagnostic beside the selected-support results with its larger budget labeled; separate native additive performance from the common evaluator; and state the differences above. All these quantities are already recorded.

Literal data-mechanism replication, tuning the source penalty grid, reproducing a sample-size/budget curve, and adding learned quadratic/Gaussian MMD or L0 logistic require new experiments and a new declared protocol. They cannot be supplied by relabeling old records. A small future supplement could keep a manageable D and d while using the source's distributions and complete selector definitions, openly labeling reduced dimensions rather than pretending to replay its figures. The frozen study and its failures would remain intact.

### Source details and limits retained for verification

- Paper p. 4 defines the paired U-statistic by excluding cross-diagonal pairs. Our common and native diagnostics use the usual unbiased two-sample form with all cross-pairs. Algorithm 1 on p. 6 shuffles held-out labels; the paper calls this bootstrap but does not give one universally fixed permutation count. Our declared 199-permutation plus-one rule and tie tolerance are implementation choices.
- The source learns on training data and tests independently. Its penalty selection must also stay inside development data. Our additional independent classifier-fitting split comes from a different evaluation goal; it is not stated as Wang's own protocol.
- The source's no-selection curves are called standard MMD without variable selection (p. 18), but the PDF does not specify whether their additive weights are equal or reoptimized without a cardinality limit. The available `Optimization/Standard_MMD.py` contains an untrained raw-dot-product linear test; the inspected drivers do not establish that function's exact correspondence to the plotted no-selection branch. Setting `budget=D` and continuing to optimize additive weights would be a new ablation, not a verified source replication.
- Support recovery uses stronger source parameters: covariance/Laplace `n=150`; mean `(tau,n)=(2,200)`; mixture `(tau,n)=(4,200)` (p. 20). Figure 4 has mechanism rows and metric columns. Its caption's contrary description of the columns should not be copied. No separate recovery repetition count or CI rule is specified.
- The optimization comparison is a separate sparse-weight problem: truncation, SDP and convex integer programming, `n=1000`, five `(D,d)` pairs from `(20,5)` to `(100,10)`; objective distributions use 100 trials (pp. 17–18). These are not solvers for our robust inner objective. Source runtime tables must not be pooled with ours.
- MNIST uses processed dimension 169 and budget 20 (pp. 22–23). The main text states training/testing sizes 20/5, but Figure 5 includes row-specific 40/30 and 20/30 settings. Those panels are visual interpretation, not one uniform benchmark. Sepsis uses 39 variables, 20,000/2,000 healthy/sepsis training records, then 100/class held-out testing repeated 2,000 times (pp. 23–24). Overlapping reservoir draws are not 2,000 independently trained selectors.
- Author-script defaults (10 trained rules and 200 conditional test batches) differ from the paper's 20-trial sample-size plots. Source boxplots/error bars are not explicitly defined as 95% confidence intervals. Our seed-bootstrap intervals are a stated reporting choice. Technical coefficient-sign and final-bandwidth discrepancies are retained at the end of `baseline_reproduction.md`, rather than leading the experimental explanation.

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
