# What the robust-testing papers actually evaluate

Read on 3 October 2026. This is a source record for designing the new experiment suite, not a report of experiments run in this project. The local PDFs were read first; the experiment pages and figures were also rendered and inspected. Page numbers below are one-based PDF page numbers, which coincide with the printed pages for these versions. Absence of a detail means that it was not stated in the inspected paper; it is not filled in from an earlier project design.

The main distinction is between **a two-class robust test** and **a feature selector evaluated through a separate predictor**. The first asks whether an individual observation comes from one of two uncertain distributions. The second chooses variables that predict an outcome well in several observed populations. These tasks share useful experimental controls, but their risk values and robustness guarantees are different.

## 1. Sources and the relevant parts

| Local source | Verified paper | Parts read for experiments |
|---|---|---|
| `external/papers/xie2021.pdf` | Liyan Xie, Rui Gao, Yao Xie, *Robust Hypothesis Testing with Wasserstein Uncertainty Sets*, arXiv:2105.14348v1, 29 May 2021 | Sections 3.4–3.5, pp. 12–14 (deployment and batch test); Section 5, pp. 19–22 (all experiments); Section 4, pp. 14–18 (what radius theory does and does not imply). The remainder of the PDF is proofs, not an additional experiment appendix. |
| `external/papers/gao2018.pdf` | Rui Gao, Liyan Xie, Yao Xie, Huan Xu, *Robust Hypothesis Testing Using Wasserstein Uncertainty Sets*, arXiv:1805.10611v1, 27 May 2018 | Sections 2–3, pp. 2–4 (risk and loss normalization); Section 5, pp. 8–9 (the experiment); pp. 12–15 are proof appendices. |
| `external/papers/swaroop2025.pdf` | Maitreyi Swaroop, Tamar Krishnamurti, Bryan Wilder, *Distributionally Robust Feature Selection*, NeurIPS 2025 / arXiv:2510.21113v1, 24 October 2025 | Section 4, pp. 6–10; Appendix C, pp. 14–17 (baselines); Appendix D, pp. 17–20 (parameters, repetition, complete tables); Appendix E, pp. 20–23 (additional synthetic experiments). |

Primary links: [Xie et al.](https://arxiv.org/abs/2105.14348), [Gao et al.](https://arxiv.org/abs/1805.10611), [Swaroop et al.](https://arxiv.org/abs/2510.21113). The names, authors, and version dates were checked against both local first pages and the arXiv records.

## 2. Xie, Gao, and Xie (2021): small-sample testing

### 2.1 Gaussian-mixture experiment

Section 5.1, pp. 19–20, Table 1 uses two 100-dimensional mixtures:

\[
P_0=\tfrac12\mathcal N(0.4e,I_{100})+\tfrac12\mathcal N(-0.4e,I_{100}),\qquad
P_1=\tfrac12\mathcal N(0.4f,I_{100})+\tfrac12\mathcal N(-0.4f,I_{100}),
\]

where every entry of \(e\) is one and \(f\) has 50 ones followed by 50 minus ones. Each class supplies 10 training observations; each class supplies 1,000 new test observations. The table is averaged over 500 trials. The radius and kernel bandwidth are chosen by cross-validation.

The metric is the average misclassification rate, shown for one observation and for batches of size \(m=2,\ldots,10\). Competitors are a fitted Gaussian mixture model, logistic regression, RBF-kernel SVM, and a three-layer perceptron with two hidden layers of 50 nodes. Competitors use majority decisions for batch observations. Section 3.5 defines the proposed batch rule by averaging the single-observation randomized acceptance probabilities and thresholding at one half; this is more precise than saying that all methods average classifier scores.

The paper does **not** specify the cross-validation fold count, candidate radii, candidate bandwidths, neural optimization settings, GMM covariance restrictions, SVM parameter grid, or random seeds. It reports means rather than standard deviations or confidence intervals. The training/test draws are described as new samples, but no separate calibration sample size is stated. Thus the printed table is not a fully specified exact replay recipe.

This is a useful difficult example: both classes have zero means and the same one-dimensional feature marginals. The signal is joint dependence. A univariate mean filter has no population signal here. For variable selection, a reduced-dimensional, sparsified analogue is a reasonable **adaptation**, provided its complete generator and its departure from the original 100-dimensional experiment are documented. It should accompany a simple mean-shift generator, rather than replace every test with a setting tailored to nonlinear methods.

### 2.2 MNIST experiment

Section 5.2, p. 20, Table 2 uses MNIST, described as 70,000 images. Five images per class are used for training; radii are selected by cross-validation. Test images from the same class are grouped into batches of size 1–10. The text describes 500 randomly selected batches and the caption describes averages over 500 trials. Competitors are logistic regression and SVM. The metric is average misclassification rate.

The paper does **not** state the digit pair, whether trials redraw the training examples or only the batches, the exact train/test index policy, normalization, or the CV grid. A two-digit evaluation in our project must explicitly choose the digits and enforce disjoint training, calibration, and test indices. It cannot be advertised as an exact reproduction of Table 2 without resolving those details. A built-in smaller digits dataset is another permissible adaptation, but must not be called MNIST.

### 2.3 WISDM change detection

Section 5.3, pp. 21–22, Figure 4 uses accelerometer activity sequences from WISDM. The paper describes 225 users. It extracts 360 sequences of length 100, each with one activity transition; the transitions are walking to jogging and jogging to walking. For each candidate time, the method uses two windows of length \(w=10\) around that time, solves for the least-favorable distributions, and accumulates their probability difference with a CUSUM-style recursion. The same uncertainty radius is used for both windows and is selected by cross-validation.

The competitor is a Hotelling \(T^2\) statistic, also accumulated through a CUSUM-style recursion. Historical pre-change observations estimate its mean and covariance. Figure 4 plots expected detection delay against Type-I error, with Type-I error levels spanning roughly 0.05–0.35. This compares speed at a comparable false-alarm cost. There are no uncertainty bands. The paper does not give a separate no-change calibration split, threshold grid, censoring policy for undetected changes, or user-disjoint splitting rule.

Sequential detection is a distinct application: expected detection delay is not a valid primary metric for the current single-observation variable-selection objective. It becomes relevant only if the project adds an explicitly defined sequence task and false-alarm protocol.

### 2.4 Deployment and radius selection

Sections 3.3–3.4, pp. 11–13, construct a randomized optimal test on the empirical support and a permissible interval off that support. The authors smooth the least-favorable distributions with a product kernel and **truncate the resulting test to the permissible interval**. Smoothing without that truncation would not inherit the same guarantee. Our midpoint-envelope deployment and their truncated kernel deployment can differ statistically even when the support objective is identical.

Section 4 gives a profile-based theoretical radius principle. Section 5 actually says that radii are selected by CV. These statements should not be merged into a claim that the experiments implemented the theoretical profile quantile or guaranteed a nominal false-positive rate. In the new suite, radius tuning must use only training/calibration data, and all radius candidates and tie-breaking rules should be saved.

The local archive contains the authors' TeX source at `external/source/xie2021/draft_arXiv_2class_2021.tex` and figures under `external/source/xie2021/images/`. No executable author experiment code was located in these paper/source artifacts. This means that a testing baseline implemented from the finite LP and extension equations is a **paper-based reimplementation**; a source tarball is not evidence of an official code reproduction.

## 3. Gao, Xie, Xie, and Xu (2018): a sequential demonstration

Section 5, pp. 8–9, uses WISDM Actitracker data released in October 2013. Measurements are three-dimensional accelerations, recorded every 50 ms (20 Hz); the paper describes 2,980,765 recorded samples. Figure 3(a) displays a jogging-to-walking segment for person 1679, with a transition at sample index 4589.

The experiments couple robust detectors with a CUSUM-like scheme and compare them with a Hotelling \(T^2\) control chart. The two robust detector choices are

\[
\varphi_{\log}(x)=\tfrac12\log(p_0^*(x)/p_1^*(x)),\qquad
\varphi_{\mathrm{sign}}(x)=\operatorname{sgn}(p_0^*(x)-p_1^*(x)).
\]

The least-favorable distributions come from the authors' finite convex problem. Equal class sample sizes motivate equal radii. Radius selection uses bootstrap samples from the pre-change data: the radius is chosen so that the divergence between the resulting least-favorable distributions is near zero with high probability under no change. The experiment compares average detection delay at controlled Type-I error on 100 sequences, in Figure 3(b). The sign detector is reported to perform better than the log-ratio detector on this task.

The paper does not state the actual window sample size, bootstrap replication count, numerical radius candidates, threshold calibration set, exact sequence extraction indices, or a train/calibration/test split. It reports the 100-sequence mean with no error bands. Appendices A and B are proofs, not a second, more detailed experimental protocol.

The important normalization distinction is on pp. 2–4. The paper starts from the maximum of the two classwise error probabilities and derives a sum of surrogate losses as a safe approximation. Its generating functions obey \(\ell(0)=1\): normalized logistic is \(\log(1+e^v)/\log2\), and hinge is \((1+v)_+\). Our losses are unnormalized logistic and **half** hinge. Consequently, raw certificate values should not be placed in one table as if they had the same units. Classification error is comparable after a common decision rule; certificate comparisons require an explicitly stated rescaling and objective convention.

The local artifacts are `external/source/gao2018/nips_2018_arxiv.tex` and a source PDF. No executable author experiment code was located in these artifacts. Hotelling \(T^2\) is a meaningful sequential baseline, but not a feature selector by itself; using it in the present static task would require a separately specified selection procedure.

## 4. Swaroop, Krishnamurti, and Wilder (2025): separating selection from prediction

### 4.1 The control that is most useful for this project

Section 4, pp. 6–7, makes selection and evaluation separate stages. For each population, 60% of observations form a **feature-selection dataset**. The remaining 40% is split 80:20 into downstream training and downstream test sets. Relative to all observations, the fractions are therefore 60%, 32%, and 8%.

Every selector is evaluated with the same downstream pipeline. The downstream predictor is fitted independently using only the features output by that selector. Synthetic regression tasks use both random forest and MLP; real-data tasks use random forest. This prevents a result from conflating a better selector with a better final model.

The random forest has 100 trees. The scikit-learn MLP has one hidden layer of 100 neurons, at most 1,000 iterations, and internal early stopping. Regression uses MSE, with \(R^2\) additionally reported on ACS. Classification uses log loss, with accuracy additionally reported on Adult. Metrics are shown separately by population. Complete tables report mean ± standard deviation and mark the optimization direction; the best population-specific values are bold.

This paper's uncertainty is **which observed population is worst**, rather than a Wasserstein neighborhood around each of two class distributions. It therefore motivates common-downstream evaluation and population-specific performance, but it does not solve our exact two-class objective.

### 4.2 Synthetic generators and repetitions

Section 4.1, pp. 7–9, and Appendices D–E, pp. 17–23 give these families:

| Family | Data and signal structure | Budget, total observations, repeats |
|---|---|---|
| Linear heterogeneous effects | 15 features, populations A/B/C. Proportions 40/35/25%. A: \(8X_0+6X_1-4X_2+3X_3+2X_4+\epsilon\); B reverses those signs and adds \(8X_5+6X_6\); C uses \(10X_7+8X_8+6X_9-5X_{10}\). Noise SD 0.1. | Budgets 5 and 10; total 36,000; three seeds. Appendix D.1, p. 17. |
| Nonlinear and heterogeneous noise | 50 features and four populations. A/B use \(4X_0+3X_1+X_2^2\); C uses \(2X_0+3X_5X_6+4\sin(2X_7)\); D uses \(3X_0+2X_1\). Noise respectively Gaussian SD .05, heteroscedastic Gaussian \(.1\exp(.5X_3+.3X_4)\eta\), Gaussian SD .1, and \(.2t_3\). | Budget 8; total 44,000; three seeds. Appendix D.2, p. 18. The listed population percentages 40/35/25/15 sum to 115%; do not silently normalize and call that an exact replay. |
| Linear with extra noise variables | The first generator expanded from 15 to 50 dimensions while retaining its informative coordinates. | Budgets 5 and 10; total 36,000; three seeds. Appendix E.1, pp. 20–22. |
| Sparse linear with correlated coordinates | 50 features; A uses coordinates 0/15/30, B uses 5/20/35, C uses 10/25/40/45. Proportions 35/35/30%. Correlation is generated through \(X_{i+1}=.3X_i+.7\eta\), with \(\eta\sim N(0,1)\), and outcome noise SD .1. | Budgets 5 and 10; total 36,000; three seeds. Appendix E.2, pp. 22–23. |

The full marginal feature generator is not completely specified for the first two families in the PDF. Tables 2 and several captions also contain numbering/label inconsistencies. Those are reasons to record exact source/code versions and explicit chosen parameters, not to invent missing details.

The proposed noise selector initializes \(\alpha\) near 1 for the first family and near 2 for the nonlinear/sparse family. It uses Adam with initial learning rate .1 and a cosine scheduler, full-batch steps, and 1,000 kernel neighbors. The first family uses 200 epochs and 10 Monte Carlo draws; the others use 150 epochs and 50 draws. The population objective is hard maximum. The penalty is reciprocal \(\ell_1\) norm of \(\alpha\); its coefficient is not given in the synthetic implementation paragraphs. The reported hardware is an Apple M3 MacBook Pro.

### 4.3 Real datasets

Adult: Section 4.2, p. 9, and D.4, pp. 19–20 combine the original Adult training/test sets, keep approximately 10%, and use male/female populations. Population-defining features are removed. There are 44 real/one-hot-encoded variables, and the budget is 5. Downstream RF classifiers are trained separately in each population. Results use accuracy and log loss across three seeds. The text initially mentions age groups but then defines the actual groups by sex; reproduction should follow an explicit recorded definition.

ACS: The 2018 person-level PUMS data use California, Florida, and New York populations, approximately 5% per state. A preliminary pooled RF ranking reduces the variables to 18, followed by a budget of 7. Downstream RF regression reports MSE and \(R^2\) over five seeds. Categorical variables are one-hot encoded, missing features median-imputed, missing outcomes dropped, and numeric variables standardized. The paper does not make the fitting partition of every preprocessing step completely explicit. In our suite, any imputation, scaling, or preliminary filtering must fit within selection/training partitions, never on the held-out test set.

Real-data selector settings: \(\alpha\) near 5; 120 epochs; 500 neighbors; 10 Monte Carlo draws; soft maximum temperature 10. ACS learning rate .01 and penalty coefficient \(10^{-4}\); Adult learning rate .02 and penalty coefficient \(5\times10^{-6}\).

### 4.4 Baselines and official code provenance

Appendix C, pp. 14–17 specifies pooled Lasso/logistic selection by coefficient magnitude, pooled XGBoost selection by model importance, population-reweighted variants of both, and an embedded MLP mask. The paper uses Lasso regularization .01 and default XGBoost parameters. Population weights start uniformly and update as \(w_p\leftarrow w_p\exp(\eta L_p)\), followed by normalization. The embedded MLP has 100 hidden units, mask restricted to [0,1], mask penalty .01 selected from [.1,.01,.001], at most 200 epochs, DRO weight updates every 10 epochs, and early stopping patience 10 with improvement threshold \(10^{-4}\).

Official repository: [maitreyiswaroop/dro-feature-selection](https://github.com/maitreyiswaroop/dro-feature-selection), linked from PDF p. 1. The initial local archive is `external/code/dro-feature-selection-main/`; its important files are `gd_pops_v8.py`, `gd_pops_v10.py`, `baselines.py`, `data_baseline_failures.py`, and `modules/baseline_methods.py`.

A newer **partial** frozen snapshot exists under `external/review_20260928/dro-feature-selection/`: `variable_selector.py`, `kernel_estimators.py`, and README. Its manifest records commit `ace7788b49770b29cb1c1a5dccb768cbca891c05`, dated 21 September 2026. It is not a complete runnable repository. The upstream package at this commit uses `dro_feature_selection/` and a unified `run_experiment.py` driver; the frozen [baseline source](https://raw.githubusercontent.com/maitreyiswaroop/dro-feature-selection/ace7788b49770b29cb1c1a5dccb768cbca891c05/dro_feature_selection/baselines.py) was read statically on 3 October 2026.

Two source distinctions affect a careful reproduction:

1. The **old local ZIP** baseline file has an invalid import from `sklearn.metrics` and instantiates `Lasso` even in the classification branch before calling `predict_proba`. The **new frozen upstream** baseline has repaired those classification paths using L1 `LogisticRegression`, `liblinear`, \(C=1/\alpha\), no intercept, and up to 10,000 iterations. It would be inaccurate to say that current upstream retains the old classification bug.
2. The new upstream still forms regression DRO data as \(w_pX,w_pY\), which squares the effective least-squares weights, and its convergence comparison mixes unnormalized updated weights with normalized weights. For classification, weights per observation are \(w_p\), without division by population size. This optimizes a sample-count-weighted quantity unless populations have equal sizes. These are differences between the printed group-risk idea and code details. Any independent correction must be disclosed rather than described as an exact author-code replay.

The old synthetic runner launches a particular `baseline_failure_5` scenario with one seed and SGD; it is not a literal replay of every PDF table setting. Read the generator and frozen parameters before choosing an experiment label.

## 5. What can transfer fairly to our project

The following are design implications from the sources, rather than claims that the papers already performed these experiments.

**Keep optimization and statistical evaluation distinct.** Exhaustive support enumeration on small problems checks whether the global-cut algorithm solves the stated empirical objective. It does not establish power, generalization, or robustness on new observations. The testing papers motivate separate held-out classwise errors; the feature-selection paper motivates a shared downstream evaluation pipeline.

**Use two complementary evaluation tracks.** One track deploys the proposed robust test and compares testing rules at the same feature budget and training sample size. A second track takes each selector's support, retrains the same downstream predictor on an independent training split, and evaluates it on the same held-out observations. The second track isolates selection quality. An all-features model is an upper-resource reference and must be labeled with its larger budget, rather than ranked as a budget-matched selector.

**Measure actual errors, not only the solver gap.** If \(q_0(x)\) denotes the probability of accepting class 0, the randomized Type-I and Type-II errors are

\[
\alpha=E_0[1-q_0(X)],\qquad \beta=E_1[q_0(X)].
\]

The equal-prior balanced error is \((\alpha+\beta)/2\). It equals half the half-hinge sum-risk for \(g=2q_0-1\). A hard score threshold defines a different test and should be reported separately. Logistic surrogate risk, deterministic balanced error, accuracy, and randomized error should not share one ambiguous label. Log loss is supported by Swaroop's classification evaluation. ROC/power at a calibrated Type-I level is a useful extension of the testing papers' matched-false-alarm comparisons, but optimizing a sum-risk alone does not guarantee a prescribed Type-I level.

**Calibration must precede test evaluation.** Thresholds, radii, regularization, and bandwidths should be chosen without examining held-out test labels. If power at nominal \(\alpha_0\) is reported, also report the realized Type-I error on independent class-0 test observations. Otherwise a method can appear powerful by violating the false-positive target. Give Monte Carlo sampling error and between-seed uncertainty separately when both exist.

**Use matched selectors and justified references.** Budget-matched L1-logistic coefficient ranking and XGBoost importance are directly motivated by Swaroop. A random support gives a scale reference. A greedy selector using our same robust oracle measures the value of global integer search rather than changing the statistical objective. The Xie full-feature robust test can verify the fixed-support solver and deployment formulas. Swaroop's group-DRO noise selector can be adapted if each declared population contains both outcomes. Treating class 0 and class 1 as the two populations gives constant labels within each population and can degenerate its conditional-mean objective; that is not a fair direct baseline. A single-population noise selector is a nominal adaptation, not the paper's group-robust experiment.

**Know when a signal-recovery metric is meaningful.** Synthetic generators can expose known informative coordinates, enabling support precision/recall and selection stability. If several correlated variables are substitutable, exact support recovery may penalize equivalent solutions; report predictive performance alongside recovery. Ground-truth coordinate recovery is unavailable for real datasets, so it must not be fabricated from a fitted model's importances.

**Do not overstate empirical robustness.** A finite collection of shifted held-out distributions can measure error across those shifts, with worst observed error as a summary. It is not the supremum over a Wasserstein ball. The training DRO certificate and empirical shift performance should have separate labels. Pair all selectors on the same seed, splits, budgets, and shifts so uncertainty in their differences is interpretable.

## 6. Presentation patterns worth borrowing

The Xie tables put observation budget on rows and competing tests on columns; the Gao/Xie figures put false-alarm rate on the horizontal axis and detection delay on the vertical axis, avoiding unfair threshold comparisons. Swaroop pairs task metrics in adjacent panels and gives complete mean ± standard deviation tables in the appendix. These are useful organizational choices.

For the new study, each figure should answer one question in its caption: whether global cuts save oracle calls, whether selected variables improve independent test performance, or whether errors degrade under a declared shift. Put method definitions, dataset generation, units, sample sizes, repetition counts, uncertainty convention, and budget policy before showing the results. Avoid importing ambiguous captions or the small three-seed count merely because it appears in a reference paper. A substantially larger repeated-seed suite can be planned according to actual runtime, and its sample count must be reported honestly.

No new experiment, solver run, or baseline training was performed while preparing this source record.
