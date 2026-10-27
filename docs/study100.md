# 100-dimensional experimental design

The numerical study uses the standard hinge loss `ell(t)=max(0,1+t)`.
Sampling, metrics and baselines are separate from the optimization core.
The dedicated batch runner in `study100.py` has not been migrated to cone_v3;
current experiment commands are listed in [experiments.md](experiments.md).

## Data and evaluation

The design retains the small-sample, 100-dimensional setting of
[Xie, Gao and Xie (2021), Section 5.1](https://arxiv.org/pdf/2105.14348),
and evaluates support recovery alongside prediction as in
[Bertsimas, Pauphilet and Van Parys, Section 3.1](https://arxiv.org/pdf/1710.01352).
It uses a sparse Gaussian mean difference rather than reproducing either study.

For a uniformly sampled support `T` of size 5, let `s` be its indicator:

```text
class 0: N(+0.6 s, I_100)
class 1: N(-0.6 s, I_100)
```

Training sizes are 20 and 25; validation and test sizes are 200 and 1,000 per class.
Every method selects five coordinates. Pooled training means and standard deviations
are used to standardize every split. A second test set increases noise standard
deviation to 1.25 while keeping the means fixed; its noise draws are paired with
the nominal test set. Models are not refitted for this evaluation.

The configuration is in `configs/study100.json`; `make_data` implements sampling
and preprocessing. Calibration uses an independent development sample. Choose a
shared radius scale from `{0.05,0.15,0.4}` using validation error on the Fisher
support, with class radius `scale*sqrt(20/n_k)`. Choose the L1 logistic parameter
`C` from `{0.1,1,10}` on development validation data. Freeze both choices before
confirmation evaluation.

## Methods and code

| Method | Selection | Detector |
| --- | --- | --- |
| Outer approximation | Joint robust-risk optimization | Robust hinge detector |
| Fisher | Largest squared mean gaps divided by summed within-class variances | Same robust detector |
| L1 robust | Five largest absolute L1 logistic coefficients | Same robust detector |
| Random robust | Uniform five-coordinate subset | Same robust detector |
| L1 logistic | L1 coefficient ranking | L2 logistic refit with `C=1` |

`study100_baselines.py` contains the four baseline operations: `fisher_ranking`,
`l1_support`, `random_support` and `fit_logistic`. Positive detector scores indicate
class 0, so `logistic_score` reverses scikit-learn's decision-function sign.

Report the sum of class errors, empirical hinge loss for robust detectors, recovered
signal count, training time and the outer-approximation optimality gap. Keep status,
timeouts and failures alongside each result. The input consists of class-specific
sample matrices and a configuration; outputs contain selected coordinates, fitted
scores, predictions, metrics and optimization traces.
