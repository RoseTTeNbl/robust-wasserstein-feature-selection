# Robust feature selection for Wasserstein hypothesis testing

Select a small set of coordinates by minimizing the best class-conditional Wasserstein robust testing risk. Every selected binary subset receives a full inner solve and one tangent at that subset. The same dual certificate also supplies a safe fractional lower bound, which can rule out more unpromising subsets. A fixed Hamming-kernel Gaussian predictor ranks at most eight proposals by mean minus standard deviation. A global binary master provides the optimization bound and a candidate every three evaluations or when the proposal pool is exhausted. Only completed inner solves update the incumbent test.

This repository contains complete English and Chinese methods and experimental reports, independent mathematical checks, and reproducible experiments. The main implementation uses open-source solvers. The private research repository is https://github.com/RoseTTeNbl/robust-wasserstein-feature-selection.

## Read first

- `note/robust_coordinate_selection.pdf`: problem formulation, exact finite reduction, dual certificates, global cuts and algorithmic guarantees; proofs and supplementary constructions are in appendices.
- `note/robust_coordinate_selection_zh.pdf`: complete Chinese counterpart, including the full mathematical body and every appendix.
- `reports/experiment_report.pdf` and `reports/experiment_report_zh.pdf`: corresponding experimental reports with design, metrics, baselines, parameters, results and reproduction details; no appendix.
- `docs/code_guide.md`: detailed Chinese guide to the directory, mathematical meaning of the core code, bounds, deployment and checks.
- `docs/adaptive_search.md`: the current 0.6.0 algorithm: full inner solves, uncertainty-guided candidate ranking, one tangent and one safe fraction from each evaluated subset, and the quantitative guarantee.
- `docs/experiment_plan.md`: protocol written after literature review and before implementation; subsequent exploratory extensions are recorded explicitly.
- `docs/baseline_reproduction.md`: paper objectives, official-source versions, adaptations and a significant source-code sign discrepancy.
- `docs/literature_robust_testing.md` and `docs/literature_variable_selection.md`: experimental reading records, with source pages and missing details.

## Install in a separate environment

Python 3.12 and `uv` are required. On Windows:

```powershell
./scripts/setup.ps1 -WithLogistic
.venv-research/Scripts/python.exe -m pytest -q
```

On other platforms:

```sh
UV_PROJECT_ENVIRONMENT=.venv-research uv sync --locked --all-extras --python 3.12
.venv-research/bin/python -m pytest -q
```

The core only needs NumPy/SciPy. `experiments` adds scikit-learn/Matplotlib; `logistic` adds CVXPY/Clarabel; `dev` adds tests/lint. `uv.lock` pins the complete environment. Existing local `.venv` and historical files are preserved and are not used by the new package.

## One small solve

```powershell
.venv-research/Scripts/python.exe -m robusthw --budget 2 --enumerate --output results/demo.json
```

An input NPZ may contain `x0`, `x1` and optional `q0`, `q1`. Choose `--loss logistic` for the optional exponential-cone oracle. `--gate box`, `triangle` or `risk` selects an integer-exact gate; the default is risk. `--cut-mode ratio` retains complete fractional certificates using shared product variables, an explicitly labeled exploratory extension. It requires the risk gate. Neither mode changes the original integer objective.

`--search adaptive` is the default for risk/tangent. Its default lower model retains both the owned tangent and the safe fraction from the same dual certificate; `--linear-bounds` reproduces the earlier GP search with only tangents. `--proposals` bounds the inexpensive candidate pool (default and maximum 8, including a refreshed master candidate). Every selected candidate is fully evaluated. The predictor reuses an incrementally updated Cholesky factor, without fitting kernel parameters between evaluations. Predictions never update the reported upper or lower bounds. `--search master` runs the earlier master-only search. Fixed-gate ablations and the exploratory ratio mode retain that path; `--root-rounds` applies only there. Existing frozen experiment method IDs retain their original search; `risk_adaptive` uses the current version for fresh experiments.

The result contains a feasible support/test upper bound, a global numerical lower bound, status, timing, visited-support count and trace. A time limit is not reported as optimal. Half-hinge risk is the sum of two classwise errors; divide by two for equal-prior randomized balanced error. This is different from a level-0.05 two-sample equality test.

## Reproduce the experiments

```powershell
.venv-research/Scripts/python.exe scripts/fetch_uci.py
.venv-research/Scripts/python.exe -m robusthw.experiments.runner --config configs/statistical.json --output results/research_20261003/statistical
.venv-research/Scripts/python.exe -m robusthw.experiments.runner --config configs/computational.json --output results/research_20261003/computational
.venv-research/Scripts/python.exe -m robusthw.experiments.runner --config configs/real.json --output results/research_20261003/real
.venv-research/Scripts/python.exe -m robusthw.experiments.runner --config configs/sensitivity.json --output results/research_20261003/sensitivity
.venv-research/Scripts/python.exe -m robusthw.experiments.runner --config configs/ratio_cuts.json --output results/research_20261003/ratio
.venv-research/Scripts/python.exe scripts/summarize.py
.venv-research/Scripts/python.exe scripts/plot_results.py
```

Run at most two experiment processes concurrently; every optimizer is single-threaded. For a small correctness pilot add `--limit 1` and use a separate output directory. Existing matching records are resumed, and conflicting configuration hashes are rejected. Per-trial result JSON files, compact tables, summaries, configuration snapshots, figures, environment and test logs are exported to GitHub. Full optimization traces are preserved locally. Raw third-party datasets, PDFs, repositories, environments and historical archives are excluded.

Synthetic evaluation uses independent selection, downstream training, classification, alternative-test and null-test samples. Every support receives the same robust and RBF-SVM downstream heads and a common calibrated RBF-MMD equality test. Confidence intervals resample training seeds, rather than pretending conditional testing repeats are independent selections. UCI results report repeated-split variability, with data attribution and hashes.

## Build the documents

The theory sources are `note/main.tex` and the complete Chinese `note/main_zh.tex`. Every English revision must update the corresponding Chinese title, abstract, body, formulas, proofs, algorithms and appendices. After reviewing both full sources, explicitly record the approved versions; the build never records a review automatically:

```powershell
.venv-research/Scripts/python.exe scripts/check_note_sync.py --record --reviewed-by "reviewer" --review-note "Complete bilingual review"
.venv-research/Scripts/python.exe scripts/check_note_sync.py
```

`note/bilingual_sync.json` stores source/structure SHA-256 fingerprints and review metadata. The checker uses only Python's standard library. It rejects stale sources, missing mathematical structure, and recording an English change without a Chinese update. Semantic review remains necessary; structure checks do not translate or certify the prose. See `docs/bilingual_note.md` and `AGENTS.md`.

The Windows build scripts use an existing XeLaTeX installation and unique system TEMP directories. `note/build.ps1` first checks the approved bilingual pair, then compiles both immutable source snapshots. Only two successful builds publish the stable English/Chinese PDFs; failures retain prior files as prior builds. The English text uses Times New Roman; Chinese text uses SimSun with SimHei headings and Times New Roman for Latin terminology. Both versions use Cambria Math. Native XeTeX Chinese line breaking avoids the incompatible xeCJK release on the verified host. These fonts are available on the verified Windows host. The app's built-in compiler was attempted but is unavailable on this host; system XeLaTeX and rendered-page inspection verify delivery. `note/main.md` is an additional English reading version.

```powershell
./note/build.ps1
./reports/build.ps1
```

The experimental reports are generated from `reports/report_template.tex`, `reports/report_template_zh.tex` and saved records. Update both templates together. Final generation checks all 1,640 historical outcomes and the separately frozen search development and confirmation evidence; it rejects stale or incomplete inputs. Both languages must compile successfully before paired publication. `-Draft` produces a labeled TEMP-only preview.

The manuscripts were reorganized on 7 October 2026 without changing the algorithms, configurations or experimental records. Revision checks are stored in `reports/verification/20261007/`. Verification under `results/research_20261003/` describes the original study and its original manuscript release; the corresponding prior PDFs and sources remain recoverable from Git history and the local archive.

## Scope of the evidence

The theory covers finite, nonnegative, nondecreasing convex losses under positive radii and fixed L1 feature scaling. The nontrivial analysis assumes a nonconstant loss with positive loss at score zero; otherwise the constant test is already optimal. The implementation and recorded study use half hinge and logistic. The 9 October 2026 rewrite derives whole-space feasibility by convex interpolation and presents the dual using a transport matrix and one correction vector; it changes no study records or core code. Independent checks also compare small exponential and squared-hinge instances with their normalized duals. Certificates in the recorded computations use declared floating-point tolerances, not interval arithmetic. Stronger fractional values do not guarantee faster masters, and an empirical robust optimum does not automatically imply support recovery, two-sample power or population coverage.

Version 0.4.0 adds history-guided acquisition, feasible-score repricing, inexpensive recentered cuts, and persistent logistic solver models. Its quantitative theorem bounds accurate binary evaluations by a data-weighted packing number and gives a directly computable feature-tail bound. These are conditional on the specified oracle/master accuracy, not a wall-time claim. New validation is saved under `reports/verification/20261010/adaptive_search/`; the 1,640 historical study records and their report remain evidence for their original algorithm version.

Version 0.5.0 introduced the full-inner search `full_inner_uncertainty_v3`: no solution transfers, candidate-centered tangents or fractional inner evaluations. Its fixed predictor trains only on completed, accuracy-checked binary evaluations. Earlier 0.4.0 and geometric-score results remain evidence for their respective versions.

Version 0.6.0 retains the same predictor, pool cap and certification schedule, and keeps the safe fractional lower bound already supplied by each evaluated support's dual certificate. The search is tagged `full_inner_fraction_v4`; the predictor is still `fixed_hamming_gp_v1`. Section 5 and Appendix C explain the lower model and binary master, while Appendix E derives the predictor. The 26-case integration regression and the 16-configuration confirmation with three timing repeats are frozen separately in `reports/verification/20261010/next_selection/current/`. Summed within-configuration median complete evaluations decrease from 598 to 453 (24.2%) relative to the original GP search. Summed within-configuration median runtime increases from 12.715 to 13.262 seconds (4.3%); tighter masters have a cost. These 16 configurations reuse four generated datasets and validate computational behavior, not statistical power or population performance.

Reproduce the frozen small-instance validation in a fresh directory:

```powershell
.venv-research/Scripts/python.exe scripts/validate_next_selection.py --help
.venv-research/Scripts/python.exe scripts/validate_next_selection.py development --root results/new_validation
.venv-research/Scripts/python.exe scripts/validate_next_selection.py confirmation --root results/new_validation
```

The runner saves its protocol and source snapshot before solving, checks every method against complete enumeration, and seals all records afterward. Timing repeats do not count as additional independent datasets. All development candidates and failures are preserved next to the final validation.
