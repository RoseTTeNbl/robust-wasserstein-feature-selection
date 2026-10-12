# Robust feature selection for Wasserstein hypothesis testing

Select a small set of coordinates by minimizing the best class-conditional Wasserstein robust testing risk. The current 0.7.0 default uses a global binary master to choose each next subset. Every selected subset receives a complete primal and dual solve, yielding an actual test, one tangent and a safe fractional lower bound from the same certificate. Feasible tests update the upper bound; the master provides a global lower bound. The algorithm stops when their gap reaches the requested accuracy.

This repository contains complete English and Chinese methods and experimental reports, independent mathematical checks, and reproducible experiments. The main implementation uses open-source solvers. The private research repository is https://github.com/RoseTTeNbl/robust-wasserstein-feature-selection.

## Read first

- `note/robust_coordinate_selection.pdf`: problem formulation, exact finite reduction, dual certificates, global cuts and algorithmic guarantees; proofs and supplementary constructions are in appendices.
- `note/robust_coordinate_selection_zh.pdf`: complete Chinese counterpart, including the full mathematical body and every appendix.
- `reports/experiment_report.pdf` and `reports/experiment_report_zh.pdf`: corresponding experimental reports with design, metrics, baselines, parameters, results and reproduction details; no appendix.
- `docs/code_guide.md`: detailed Chinese guide to the directory, mathematical meaning of the core code, bounds, deployment and checks.
- `docs/adaptive_search.md`: the current 0.7.0 algorithm: complete binary evaluations, owned certificates, a global master at every step, and its precision requirements. The filename is retained for existing links.
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

An input NPZ may contain `x0`, `x1` and optional `q0`, `q1`. Choose `--loss logistic` for the optional exponential-cone oracle. The default `--search cuts` uses the risk gate and retains both tangents and safe fractions. `--linear-bounds` selects the tangent-only comparison. It has the same inner-accuracy requirements and global-master schedule. The current default creates no risk predictor or proposal pool and performs no fractional-root or core-point oracle solves.

The Python API defaults to absolute tolerance `1e-6` for `search="cuts"`. An explicitly supplied `relative_tolerance` retains the mixed stopping rule; use zero when comparing with the theory's fixed absolute guarantee. Each evaluated subset must attain an inner gap at most one quarter of the absolute tolerance. Unresolved oracle or master accuracy returns a failure status and the current valid interval.

The older paths remain explicit comparisons: `--search adaptive` selects the historical GP and small-pool search; `--proposals` applies only there. `--search master` retains the earlier outer loop, gate comparisons and optional fractional preprocessing. Use `--search master` with `--gate box` or `--gate triangle` for those gate comparisons, and `--search master --cut-mode ratio` for the earlier ratio mode. `--root-rounds` applies only to that older loop. Their default relative tolerance remains `1e-5`. Frozen experiment method IDs and records retain their original algorithms.

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

Historical version 0.6.0 retained the predictor, pool cap and certification schedule, and added the safe fractional lower bound already supplied by each evaluated subset's dual certificate. Its search is tagged `full_inner_fraction_v4`; its predictor is `fixed_hamming_gp_v1`. The prior manuscript explained that predictor in Appendix E; it is no longer part of the current theory. The 26-case integration regression and the 16-configuration confirmation with three timing repeats are frozen in `reports/verification/20261010/next_selection/current/`. Summed within-configuration median complete evaluations decreased from 598 to 453 (24.2%) relative to the original GP search. Summed within-configuration median runtime increased from 12.715 to 13.262 seconds (4.3%); tighter masters had a cost. These records compare the earlier searches and do not establish that GP benefits the new default.

Version 0.7.0 uses `certified_global_cuts_v1`: a complete binary inner evaluation followed by a certified global master at each step. It retains the same feasible-test and certificate repairs, matrix reuse, and optional safe fractions. The GP and its candidate pool remain available only to reproduce historical comparisons. The new paired ablation uses the same sixteen publicly specified configurations and four underlying generated datasets; it is a retrospective process comparison, not independent confirmation or an isolated test of GP alone. Its source snapshot and records are saved separately from the earlier evidence.

The frozen 0.7.0 comparison is in `reports/verification/20261012/no_prediction/`.
Each method has 48 runs: three timing repeats of each of sixteen configurations.
Totals below sum the within-configuration medians, rather than pooling all timing
repeats as independent datasets.

| Search process | Retained bounds | Complete subset evaluations | Time (s) |
| --- | --- | ---: | ---: |
| Current global master | Tangents | 601 | 15.672 |
| Current global master | Tangents and safe fractions | 446 | 18.509 |
| Historical GP/pool | Tangents | 598 | 11.627 |
| Historical GP/pool | Tangents and safe fractions | 453 | 11.779 |

Keeping fractions in the current process reduced complete evaluations by 25.8%
but increased measured time by 18.1%. This supports fewer inner evaluations,
not a runtime improvement. The historical GP/pool process also changes candidate
generation and master frequency, so this comparison does not identify GP's
individual contribution. All 192 outcomes passed their saved checks against the
existing complete-enumeration references.

Reproduce the frozen small-instance validation in a fresh directory:

```powershell
.venv-research/Scripts/python.exe scripts/validate_next_selection.py --help
.venv-research/Scripts/python.exe scripts/validate_next_selection.py development --root results/new_validation
.venv-research/Scripts/python.exe scripts/validate_next_selection.py confirmation --root results/new_validation
```

The runner saves its protocol and source snapshot before solving, checks every method against complete enumeration, and seals all records afterward. Timing repeats do not count as additional independent datasets. All development candidates and failures are preserved next to the final validation.

Reproduce the current two-by-two comparison in a fresh directory:

```powershell
.venv-research/Scripts/python.exe scripts/validate_certified_search.py --root results/new_certified_validation
```

Its four methods are current global-master search with tangents or with tangents plus fractions, and each corresponding historical GP search. Data and existing complete-enumeration reference hashes are verified before execution. Each method has three timing repetitions per configuration; every outcome and failure is retained without tuning the selection rule on those results.
