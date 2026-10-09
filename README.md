# Robust feature selection for Wasserstein hypothesis testing

Select a small set of coordinates by minimizing the best class-conditional Wasserstein robust testing risk. The implementation uses full convex oracles, objective-scaled dual certificates, and a globally solved binary master. It does not use a fixed historical transport plan or a local-neighborhood restriction.

This repository contains English and complete Chinese methods reports, a separate English empirical evaluation, independent mathematical checks, and reproducible experiments. The main implementation uses open-source solvers. The private research repository is https://github.com/RoseTTeNbl/robust-wasserstein-feature-selection.

## Read first

- `note/robust_coordinate_selection.pdf`: problem formulation, exact finite reduction, dual certificates, global cuts and algorithmic guarantees; proofs and supplementary constructions are in appendices.
- `note/robust_coordinate_selection_zh.pdf`: complete Chinese counterpart, including the full mathematical body and every appendix.
- `reports/experiment_report.pdf`: experimental design and principal findings, with complete numerical results and reproduction details in appendices.
- `docs/code_guide.md`: detailed Chinese guide to the directory, mathematical meaning of the core code, bounds, deployment and checks.
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

The experimental report is generated from `reports/report_template.tex` and saved records; change its explanatory source in the template before rebuilding. Final generation checks all 1,640 scheduled outcomes and rejects stale or incomplete summaries. The published result JSON and provenance allow reconstruction without local historical archives or raw datasets. `-Draft` produces a labeled TEMP-only preview.

The manuscripts were reorganized on 7 October 2026 without changing the algorithms, configurations or experimental records. Revision checks are stored in `reports/verification/20261007/`. Verification under `results/research_20261003/` describes the original study and its original manuscript release; the corresponding prior PDFs and sources remain recoverable from Git history and the local archive.

## Scope of the evidence

The theory covers finite, nonnegative, nondecreasing convex losses under positive radii and fixed L1 feature scaling. The nontrivial analysis assumes a nonconstant loss with positive loss at score zero; otherwise the constant test is already optimal. The implementation and recorded study use half hinge and logistic. The 9 October 2026 rewrite derives whole-space feasibility by convex interpolation and presents the dual using a transport matrix and one correction vector; it changes no study records or core code. Independent checks also compare small exponential and squared-hinge instances with their normalized duals. Certificates in the recorded computations use declared floating-point tolerances, not interval arithmetic. Stronger fractional values do not guarantee faster masters, and an empirical robust optimum does not automatically imply support recovery, two-sample power or population coverage.
