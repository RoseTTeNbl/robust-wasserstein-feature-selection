# Runtime, data, and baseline inventory

This file preserves the **initial planning snapshot**, before the new implementation, environment and repository were created. Statements about what was then active or installed are historical. For the delivered state, use the root README, `uv.lock`, `docs/code_guide.md` and `results/research_20261003/environment.json`.

Checked on 2026-10-03 (Asia/Shanghai). This is a read-only inventory for planning the risk-adaptive global-cut implementation and experiments. No algorithm, experiment, package installation, Git initialization, or publication was performed for this inventory. The only solver probe created a COPT environment; it did not construct or solve a model. No credentials or tokens were read.

## Current active implementation

The active package is `code/robusthw`, version 0.2.0. It still implements the 2026-09-30 box-bound gate baseline. The revised risk-adaptive theory in `note/main.md` has not yet replaced it. Existing results must retain their old method labels.

| File | Current responsibility |
|---|---|
| `problem.py` | Finite input/weights/radii validation, full coordinate distances, affine cuts and results |
| `oracle.py` | COPT gate primal and explicit dual, numerical certificate repair, test extension |
| `solver.py` | SciPy/HiGHS integer master, upper/lower bounds, support cache, independent enumeration |
| `cli.py` | One command-line entry, NPZ input, JSON result persistence |
| `code/tests/` | Analytic cases, independent overlap transport LP, cut validity and solver-status behavior |

Keep this core separation. A minimal new implementation can share one fixed-support oracle among box, shared-risk, and risk-adaptive cut builders. The experiment layer should add only a deterministic dataset/split module, thin baseline adapters, one runner, and one aggregation/plot command. Certificate construction and verification belong in the core rather than the experiment scripts. Freeze the objective, baselines, and metrics before implementing those components.

## Installed runtime and hardware

| Item | Observed value |
|---|---|
| OS | Windows 11 Home, Chinese edition |
| CPU | AMD Ryzen 9 7945HX, 16 physical cores, 32 logical processors |
| RAM | 15.69 GiB total; 2.98 GiB free at the inventory snapshot |
| Python | Project `.venv`, Python 3.12.14, 64-bit |
| NumPy / SciPy | 2.5.3 / 1.18.1 |
| COPT | `coptpy` 8.0.6; environment starts without a license, with size limitations |
| Test tools | pytest 9.1.1; ruff 0.16.7 |
| Existing experiment libraries | scikit-learn 1.9.1; matplotlib 3.11.2 |
| Not installed in project `.venv` | CVXPY, SCS, Clarabel, highspy, PyTorch, pandas, psutil, POT |
| Available commands | Git 2.45.1; uv 0.9.16; Julia 1.9.0-alpha1 |
| GitHub CLI | `gh` not on PATH and absent at the two conventional installation locations checked |

COPT's environment output explicitly reports no license and limited-size noncommercial mode. Successful small old runs do not establish availability for larger experiments. Do not depend on this license mode for the main publication workflow.

The current dense coordinate-distance arrays take approximately `8*N*N*D` bytes, where `N=n0+n1`. This is about 76.5 MiB for the full 569-by-30 WDBC data and 32.0 MiB for 351-by-34 Ionosphere, before temporary arrays, solver matrices, or certificates. With only about 3 GiB free at this snapshot, start with at most two experiment workers and one solver/BLAS thread per worker. Record realized wall time and peak memory; 32 logical processors do not justify 32 simultaneous cone solves.

## Open solver route

SciPy is already installed and supplies HiGHS-backed continuous LP and MILP interfaces. This is the natural first route for the half-hinge/overlap oracle and integer master. The MILP result exposes `mip_dual_bound`, so limited runs can preserve a real global lower bound. See the official [LP documentation](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ipm.html) and [MILP documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html). HiGHS uses an [MIT license](https://github.com/ERGO-Code/HiGHS/blob/master/LICENSE.txt).

Logistic needs exponential-cone support. Both [Clarabel](https://clarabel.org/stable/api_cone_types/) and [SCS](https://www.cvxgrp.org/scs/api/cones.html) support this cone. They are not installed here. Prefer an explicitly selected open exponential-cone backend, with SCS available for independent checks; this is a planning recommendation, not a measured performance result. Every backend must return numerically repaired primal/dual certificates and an interval, especially after an inaccurate or time-limited solve. Solver status alone does not establish a trustworthy cut. SCS is available under its [upstream license](https://github.com/cvxgrp/scs/blob/master/LICENSE.txt); Clarabel's [upstream license](https://github.com/oxfordcontrol/Clarabel.rs/blob/main/LICENSE.md) should accompany any redistribution.

Use direct sparse matrices or one small modeling layer consistently. Avoid retaining COPT as a mandatory import for users who only want the open route.

## Isolation and dependency recommendation

Preserve the existing `.venv` for reproducing old results. After the experimental protocol is frozen, create a dedicated new environment through uv and keep all dependencies in one `pyproject.toml` plus `uv.lock`.

Recommended dependency groups are:

- Core: NumPy, SciPy, and the chosen open exponential-cone solver.
- `experiments`: scikit-learn, matplotlib, and only libraries actually needed by data/evaluation.
- `baselines`: PyTorch/pandas/tqdm/six for the approved upstream adapters; keep these optional.
- `copt`: optional `coptpy` for cross-checks or licensed runs.
- `dev`: pytest and ruff.

Pin a stable Python minor version, commit the lock, and use a frozen sync for reproduction. The existing `requirements-lock.txt` contains only four top-level pins and is not a transitive environment lock. Do not install PyTorch or solver packages into the user's global Anaconda environment merely because uv resides there.

## Available data and provenance

`data/raw/uci/manifest.json` records URLs, SHA-256 hashes, DOI, dimensions, class mapping, and exclusions for two local datasets:

| Dataset | Local arrays | Source and licensing |
|---|---|---|
| WDBC | `x`: 569 x 30; `y`: 569 | [UCI WDBC](https://archive.ics.uci.edu/dataset/17/breast+cancer), DOI 10.24432/C5DW2B, CC BY 4.0 |
| Ionosphere | `x`: 351 x 34; `y`: 351 | [UCI Ionosphere](https://archive.ics.uci.edu/dataset/52/ionosphere), DOI 10.24432/C5W01B, CC BY 4.0 |

The official dataset pages were checked on this inventory date and confirm those licenses. Original ZIPs are already local. New statistical experiments should generate fresh train/validation/test splits from the raw arrays, fit scaling on training data only, and persist seeds and selected indices. Preserve the existing source checksums.

`data/generated/` has twelve historical NPZ files: small examples with five observations per class and eight features, and GMM examples with ten observations per class and 100 features. Each contains `x0,x1`. These are debugging inputs and historical data, not a sufficient multi-seed publication benchmark. A new experiment should use a documented generator and new seeds, rather than relabel old results.

## Local upstream baselines

No upstream implementation was executed during this inventory.

| Local snapshot | Structure and minimum integration | Scope/provenance issues |
|---|---|---|
| `external/code/spinoptmmd` | `src/tst.py` exposes `LinearMMDTest` and `QuadMMDTest` with `optimize_w`; calls SciPy plus PyTorch. `src/utils.py` also imports pandas and tqdm; `src/kernel.py` requires PyTorch/six. A thin adapter can call those interfaces without running all upstream experiments. | BSD 3-Clause license exists; reused `jitkrittum` has its own notice. It is a sparse MMD two-sample method, so its statistical test and selection objective differ from our Wasserstein minimax risk. |
| `external/code/SDRHT` | `Models/WDRO.py` provides a finite transport CVXPY reference, and standard classifiers are separated from neural Sinkhorn methods. | MIT license exists. The WDRO function uses Euclidean distances, equal class sample counts, and `2*sum(min(p0,p1))`. An L1/half-hinge adaptation must be labeled and normalized explicitly; it is not an unchanged same-objective oracle. Full neural experiments require PyTorch/torchdiffeq and are unnecessary for a minimal fixed-support reference. |
| `external/code/dro-feature-selection-main` | An older source ZIP with experiment scripts, `modules/variable_selector.py`, and kernel/Monte Carlo estimators; no installable package/requirements/license file found in the local snapshot. | Group-DRO feature degradation with squared-loss conditional-variance objectives, rather than the present two-class Wasserstein testing objective. The ZIP tracks an old main branch and must not be treated as the frozen latest version. |
| `external/review_20260928/dro-feature-selection` | Only README and two core Python files were downloaded. Imports reference missing `dro_feature_selection.config` and other package modules, so this partial snapshot is not directly runnable. | Manifest pins upstream commit `ace7788b49770b29cb1c1a5dccb768cbca891c05` and hashes. Reproduction requires a complete pinned author checkout and license check. |
| `external/review_20260927/trimmed_lasso/official_gsm` | MATLAB 2018b-era least-squares solver, with MEX C components and examples. | No license file found locally; the objective is sparse least squares. Not a drop-in robust-test baseline. Its old adapted smoothing paths are historical project code and should not be inherited as the new method. |

Use the original author code through isolated adapters and retain upstream identity, commit/hash, parameter mapping, and failure status. Comparing different selection objectives through a common downstream test is legitimate when labeled; comparing their reported objective values directly is not.

## GitHub/export readiness

The project root is not currently a Git repository. Git's configured credential helper is `manager`; the helper's stored secrets were not queried. The connected GitHub plugin's read-only identity request succeeded for account `RoseTTeNbl`. That establishes authenticated connector access, not permission to create a repository or push code. No repository was created, no remote configured, and no files uploaded.

The current `.gitignore` only covers the virtual environment, bytecode, test cache, egg-info, and some logs. It does not protect `external/`, `_archive/`, `data/`, or old `results/` from accidental staging.

For publication, build an explicit export set containing own active source/tests, frozen experiment configs, dependency lock, README, citations/provenance, license, and selected aggregate result tables/figures. Exclude original third-party PDFs/source checkouts, original data archives, virtual environments, solver licenses, historical archives, and bulk logs/results. Ship download/fetch manifests and scripts instead of silently vendoring third-party code or data. Include only intentionally selected own note artifacts. Confirm that a clean clone can run the small verification suite and documented smoke experiment before publishing.
