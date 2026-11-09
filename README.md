# Wasserstein robust variable selection

Choose a small set of coordinates so that the worst-case risk of a two-sample test,
over Wasserstein balls around the empirical distributions, is as small as possible.
The current model is `hinge_ratio_cut_v5`. It implements the finite transport problem,
the convex selection function $F$ (Proposition 2.4, stated through $1/F$) and the
outer-approximation algorithm with ratio cuts from the theory note. The numerical code
uses the hinge loss and allows unequal class sizes.

## Entry points

- [Theory note](note/robust_coordinate_selection.pdf) · [LaTeX source](note/main.tex) · [Markdown copy](note/main.md)
- [Research directions (Chinese)](note/directions/research_directions.pdf) · [LaTeX source](note/directions/directions.tex) (XeLaTeX) · [development checks](note/directions/dev_checks.py)
- [Reading guide (Chinese): the problem, every formula, the derivation of Eqs. (6)–(9), code flow, experiments and how to read the results](docs/reading_guide.md)
- [20-dimensional study results](results/study20-v2/summary.md) · [run script](scripts/run_study20.py) · [configuration](configs/study20.json)
- [Algorithm-to-code map](docs/algorithm.md)
- [Experiment tools and records](docs/experiments.md)
- [Maintaining the theory note](docs/theory_note.md)

## Install and check

Python 3.12; dependencies are pinned in `uv.lock`. The core needs NumPy, SciPy and the open-source HiGHS solver.

```powershell
./scripts/setup.ps1
.venv-research/Scripts/python.exe -m robusthw demo --budget 2
.venv-research/Scripts/python.exe -m pytest -q
.venv-research/Scripts/python.exe -m ruff check src tests scripts
```

On other platforms run `UV_PROJECT_ENVIRONMENT=.venv-research uv sync --locked --extra experiments --extra dev`
and then use `.venv-research/bin/python`.

## Code layout

```text
src/robusthw/algorithm/
  model.py          problem data, results and the off-sample prediction rule (Eq. 5)
  oracle.py         one LP per fixed subset: detector, bounds and ratio cut (Eqs. 3, 4, 8)
  selection.py      outer approximation (Algorithm 1), enumeration and greedy search
src/robusthw/experiments/
                    data generation, baselines, protocol, metrics and summaries
configs/            experiment settings
note/               theory note, review record and PDF
scripts/            setup, checks, builds, experiments and reports
tests/              mathematical checks and end-to-end regressions
```

Outer approximation returns a feasible upper bound on the risk and a global lower bound,
and reports optimality only when the gap reaches the tolerance. A stored cut is Eq. (8):
$1/F(z)\le$ `intercept + slope @ z`. See the [algorithm-to-code map](docs/algorithm.md)
for the line-by-line correspondence. Experiments, baselines and result processing are
separate from the core algorithm; each new study writes to its own output directory.

## Building the theory note

After editing, review the text, formulas, proofs and references, then run:

```powershell
.venv-research/Scripts/python.exe scripts/check_note_sync.py --record --reviewed-by reviewer
./note/build.ps1
```

Section 1 (problem formulation) is kept verbatim. Intermediate build files go to a private
system TEMP directory; check every PDF page before publishing.
