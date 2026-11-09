"""results/<study>/summary.md and its figures, from accuracy.json and computation.json."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import study20_figures as figures
from .study20 import CASES, D, TRUTH

METHODS = ("select", "marginal", "full", "oracle", "logistic", "sparse_logistic", "svm_rbf")
ROBUST = ("select", "marginal", "full", "oracle")
GROUND = [c for c in CASES if TRUTH[c]]

INTRODUCTION = """\
## Question

Two classes differ only on a few coordinates. Can a test built from a handful of training
points per class still tell them apart, and does it find the right coordinates? Each case has
D = 20 coordinates: **four ground-truth coordinates**, on which the two class distributions
differ, and **sixteen distractors**, which are independent of everything and have the same
distribution in both classes. Coordinates are shuffled in every trial, so no method can find
the ground truth by its index.

## Data

| case | ground-truth coordinates: class 0 vs class 1 | distractors (both classes) | source | what it tests |
|---|---|---|---|---|
| `null` | none (the classes are identical) | 20 x N(0, 1) | - | false alarms: every test should have error 0.5 |
| `mean_shift` | N(0, S) vs N(mu, S), mu = (2, 1, 0.67, 0.5), S_ab = 0.5^abs(a-b) | 16 x N(0, 1) | Wang, Dey, Xie (2023), case I | the common sparse signal, visible coordinate by coordinate |
| `mixture` | N(0, I) vs N(+-mu, I) with a random sign, mu as above | 16 x N(0, 1) | Wang, Dey, Xie (2023), case IV | same mean, larger spread: invisible to a linear rule |
| `interaction` | 1/2 N(c0, I) + 1/2 N(-c0, I) vs the same with c1, c0 = (1, 1, 1, 1), c1 = (1, 1, -1, -1) | 16 x (1/2 N(1, 1) + 1/2 N(-1, 1)) | Xie, Gao, Xie (2021), Sec. 5.1, on four coordinates | every single coordinate has the same distribution in both classes; the signal is only in how the true coordinates move together |

The likelihood ratio depends on the ground-truth coordinates only, so the Bayes error (the
smallest error any test can reach, computed from the known densities on the test points) is
the same as with the four true coordinates alone.

## Protocol

As in Xie, Gao and Xie (2021, Sec. 5.1): in each trial draw n training points per class
(n = {sizes}) and 1000 fresh test points per class; choose every tuning parameter by 5-fold
cross-validation on the training points, minimizing the validation error; refit on all
training points; measure on the test points. {trials} trials per case and n.

## Methods

| method | what it does | tuned by cross-validation |
|---|---|---|
| **selection (ours)** | among all subsets S of d coordinates, take the one with the smallest worst-case risk V(S); fit the robust test (Eq. (4)) on S and extend it to new points by Eq. (5) | d in 1..4 and the radius |
| marginal screening | score every coordinate alone by V({m}) and keep the d best; then the same robust test | d and the radius |
| all 20 coordinates | the robust test of Xie, Gao and Xie (2021) on all coordinates, no selection | the radius |
| true coordinates (reference) | the same robust test on the four ground-truth coordinates; not available in practice | the radius |
| logistic | logistic regression (a linear rule) on all coordinates; baseline of Xie, Gao and Xie | L2 penalty |
| L1 logistic | logistic regression with an L1 penalty, which sets the weights of unused coordinates to zero; sparse baseline of Wang, Dey and Xie | L1 penalty |
| RBF SVM | support vector machine with a Gaussian kernel (a curved rule); baseline of Xie, Gao and Xie | C |

The robust methods use the hinge loss and the l1 transport cost; the radius grid is 17
values from 0.01 to 100 on a log scale, split equally between the classes.

## Metrics

A test declares class 0 or class 1 for each point; for the robust tests, a nonnegative score
means class 0.

- **Type I error** = share of class-0 test points declared class 1.
  **Type II error** = share of class-1 test points declared class 0.
- **Test error** = (type I + type II) / 2. A coin flip gives 0.5; the Bayes error is the floor.
- **FDP** (false discovery proportion) = |S minus truth| / |S|: the share of chosen coordinates
  that are distractors. **NDP** (non-discovery proportion) = |truth minus S| / 4: the share of
  ground-truth coordinates missed (Wang, Dey and Xie 2023, Eq. (24)). Choosing d coordinates at
  random gives FDP = 0.8 and NDP = 1 - d/20.
- **Certificate** = V(S)/4, computed from the training points alone: the largest average error
  that the randomized test (declare class 0 with probability (1 + g)/2, g the score) can have
  when each true class distribution lies in its Wasserstein ball. **Randomized error** is that
  test's average error measured on the test points; the certificate **holds** in a trial when
  certificate >= randomized error.

Every number below is a mean over trials; standard errors are in parentheses, and the bands in
the figures are +-2 standard errors.
"""


def _mean(values):
    values = np.asarray(values, float)
    return f"{values.mean():.3f} ({values.std(ddof=1) / np.sqrt(len(values)):.3f})"


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def _groups(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["case"], record["n"]].append(record)
    return groups, sorted({n for _, n in groups})


def error_tables(groups, sizes):
    out = []
    for case in CASES:
        rows = [["Bayes", *[_mean([r["bayes"]["error"] for r in groups[case, n]]) for n in sizes]]]
        for method in METHODS:
            row = [figures.LABEL[method]]
            for n in sizes:
                errors = [r["methods"][method]["error"] for r in groups[case, n]
                          if method in r["methods"]]
                row.append(_mean(errors) if errors else "-")
            if row[1:] != ["-"] * len(sizes):
                rows.append(row)
        out += [f"**{case}**", "", _table(["method", *[f"n = {n}" for n in sizes]], rows), ""]
    return out


def type_tables(groups, sizes):
    out = []
    for case in CASES:
        rows = [["Bayes", *[f"{np.mean([r['bayes']['type1'] for r in groups[case, n]]):.2f} / "
                            f"{np.mean([r['bayes']['type2'] for r in groups[case, n]]):.2f}"
                            for n in sizes]]]
        for method in METHODS:
            if method not in groups[case, sizes[0]][0]["methods"]:
                continue
            rows.append([figures.LABEL[method], *[
                f"{np.mean([r['methods'][method]['type1'] for r in groups[case, n]]):.2f} / "
                f"{np.mean([r['methods'][method]['type2'] for r in groups[case, n]]):.2f}"
                for n in sizes]])
        out += [f"**{case}**", "", _table(["method", *[f"n = {n}" for n in sizes]], rows), ""]
    return out


def recovery_table(groups, sizes):
    rows = []
    for case in GROUND:
        for method in ("select", "marginal", "sparse_logistic"):
            for n in sizes:
                trials = groups[case, n]
                rows.append([case, figures.LABEL[method], n,
                             _mean([r["methods"][method]["fdp"] for r in trials]),
                             _mean([r["methods"][method]["ndp"] for r in trials]),
                             f"{np.mean([len(r['methods'][method]['support']) for r in trials]):.1f}"])
    return [_table(["case", "method", "n", "FDP", "NDP", "mean |S|"], rows), ""]


def certificate_table(groups, sizes, last):
    rows = []
    for case in CASES:
        for method in ROBUST:
            for n in sizes:
                trials = [r["methods"][method] for r in groups[case, n] if method in r["methods"]]
                if not trials:
                    continue
                certificate = np.array([t["certificate"] for t in trials])
                randomized = np.array([t["randomized_error"] for t in trials])
                ends = np.mean([t["radius_index"] in (0, last) for t in trials])
                sizes_ = np.bincount([len(t["support"]) for t in trials], minlength=5)[1:5]
                rows.append([case, figures.LABEL[method], n, _mean(certificate),
                             _mean(randomized), f"{np.mean(certificate >= randomized):.2f}",
                             f"{np.median([t['radius'] for t in trials]):.3g}", f"{ends:.2f}",
                             "/".join(map(str, sizes_)) if method in ("select", "marginal")
                             else "-"])
    return ["held = share of trials with certificate >= randomized error; ends = share of "
            "trials whose radius is an end of the grid; d = 1/2/3/4 counts the chosen sizes.", "",
            _table(["case", "method", "n", "certificate", "randomized", "held", "median radius",
                    "ends", "d = 1/2/3/4"], rows), ""]


def computation_section(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["case"], record["n"], record["budget"]].append(record)
    rows = []
    for (case, n, d), trials in sorted(groups.items()):
        best = np.array([t["enumeration"]["value"] for t in trials])
        milp = [t["compact_program"] for t in trials]
        oa = [t["outer_approximation"] for t in trials]
        tree = [t["branch_and_bound"] for t in trials]
        rows.append([
            case, n, d, trials[0]["enumeration"]["subsets"],
            f"{np.mean([t['enumeration']['seconds'] for t in trials]):.2f}",
            f"{np.mean([m['value'] is not None and abs(m['value'] - b) < 1e-5 and m['status'] == 0 for m, b in zip(milp, best)]):.1f}",
            f"{np.mean([m['seconds'] for m in milp]):.1f}",
            f"{np.mean([m['nodes'] for m in milp]):.0f}",
            f"{np.mean([o['status'] == 'optimal' for o in oa]):.1f}",
            f"{np.mean([o['subsets'] for o in oa]):.0f}",
            f"{np.mean([o['seconds'] for o in oa]):.0f}",
            f"{np.mean([abs(t['value'] - b) < 1e-6 for t, b in zip(tree, best)]):.1f}",
            f"{np.mean([t['lps'] for t in tree]):.0f}",
            f"{np.mean([t['seconds'] for t in tree]):.1f}",
            f"{np.mean([t['commitment_bound'] / b for t, b in zip(trials, best)]):.2f}",
        ])
    return [
        "## Computation: exact selection",
        "",
        "Selection has to find the subset with the smallest V(S) among all C(20, d) subsets. "
        "Four exact algorithms, at the radius that cross-validation chose for selection in the "
        "accuracy study, 10 trials per setting, time limit 600 s:",
        "",
        "- **enumeration**: compute V(S) for every subset, by the matching form of V for the "
        "hinge loss (Hungarian algorithm);",
        "- **compact MILP**: the whole selection problem written as one mixed-integer linear "
        "program (binary z_m = coordinate m chosen; Eq. (4) of the directions note) and solved "
        "by HiGHS. Its linear relaxation (z between 0 and 1) is the *commitment bound*;",
        "- **branch and bound**: a search tree over \"coordinate m in / out\", pruned by the "
        "commitment bound of each node;",
        "- **outer approximation** (Algorithm 1 of the theory note): solve one subset, add the "
        "ratio cut Eq. (8), solve the master problem Eq. (9) for the next subset.",
        "",
        "Columns: share of trials solved to the optimum (checked against enumeration), mean "
        "seconds and the work done (MILP nodes, subsets evaluated by outer approximation, LPs "
        "of branch and bound); bound/opt = commitment bound / optimum.",
        "",
        "![computation](figures/computation.png)",
        "",
        _table(["case", "n", "d", "subsets", "enum s", "MILP opt", "MILP s", "MILP nodes",
                "OA opt", "OA subsets", "OA s", "B&B opt", "B&B LPs", "B&B s", "bound/opt"], rows),
        "",
    ]


def summarize(output):
    output = Path(output)
    accuracy = json.loads((output / "accuracy.json").read_text())
    config, records = accuracy["config"], accuracy["records"]
    groups, sizes = _groups(records)
    (output / "figures").mkdir(exist_ok=True)
    figures.error_figure(records, output / "figures" / "error.png",
                         ("select", "marginal", "full", "oracle", "logistic", "sparse_logistic",
                          "svm_rbf"))
    figures.recovery_figure(records, output / "figures" / "recovery.png")
    figures.certificate_figure(records, output / "figures" / "certificate.png")
    environment = accuracy["environment"]
    lines = [
        f"# {config['study_id']}: four ground-truth coordinates among {D - 4} distractors", "",
        f"Model `{environment['model_version']}`, commit `{environment['git_commit']}` (dirty: "
        f"{environment['git_dirty']}). Generated by `scripts/run_study20.py summarize` from the "
        "JSON files in this directory; configuration `configs/study20.json`.", "",
        INTRODUCTION.replace("{sizes}", ", ".join(map(str, sizes))).replace("{trials}", str(config["trials"])),
        "## Results", "",
        "### Can the test tell the classes apart?", "",
        "![test error](figures/error.png)", "",
        "Test error by case (rows: methods, columns: training points per class).", "",
        *error_tables(groups, sizes),
        "Type I / type II error (means).", "",
        *type_tables(groups, sizes),
        "### Does it find the ground-truth coordinates?", "",
        "![recovery](figures/recovery.png)", "",
        "The dotted line FDP = 0.8 is random choice.", "",
        *recovery_table(groups, sizes),
        "### Does the certificate hold?", "",
        "![certificate](figures/certificate.png)", "",
        "Selection (ours): the promised worst-case error V(S)/4 against the measured error of the "
        "randomized test, at the cross-validated radius.", "",
        *certificate_table(groups, sizes, config["radii"][2] - 1),
    ]
    if (output / "computation.json").exists():
        computation = json.loads((output / "computation.json").read_text())["records"]
        figures.computation_figure(computation, output / "figures" / "computation.png")
        lines += computation_section(computation)
    (output / "summary.md").write_text("\n".join(lines))
