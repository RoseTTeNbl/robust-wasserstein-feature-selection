"""Tables of results/study20-v2/summary.md from accuracy.json and computation.json."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .study20 import CASES, TRUTH, D

METHODS = ("select", "marginal", "full", "oracle", "logistic", "sparse_logistic", "svm_rbf")
ROBUST = ("select", "marginal", "full", "oracle")


def _mean(values):
    values = np.asarray(values, float)
    return f"{values.mean():.3f} ({values.std(ddof=1) / np.sqrt(len(values)):.3f})"


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def accuracy_tables(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["case"], record["n"]].append(record)
    sizes = sorted({n for _, n in groups})
    out = []
    for n in sizes:
        rows = [["Bayes", *[_mean([r["bayes"] for r in groups[c, n]]) for c in CASES]]]
        for method in METHODS:
            row = [method]
            for case in CASES:
                errors = [r["methods"][method]["error"] for r in groups[case, n]
                          if method in r["methods"]]
                row.append(_mean(errors) if errors else "-")
            rows.append(row)
        out += [f"### Test error, n = {n} per class (mean over trials, standard error)", "",
                _table(["method", *CASES], rows), ""]
    rows = []
    for n in sizes:
        for case in CASES:
            if not 0 < len(TRUTH[case]) < D:
                continue
            for method in ("select", "marginal", "sparse_logistic"):
                trials = groups[case, n]
                sizes_ = [len(r["methods"][method]["support"]) for r in trials]
                rows.append([n, case, method, _mean([r["methods"][method]["fdp"] for r in trials]),
                             _mean([r["methods"][method]["ndp"] for r in trials]),
                             f"{np.mean(sizes_):.1f}"])
    out += ["### Recovered coordinates (FDP and NDP of Wang, Dey and Xie, Eq. (24))", "",
            _table(["n", "case", "method", "FDP", "NDP", "mean |S|"], rows), ""]
    rows = []
    for n in sizes:
        for case in CASES:
            for method in ROBUST:
                trials = [r["methods"][method] for r in groups[case, n] if method in r["methods"]]
                if not trials:
                    continue
                certificate = [t["certificate"] for t in trials]
                randomized = [t["randomized_error"] for t in trials]
                radii = [t["radius"] for t in trials]
                ends = np.mean([t["radius_index"] in (0, 12) for t in trials])
                held = np.mean(np.array(certificate) >= np.array(randomized))
                sizes_ = np.bincount([len(t["support"]) for t in trials], minlength=5)[1:5]
                rows.append([n, case, method, _mean(certificate), _mean(randomized),
                             f"{held:.2f}", f"{np.median(radii):.3g}", f"{ends:.2f}",
                             "/".join(map(str, sizes_)) if method in ("select", "marginal") else "-"])
    out += ["### Certificates of the robust tests at the cross-validated radius", "",
            "certificate = V(S)/4, the worst-case error of the randomized test over the two "
            "Wasserstein balls; randomized = its error on the test set; held = fraction of trials "
            "with certificate >= randomized; ends = fraction of trials choosing an end of the "
            "radius grid; d = 1/2/3/4 counts the chosen sizes.", "",
            _table(["n", "case", "method", "certificate", "randomized", "held", "median radius",
                    "ends", "d = 1/2/3/4"], rows), ""]
    return out


def computation_table(records):
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
    return ["### Exact selection at the cross-validated radius", "",
            "Columns per method: fraction of trials solved to the optimum, mean seconds, and "
            "the work done (MILP nodes, subsets evaluated by outer approximation, LPs of branch "
            "and bound). bound/opt is the commitment bound over the optimum.", "",
            _table(["case", "n", "d", "subsets", "enum s", "MILP opt", "MILP s", "MILP nodes",
                    "OA opt", "OA subsets", "OA s", "B&B opt", "B&B LPs", "B&B s",
                    "bound/opt"], rows), ""]


def summarize(output):
    output = Path(output)
    accuracy = json.loads((output / "accuracy.json").read_text())
    lines = ["# study20-v2", "",
             f"Model `{accuracy['environment']['model_version']}`, commit "
             f"`{accuracy['environment']['git_commit']}` (dirty: "
             f"{accuracy['environment']['git_dirty']}). Generated from the JSON files here.", ""]
    lines += accuracy_tables(accuracy["records"])
    if (output / "computation.json").exists():
        lines += computation_table(json.loads((output / "computation.json").read_text())["records"])
    (output / "summary.md").write_text("\n".join(lines))
