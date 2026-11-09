"""Figures of results/<study>/summary.md (PNG, matplotlib)."""

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .study20 import CASES, TRUTH  # noqa: E402

# Categorical slots in fixed order; references in ink.
COLOR = {"select": "#2a78d6", "marginal": "#eb6834", "full": "#1baf7a", "logistic": "#eda100",
         "sparse_logistic": "#e87ba4", "svm_rbf": "#008300", "oracle": "#6250d6"}
LABEL = {"select": "selection (ours)", "marginal": "marginal screening",
         "full": "all 20 coordinates", "oracle": "true coordinates", "logistic": "logistic",
         "sparse_logistic": "L1 logistic", "svm_rbf": "RBF SVM"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GROUND = [c for c in CASES if TRUTH[c]]

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
    "lines.markersize": 5, "legend.frameon": False, "figure.dpi": 150,
})


def _groups(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["case"], record["n"]].append(record)
    return groups, sorted({n for _, n in groups})


def _band(ax, sizes, values, color, label, style="-"):
    mean = np.array([np.mean(v) for v in values])
    se = np.array([np.std(v, ddof=1) / np.sqrt(len(v)) for v in values])
    ax.plot(sizes, mean, style, color=color, label=label, marker="o")
    ax.fill_between(sizes, mean - 2 * se, mean + 2 * se, color=color, alpha=0.12, linewidth=0)


def _legend(fig, axes):
    entries = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            entries.setdefault(label, handle)
    fig.legend(list(entries.values()), list(entries), loc="lower center", ncol=min(len(entries), 4),
               bbox_to_anchor=(0.5, -0.02))


def error_figure(records, path, methods):
    groups, sizes = _groups(records)
    fig, axes = plt.subplots(1, len(CASES), figsize=(3.0 * len(CASES), 3.0), sharey=True)
    for ax, case in zip(axes, CASES):
        bayes = [np.mean([r["bayes"]["error"] for r in groups[case, n]]) for n in sizes]
        ax.plot(sizes, bayes, "--", color=INK, linewidth=1.2, label="Bayes (best possible)")
        ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
        for method in methods:
            values = [[r["methods"][method]["error"] for r in groups[case, n]
                       if method in r["methods"]] for n in sizes]
            if all(values):
                _band(ax, sizes, values, COLOR[method], LABEL[method],
                      "-." if method == "oracle" else "-")
        ax.set_title(case, color=INK)
        ax.set_xlabel("training points per class")
        ax.set_xticks(sizes)
    axes[0].set_ylabel("test error (mean of type I and II)")
    axes[0].set_ylim(0.1, 0.53)
    _legend(fig, axes)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def recovery_figure(records, path):
    groups, sizes = _groups(records)
    fig, axes = plt.subplots(2, len(GROUND), figsize=(3.0 * len(GROUND), 5.2), sharex=True,
                             sharey=True)
    for column, case in enumerate(GROUND):
        for row, key in enumerate(("fdp", "ndp")):
            ax = axes[row, column]
            for method in ("select", "marginal", "sparse_logistic"):
                values = [[r["methods"][method][key] for r in groups[case, n]] for n in sizes]
                _band(ax, sizes, values, COLOR[method], LABEL[method])
            if row == 0:
                ax.axhline(0.8, color=MUTED, linewidth=1, linestyle=":")
                ax.set_title(case, color=INK)
            else:
                ax.set_xlabel("training points per class")
                ax.set_xticks(sizes)
        axes[0, 0].set_ylabel("FDP: share of chosen\ncoordinates that are distractors")
        axes[1, 0].set_ylabel("NDP: share of true\ncoordinates missed")
    axes[0, 0].set_ylim(0, 1)
    _legend(fig, axes[0])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def certificate_figure(records, path):
    groups, sizes = _groups(records)
    fig, axes = plt.subplots(1, len(CASES), figsize=(3.0 * len(CASES), 3.0), sharey=True)
    for ax, case in zip(axes, CASES):
        for key, color, label in (("certificate", "#2a78d6", "certificate V(S)/4 (promised)"),
                                  ("randomized_error", "#eb6834",
                                   "randomized test, measured error")):
            values = [[r["methods"]["select"][key] for r in groups[case, n]] for n in sizes]
            _band(ax, sizes, values, color, label)
        ax.set_title(case, color=INK)
        ax.set_xlabel("training points per class")
        ax.set_xticks(sizes)
    axes[0].set_ylabel("error of the selected test")
    axes[0].set_ylim(0, 0.55)
    _legend(fig, axes)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def computation_figure(records, path):
    groups = defaultdict(list)
    for record in records:
        groups[record["case"], record["n"]].append(record)
    keys = sorted(groups)
    fig, axes = plt.subplots(1, len(keys), figsize=(2.8 * len(keys), 3.0), sharey=True)
    series = (("enumeration", "seconds", "enumeration", "#2a78d6"),
              ("compact_program", "seconds", "compact MILP (HiGHS)", "#eb6834"),
              ("branch_and_bound", "seconds", "branch and bound", "#1baf7a"),
              ("outer_approximation", "seconds", "outer approximation (Alg. 1)", "#eda100"))
    for ax, key in zip(np.atleast_1d(axes), keys):
        budgets = sorted({r["budget"] for r in groups[key]})
        for name, field, label, color in series:
            means = [np.mean([r[name][field] for r in groups[key] if r["budget"] == d])
                     for d in budgets]
            ax.plot(budgets, means, "-o", color=color, label=label)
        ax.axhline(groups[key][0].get("time_limit", 600), color=MUTED, linewidth=1, linestyle=":")
        ax.set_yscale("log")
        ax.set_title(f"{key[0]}, n = {key[1]}", color=INK)
        ax.set_xlabel("d (coordinates chosen)")
        ax.set_xticks(budgets)
    np.atleast_1d(axes)[0].set_ylabel("mean seconds (log scale)")
    _legend(fig, np.atleast_1d(axes))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
