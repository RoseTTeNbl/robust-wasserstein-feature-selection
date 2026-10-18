"""Shared English figures drawn only from verified experimental summaries.

Both report languages use these same vector assets. Error bars are recorded
standard deviations; search panels keep each model/sample-size condition apart.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

MODELS = ("mean", "variance", "correlation", "mixture")
MODEL_NAMES = ("Mean", "Variance", "Correlation", "Mixture")
METHOD_NAMES = {
    "global": "Global search",
    "forward": "Forward",
    "forward_swap": "Forward + swaps",
    "marginal_mmd": "Marginal MMD",
    "l1_logistic": "L1 logistic",
    "mrmr": "mRMR",
    "joint_mmd": "Joint MMD",
    "random": "Random",
    "enumeration": "Exhaustive",
}
HIGHLIGHT = ("global", "forward_swap", "joint_mmd", "random")
COLORS = {
    "global": "#355F7D",
    "forward_swap": "#6F8C76",
    "joint_mmd": "#B47E51",
    "random": "#8B8B8B",
}
MARKERS = {"global": "o", "forward_swap": "s", "joint_mmd": "^", "random": "x"}
MODEL_COLORS = ("#355F7D", "#B47E51", "#6F8C76", "#85818C")
MODEL_MARKERS = ("o", "s", "^", "v")


def style():
    """Quiet serif typography, thin axes, and a stable muted palette."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.titleweight": "normal",
            "axes.titlepad": 9,
            "axes.labelsize": 10,
            "axes.labelpad": 6,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "0.3",
            "axes.unicode_minus": True,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 220,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def groups(summary, **condition):
    return [
        row
        for row in summary["groups"]
        if row.get("n_instances", 0) > 0
        and all(row["group"].get(k) == v for k, v in condition.items())
    ]


def measure(row, metric):
    item = row["metrics"].get(metric, {})
    return item.get("mean"), item.get("sd")


def save(fig, output, key):
    path = output / f"{key}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.07)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.07)
    plt.close(fig)
    return path.resolve().as_posix()


def method_legend(fig):
    handles = [
        Line2D([], [], color=COLORS[method], marker=MARKERS[method], label=METHOD_NAMES[method])
        for method in HIGHLIGHT
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, columnspacing=1.6)


def reference(ax, orientation="horizontal", level=50):
    """A constant-decision error reference, not a claim of no information."""
    line = ax.axhline if orientation == "horizontal" else ax.axvline
    line(level, color="0.72", linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)


def synthetic(summary, output):
    selected = groups(summary, suite="main", D=50, n=20, s=4, role="comparison")
    if not selected:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), sharey=True, layout="constrained")
    for panel, (model, title, ax) in enumerate(zip(MODELS, MODEL_NAMES, axes.flat)):
        for method in HIGHLIGHT:
            rows = sorted(
                [
                    row
                    for row in selected
                    if row["group"]["model"] == model and row["group"]["method"] == method
                ],
                key=lambda row: row["group"]["budget"],
            )
            rows = [row for row in rows if measure(row, "ber_prob")[0] is not None]
            if rows:
                means, sd = zip(*(measure(row, "ber_prob") for row in rows))
                ax.errorbar(
                    [row["group"]["budget"] for row in rows],
                    np.array(means) * 100,
                    yerr=np.array([v or 0 for v in sd]) * 100,
                    color=COLORS[method],
                    marker=MARKERS[method],
                    capsize=2,
                    elinewidth=0.8,
                    markeredgewidth=0.8,
                )
        reference(ax)
        ax.axhline(20, color="0.75", linewidth=0.8, linestyle=":", zorder=0)
        ax.set(title=f"({chr(97 + panel)}) {title}", xticks=[2, 4, 8], ylim=(0, 55))
    for ax in axes[1, :]:
        ax.set_xlabel("Selected variables")
    for ax in axes[:, 0]:
        ax.set_ylabel("Balanced error (%)")
    method_legend(fig)
    return save(fig, output, "main")


def public(summary, output):
    datasets = ("wdbc", "ionosphere", "sonar", "har")
    titles = ("WDBC", "Ionosphere", "Sonar", "HAR")
    selected = [
        row
        for row in groups(summary, suite="public", role="comparison")
        if (row["group"]["n"], row["group"]["budget"])
        == ((40, 16) if row["group"]["model"] == "har" else (20, 4))
    ]
    if not selected:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 3.3), sharey=True, layout="constrained")
    methods = tuple(METHOD_NAMES)[:8]
    for panel, (dataset, title, ax) in enumerate(zip(datasets, titles, axes)):
        for i, method in enumerate(methods):
            rows = [
                row
                for row in selected
                if row["group"]["model"] == dataset and row["group"]["method"] == method
            ]
            if rows and measure(rows[0], "ber_prob")[0] is not None:
                mean, sd = measure(rows[0], "ber_prob")
                ax.errorbar(
                    100 * mean,
                    i,
                    xerr=100 * (sd or 0),
                    fmt=MARKERS.get(method, "o"),
                    color=COLORS.get(method, "#777777"),
                    capsize=2,
                    elinewidth=0.8,
                    markersize=4,
                )
        ax.set(title=f"({chr(97 + panel)}) {title}", xlim=(0, 55), xticks=[0, 25, 50])
        ax.set_xlabel("Balanced error (%)")
        reference(ax, "vertical")
        ax.set_yticks(range(8), [METHOD_NAMES[method] for method in methods])
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    return save(fig, output, "public")


def shift(summary, output):
    rows = summary.get("shift_groups", [])
    if not rows:
        return None
    fig, axes = plt.subplots(
        2, 4, figsize=(8.4, 4.6), sharex=True, sharey=True, layout="constrained"
    )
    for col, (model, title) in enumerate(zip(MODELS, MODEL_NAMES)):
        for rownum, kind in enumerate(("translation", "gaussian_noise")):
            ax = axes[rownum, col]
            for method in HIGHLIGHT:
                points = sorted(
                    [
                        row
                        for row in rows
                        if row["group"].get("model") == model
                        and row["group"].get("method") == method
                        and row["group"].get("shift_kind", row["group"].get("shift_type")) == kind
                        and measure(row, "ber_prob")[0] is not None
                    ],
                    key=lambda row: row["group"]["delta_factor"],
                )
                if points:
                    ax.plot(
                        [row["group"]["delta_factor"] for row in points],
                        np.array([measure(row, "ber_prob")[0] for row in points]) * 100,
                        color=COLORS[method],
                        marker=MARKERS[method],
                    )
            reference(ax)
            ax.set(xticks=[0, 0.5, 1, 2], ylim=(0, 60))
            if rownum == 0:
                ax.set_title(title)
            else:
                ax.set_xlabel("Shift / mean radius")
    axes[0, 0].set_ylabel("Translation\nBalanced error (%)")
    axes[1, 0].set_ylabel("Added noise\nBalanced error (%)")
    method_legend(fig)
    return save(fig, output, "shift")


def search(summary, output):
    selected = groups(summary, suite="small", role="comparison")
    if not selected:
        return None
    methods = ("enumeration", "global", "forward", "forward_swap")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.95), sharey=True, layout="constrained")
    offsets = np.linspace(-0.17, 0.17, 8)
    for i, method in enumerate(methods):
        rows = [row for row in selected if row["group"]["method"] == method]
        # Each dot is one observed model/sample-size condition mean.
        for col, metric in enumerate(("excess_risk", "fit_count")):
            values = []
            for j, model in enumerate(MODELS):
                family = sorted(
                    [row for row in rows if row["group"]["model"] == model],
                    key=lambda row: row["group"]["n"],
                )
                for k, row in enumerate(family):
                    value = measure(row, metric)[0]
                    if value is None:
                        continue
                    values.append((value, row["n_instances"]))
                    axes[col].scatter(
                        value,
                        i + offsets[(2 * j + k) % len(offsets)],
                        s=23,
                        marker=MODEL_MARKERS[j],
                        facecolor=MODEL_COLORS[j] if k % 2 else "white",
                        edgecolor=MODEL_COLORS[j],
                        linewidth=0.9,
                        zorder=3,
                    )
            if values:
                mean = np.average([v for v, _ in values], weights=[n for _, n in values])
                axes[col].plot(mean, i + 0.29, "|", color="0.15", markersize=7, zorder=4)
    axes[0].set_yticks(range(len(methods)), [METHOD_NAMES[m] for m in methods])
    axes[0].set(title="(a) Solution quality", xlabel="Risk above the optimum")
    axes[1].set(title="(b) Optimization work", xlabel="Fixed-subset fits")
    axes[0].axvline(0, color="0.78", linewidth=0.8, zorder=0)
    axes[0].set_xlim(left=-0.002)
    axes[1].set_xlim(left=0)
    axes[1].set_xticks([0, 40, 80, 120])
    axes[0].invert_yaxis()
    for ax in axes:
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            color=color,
            markerfacecolor="white",
            markersize=5,
            label=label,
        )
        for marker, color, label in zip(MODEL_MARKERS, MODEL_COLORS, MODEL_NAMES)
    ]
    handles.append(Line2D([], [], linestyle="none", marker="|", color="0.15", label="Overall mean"))
    fig.legend(handles=handles, loc="outside lower center", ncol=5, columnspacing=1.4)
    return save(fig, output, "search")


def anchor(summary, output):
    rows = groups(summary, suite="anchor", method="all_variables")
    if not rows:
        return None
    row = rows[0]
    metrics = ("upper", "ber_prob", "ber_hard", "error0_hard", "error1_hard")
    values = [measure(row, key)[0] for key in metrics]
    if any(value is None for value in values):
        return None
    means = np.array(values) * 100
    sd = np.array([measure(row, key)[1] or 0 for key in metrics]) * 100
    means[0] /= 2
    sd[0] /= 2
    fig, axes = plt.subplots(
        1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.3, 1]}, layout="constrained"
    )
    for i in range(3):
        axes[0].errorbar(
            means[i],
            i,
            xerr=sd[i],
            fmt="o",
            markersize=5,
            color=COLORS["forward_swap"] if i == 0 else COLORS["global"],
            capsize=2,
            elinewidth=0.9,
        )
        axes[0].annotate(
            f"{means[i]:.1f}",
            (means[i] + sd[i], i),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )
    axes[0].set(
        title="(a) Training objective and test errors",
        yticks=range(3),
        yticklabels=["Training robust risk / 2", "Test probability error", "Test threshold error"],
        xlabel="Risk / error (%)",
        xlim=(0, 58),
        xticks=[0, 25, 50],
        ylim=(2.65, -0.65),
    )
    reference(axes[0], "vertical")
    for i in range(2):
        axes[1].plot(means[i + 3], i, "o", color=COLORS["joint_mmd"], markersize=5)
        axes[1].annotate(
            f"{means[i + 3]:.0f}",
            (means[i + 3], i),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
        )
    axes[1].set(
        title="(b) Class-specific threshold errors",
        yticks=[0, 1],
        yticklabels=["True class 0", "True class 1"],
        xlabel="Error (%)",
        xlim=(-3, 114),
        xticks=[0, 50, 100],
        ylim=(1.65, -0.65),
    )
    for ax in axes:
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    return save(fig, output, "anchor")


def build_figures(summary, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    style()
    figures = {}
    for key, function in (
        ("main", synthetic),
        ("public", public),
        ("shift", shift),
        ("search", search),
        ("anchor", anchor),
    ):
        path = function(summary, output)
        figures[key] = {"en": path, "zh": path}
    return figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_figures(json.loads(args.summary.read_text(encoding="utf-8")), args.output)
    (args.output / "figures.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
