"""Publication-style figures generated exclusively from saved summaries."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

COLORS = dict(
    risk="#25658A",
    risk_no_root="#70A2BC",
    risk_ratio="#163B55",
    box="#7F8790",
    triangle="#C88742",
    greedy="#D18A3D",
    mmd="#43876B",
    wang="#8367A6",
    l1="#B85F64",
    random="#8B9299",
)
LABEL = dict(
    risk="Global",
    risk_no_root="No root cuts",
    risk_ratio="Complete ratios",
    box="Box",
    triangle="Triangle",
    greedy="Greedy",
    mmd="MMD screen",
    wang="Wang objective",
    l1="L1 logistic",
    random="Random",
)
METHODS = ["risk", "greedy", "mmd", "wang", "l1", "random"]
MECHANISMS = ["mean", "variance", "dependence", "mixture"]
TITLES = dict(
    mean="Mean difference",
    variance="Variance difference",
    dependence="Equal marginals, different dependence",
    mixture="Symmetric mixture",
)


def style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "savefig.bbox": "tight",
        }
    )


def save(fig, path):
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(path.with_suffix("." + suffix), dpi=220)
    plt.close(fig)


def clean(ax):
    ax.grid(axis="y", alpha=0.18, lw=0.6)
    ax.set_axisbelow(True)


def lookup(summary, suite, dataset, method, dimension=None, theta=0.25):
    return next(
        (
            r
            for r in summary
            if r["suite"] == suite
            and r["dataset"] == dataset
            and r["method"] == method
            and abs(r["theta"] - theta) < 1e-10
            and (dimension is None or r["dimension"] == dimension)
        ),
        None,
    )


def interval(r, metric, real=False):
    mean = r[metric + "_mean"]
    if real:
        lo, hi = mean - r[metric + "_sd"], mean + r[metric + "_sd"]
    else:
        lo, hi = r[metric + "_lo"], r[metric + "_hi"]
    return mean, np.array([[max(0.0, mean - lo)], [max(0.0, hi - mean)]])


def analytic(out):
    z = np.linspace(0, 1, 401)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), layout="constrained")
    axes[0].plot(
        z, np.maximum(0.5, 1 - 2 * z), color=COLORS["box"], ls="--", label="Box"
    )
    axes[0].plot(
        z, np.maximum(0.5, 1 - z), color=COLORS["triangle"], ls="-.", label="Triangle"
    )
    axes[0].plot(z, 1 / (1 + z), color=COLORS["risk"], lw=2, label="Risk adaptive")
    axes[0].scatter([0, 1], [1, 0.5], s=25, color="black", zorder=4)
    axes[0].set(
        xlabel="Fractional coordinate activation z",
        ylabel="Convex oracle value",
        title="(a) Stronger between integer supports",
        ylim=(0.45, 1.04),
    )
    axes[0].legend(frameon=False)
    anchor = 0.4
    v = 1 / (1 + anchor)
    g = -1 / (1 + anchor) ** 2
    axes[1].plot(
        z, 1 / (1 + z), color=COLORS["risk"], lw=2, label="Complete certificate"
    )
    axes[1].plot(
        z, v + g * (z - anchor), color=COLORS["triangle"], ls="--", label="Its tangent"
    )
    axes[1].scatter([anchor], [v], s=28, color=COLORS["risk"])
    axes[1].set(
        xlabel="A different support coordinate",
        ylabel="Reusable lower bound",
        title="(b) Retaining the curve keeps more information",
        ylim=(0.35, 1.04),
    )
    axes[1].legend(frameon=False)
    for ax in axes:
        clean(ax)
    save(fig, out / "gate_mechanism")


def split_diagram(out):
    fig, ax = plt.subplots(figsize=(7.2, 2.1))
    ax.set(xlim=(0, 10), ylim=(0, 3))
    ax.axis("off")
    boxes = [
        (0, 1.3, 2.2, 1.0, "Selection samples\nfit scale and support"),
        (3.2, 1.8, 2.55, 0.85, "Independent head training\nrobust test and RBF SVM"),
        (7, 1.8, 2.75, 0.85, "Held-out individual samples\nerror, AUC, stress"),
        (3.2, 0.25, 2.55, 0.9, "Freeze support and bandwidth\nno test-label fitting"),
        (7, 0.25, 2.75, 0.9, "Fresh two-sample batches\npower and null rejection"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(
            Rectangle((x, y), w, h, fill=False, edgecolor=COLORS["risk"], lw=1.1)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    for a, b in [
        ((2.2, 1.8), (3.2, 2.22)),
        ((2.2, 1.8), (3.2, 0.7)),
        ((5.75, 2.22), (7, 2.22)),
        ((5.75, 0.7), (7, 0.7)),
    ]:
        ax.annotate(
            "",
            xy=b,
            xytext=a,
            arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#69747B"},
        )
    save(fig, out / "evaluation_split")


def testing(summary, out):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5), layout="constrained")
    for ax, dataset in zip(axes.flat, MECHANISMS):
        for j, method in enumerate(METHODS):
            r = lookup(summary, "statistical", dataset, method)
            if not r:
                continue
            for offset, metric, marker, fill in [
                (-0.09, "power", "o", True),
                (0.09, "null_rejection", "^", False),
            ]:
                y, err = interval(r, metric)
                ax.errorbar(
                    j + offset,
                    y,
                    yerr=err,
                    color=COLORS[method],
                    marker=marker,
                    mfc=COLORS[method] if fill else "white",
                    ms=4,
                    capsize=2,
                    lw=1,
                )
        ax.axhline(0.05, color="#777", ls=":", lw=1)
        ax.set(
            title=TITLES[dataset],
            ylim=(-0.025, 1.035),
            ylabel="Rejection probability",
            xticks=np.arange(6),
            xticklabels=["Global", "Greedy", "MMD", "Wang", "L1", "Random"],
        )
        clean(ax)
    fig.legend(
        handles=[
            Line2D([], [], color="black", marker="o", ls="", label="Alternative power"),
            Line2D(
                [],
                [],
                color="black",
                marker="^",
                mfc="white",
                ls="",
                label="Null rejection",
            ),
        ],
        loc="outside upper center",
        ncol=2,
        frameon=False,
    )
    save(fig, out / "power_and_calibration")


def recovery(summary, out):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), layout="constrained")
    for ax, dataset in zip(axes.flat, MECHANISMS):
        for j, method in enumerate(METHODS):
            r = lookup(summary, "statistical", dataset, method)
            if not r:
                continue
            y, err = interval(r, "recall")
            ax.bar(j, y, color=COLORS[method], width=0.62, alpha=0.9)
            ax.errorbar(j, y, yerr=err, color="#343434", lw=0.9, capsize=2)
            ax.scatter(
                j,
                r["stability_jaccard"],
                marker="D",
                s=18,
                color="white",
                edgecolors="#333",
                zorder=4,
            )
        ax.set(
            title=TITLES[dataset],
            ylim=(0, 1.08),
            ylabel="Recall / stability",
            xticks=np.arange(6),
            xticklabels=["Global", "Greedy", "MMD", "Wang", "L1", "Random"],
        )
        clean(ax)
    axes[0, 0].legend(
        handles=[
            Rectangle((0, 0), 1, 1, color="#888", label="Signal recall"),
            Line2D(
                [],
                [],
                color="#333",
                marker="D",
                mfc="white",
                ls="",
                label="Support stability",
            ),
        ],
        frameon=False,
    )
    save(fig, out / "support_recovery")


def stress(summary, out):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), layout="constrained")
    for ax, dataset in zip(axes.flat, MECHANISMS):
        for method in METHODS:
            r = lookup(summary, "statistical", dataset, method)
            if not r:
                continue
            rho = np.array([0, 0.25, 0.5])
            metrics = [f"stress_ber_{x:.2f}" for x in rho]
            ys = np.array([r[m + "_mean"] for m in metrics])
            lo = np.array([r[m + "_lo"] for m in metrics])
            hi = np.array([r[m + "_hi"] for m in metrics])
            ax.plot(
                rho,
                ys,
                color=COLORS[method],
                marker="o",
                ms=3,
                lw=1.4,
                label=LABEL[method],
            )
            ax.fill_between(rho, lo, hi, color=COLORS[method], alpha=0.08, lw=0)
        ax.set(
            title=TITLES[dataset],
            xlabel="L1 translation size",
            ylabel="Deterministic balanced error",
            ylim=(0, 0.6),
            xticks=[0, 0.25, 0.5],
        )
        clean(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    save(fig, out / "population_stress")


def computational(summary, out):
    methods = ["box", "triangle", "risk_no_root", "risk"]
    line_style = {"box": "--", "triangle": "-.", "risk_no_root": ":", "risk": "-"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
    dims = [8, 12, 20, 40, 80]
    for method in methods:
        for ax, dataset in zip(axes, ["mean", "mixture"]):
            points = [
                lookup(summary, "computational", dataset, method, d) for d in dims
            ]
            pairs = [(d, r) for d, r in zip(dims, points) if r]
            if not pairs:
                continue
            xs = [x for x, r in pairs]
            ys = [r["selection_seconds_mean"] for x, r in pairs]
            ax.errorbar(
                xs,
                ys,
                yerr=np.array(
                    [interval(r, "selection_seconds")[1].ravel() for _, r in pairs]
                ).T,
                marker="o",
                ms=4,
                capsize=2,
                ls=line_style[method],
                color=COLORS[method],
                label=LABEL[method],
                lw=1.5,
            )
            ax.set(
                title=TITLES[dataset],
                xlabel="Number of available coordinates D",
                ylabel="Mean runtime (seconds)",
                yscale="log",
                xticks=dims,
            )
            clean(ax)
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, out / "optimization_runtime")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
    for method in methods:
        for ax, dataset in zip(axes, ["mean", "mixture"]):
            pairs = [
                (d, lookup(summary, "computational", dataset, method, d)) for d in dims
            ]
            pairs = [(d, r) for d, r in pairs if r]
            ax.plot(
                [d for d, r in pairs],
                [r["solved_fraction"] for d, r in pairs],
                marker="o",
                ms=4,
                color=COLORS[method],
                label=LABEL[method],
                ls=line_style[method],
            )
            ax.set(
                title=TITLES[dataset],
                xlabel="Available coordinates D",
                ylabel="Fraction certified optimal",
                ylim=(-0.03, 1.04),
                xticks=dims,
            )
            clean(ax)
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, out / "optimization_success")


def supplementary(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    for ax, dataset in zip(axes, ["mean", "mixture"]):
        for method in ["risk", "risk_ratio"]:
            pairs = [
                (d, lookup(summary, "ratio", dataset, method, d)) for d in [8, 12, 20]
            ]
            pairs = [(d, r) for d, r in pairs if r]
            if pairs:
                ax.errorbar(
                    [d for d, r in pairs],
                    [r["selection_seconds_mean"] for d, r in pairs],
                    yerr=np.array(
                        [interval(r, "selection_seconds")[1].ravel() for _, r in pairs]
                    ).T,
                    marker="o",
                    capsize=2,
                    ls="--" if method == "risk" else "-",
                    color=COLORS[method],
                    label="Tangents" if method == "risk" else LABEL[method],
                )
        ax.set(
            title=TITLES[dataset],
            xlabel="Available coordinates D",
            ylabel="Mean runtime (s)",
            yscale="log",
            xticks=[8, 12, 20],
        )
        clean(ax)
    axes[1].legend(frameon=False)
    save(fig, out / "ratio_supplement")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    for ax, dataset in zip(axes, ["mean", "mixture"]):
        for metric, lab, color in [
            ("power", "Equality-test power", COLORS["risk"]),
            ("robust_balanced_error", "Individual balanced error", COLORS["greedy"]),
        ]:
            pairs = [
                (t, lookup(summary, "sensitivity", dataset, "risk", theta=t))
                for t in [0.1, 0.25, 0.5]
            ]
            pairs = [(t, r) for t, r in pairs if r]
            if pairs:
                ax.errorbar(
                    [t for t, r in pairs],
                    [r[metric + "_mean"] for t, r in pairs],
                    yerr=np.array([interval(r, metric)[1].ravel() for t, r in pairs]).T,
                    marker="o",
                    capsize=2,
                    color=color,
                    label=lab,
                )
        ax.set(
            title=TITLES[dataset],
            xlabel="Wasserstein radius per class",
            ylabel="Probability",
            ylim=(0, 1),
            xticks=[0.1, 0.25, 0.5],
        )
        clean(ax)
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, out / "radius_sensitivity")


def real_support(root, out):
    records = [
        json.loads(p.read_text()) for p in (root / "real/records").glob("*.json")
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.2), layout="constrained")
    for ax, dataset in zip(axes, ["wdbc", "ionosphere"]):
        chosen = [
            r
            for r in records
            if r["dataset"] == dataset
            and r["method"] == "risk"
            and r["run_status"] == "complete"
        ]
        if not chosen:
            continue
        dimension = chosen[0]["dimension"]
        counts = np.zeros(dimension)
        for r in chosen:
            counts[r["selection"]["support"]] += 1
        ax.bar(
            np.arange(dimension) + 1,
            counts / len(chosen),
            color=COLORS["risk"],
            width=0.78,
        )
        ax.set(
            title=dataset.upper() + " — global selection",
            ylabel="Selection frequency",
            xlabel="Original coordinate number",
            ylim=(0, 1),
            xticks=np.arange(1, dimension + 1, 3),
        )
        clean(ax)
    save(fig, out / "real_support_frequency")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/research_20261003"))
    args = parser.parse_args()
    out = Path("reports/figures")
    out.mkdir(parents=True, exist_ok=True)
    style()
    summary = json.loads((args.root / "summary.json").read_text())
    analytic(out)
    split_diagram(out)
    testing(summary, out)
    recovery(summary, out)
    stress(summary, out)
    computational(summary, out)
    supplementary(summary, out)
    real_support(args.root, out)
    print("Saved vector PDF/SVG and inspection PNG figures to", out)


if __name__ == "__main__":
    main()
