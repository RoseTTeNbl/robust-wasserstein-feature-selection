"""A small set of presentation figures from verified, condition-separated summaries.

The statistical summary remains the source of truth. This module only chooses
predeclared views, labels them, and writes matching English/Chinese vector PDFs.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ("mean", "variance", "correlation", "mixture")
MODEL_NAMES = {
    "en": (
        "Mean difference",
        "Variance difference",
        "Correlation difference",
        "Mixture difference",
    ),
    "zh": ("均值不同", "波动不同", "相关不同", "混合不同"),
}
METHOD_NAMES = {
    "global": ("Global search", "全局搜索"),
    "forward": ("Forward", "前向选择"),
    "forward_swap": ("Forward + swaps", "前向加交换"),
    "marginal_mmd": ("Marginal MMD", "逐变量 MMD"),
    "l1_logistic": ("L1 logistic", "L1 逻辑回归"),
    "mrmr": ("mRMR", "mRMR"),
    "joint_mmd": ("Joint MMD", "联合 MMD"),
    "random": ("Random", "均匀随机"),
    "enumeration": ("Enumeration", "穷举"),
}
HIGHLIGHT = ("global", "forward_swap", "joint_mmd", "random")
COLORS = {
    "global": "#315F7C",
    "forward_swap": "#709079",
    "joint_mmd": "#B97B48",
    "random": "#8C8C8C",
}
MARKERS = {"global": "o", "forward_swap": "s", "joint_mmd": "^", "random": "x"}


def style(language):
    plt.rcParams.update(
        {
            "font.family": "SimSun" if language == "zh" else "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 180,
        }
    )


def text(en, zh, language):
    return zh if language == "zh" else en


def name(method, language):
    return METHOD_NAMES.get(method, (method, method))[language == "zh"]


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


def save(fig, output, key, language):
    path = output / f"{key}_{language}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path.resolve().as_posix()


def synthetic(summary, output, language):
    selected = groups(summary, suite="main", D=50, n=20, s=4, role="comparison")
    if not selected:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharey=True, layout="constrained")
    for model, title, ax in zip(MODELS, MODEL_NAMES[language], axes.flat):
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
                    linewidth=1.5,
                    capsize=2,
                    label=name(method, language),
                )
        ax.axhline(50, color="0.7", linewidth=0.8, linestyle="--")
        ax.axhline(20, color="0.7", linewidth=0.8, linestyle=":")
        ax.set(
            title=title,
            xticks=[2, 4, 8],
            ylim=(0, 55),
            xlabel=text("Selected variables", "所选变量数", language),
        )
    for ax in axes[:, 0]:
        ax.set_ylabel(text("Balanced error (%)", "两类平均错误率（%）", language))
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    return save(fig, output, "main", language)


def public(summary, output, language):
    datasets = ("wdbc", "ionosphere", "sonar", "har")
    selected = [
        row
        for row in groups(summary, suite="public", role="comparison")
        if (row["group"]["n"], row["group"]["budget"])
        == ((40, 16) if row["group"]["model"] == "har" else (20, 4))
    ]
    if not selected:
        return None
    fig, axes = plt.subplots(1, 4, figsize=(8.2, 4.2), sharey=True, layout="constrained")
    methods = tuple(METHOD_NAMES)[:8]
    for dataset, ax in zip(datasets, axes):
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
                    fmt="o",
                    color=COLORS.get(method, "#777777"),
                    capsize=2,
                    markersize=4,
                )
        ax.set(
            title=dataset.upper() if dataset in ("wdbc", "har") else dataset.title(),
            xlim=(0, 55),
            xlabel=text("Error (%)", "错误率（%）", language),
        )
        ax.axvline(50, color="0.7", linestyle="--", linewidth=0.8)
        ax.set_yticks(range(8), [name(method, language) for method in methods])
    axes[0].invert_yaxis()
    return save(fig, output, "public", language)


def shift(summary, output, language):
    rows = summary.get("shift_groups", [])
    if not rows:
        return None
    fig, axes = plt.subplots(
        2, 4, figsize=(9.0, 5.0), sharex=True, sharey=True, layout="constrained"
    )
    for col, (model, title) in enumerate(zip(MODELS, MODEL_NAMES[language])):
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
                    ],
                    key=lambda row: row["group"]["delta_factor"],
                )
                if points:
                    means = [measure(row, "ber_prob")[0] for row in points]
                    ax.plot(
                        [row["group"]["delta_factor"] for row in points],
                        np.array(means) * 100,
                        color=COLORS[method],
                        marker=MARKERS[method],
                        linewidth=1.4,
                        label=name(method, language),
                    )
            ax.set(xticks=[0, 1, 2], ylim=(0, 60))
            if rownum == 0:
                ax.set_title(title)
            if rownum == 1:
                ax.set_xlabel(text("Shift / mean radius", "变化量 / 平均半径", language))
    axes[0, 0].set_ylabel(text("Translation\nError (%)", "整体平移\n错误率（%）", language))
    axes[1, 0].set_ylabel(text("Added noise\nError (%)", "增加噪声\n错误率（%）", language))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    return save(fig, output, "shift", language)


def search(summary, output, language):
    selected = groups(summary, suite="small", role="comparison")
    if not selected:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), layout="constrained")
    methods = ("enumeration", "global", "forward", "forward_swap")
    for i, method in enumerate(methods):
        rows = [row for row in selected if row["group"]["method"] == method]
        for j, model in enumerate(MODELS):
            family = [row for row in rows if row["group"]["model"] == model]
            if family:
                vals = [measure(row, "excess_risk")[0] for row in family]
                vals = [v for v in vals if v is not None]
                if vals:
                    axes[0].scatter(
                        np.repeat(i + (j - 1.5) * 0.12, len(vals)),
                        np.maximum(vals, 0),
                        s=25,
                        color=("#315F7C", "#B97B48", "#709079", "#888888")[j],
                        label=MODEL_NAMES[language][j] if i == 0 else None,
                    )
        counts = [measure(row, "fit_count")[0] for row in rows]
        counts = [v for v in counts if v is not None]
        if counts:
            axes[1].bar(i, np.mean(counts), width=0.55, color=COLORS.get(method, "#9A9A9A"))
    for ax in axes:
        ax.set_xticks(
            range(4),
            [name(method, language).replace(" + ", "\n+ ") for method in methods],
            rotation=12,
            ha="right",
        )
    axes[0].set_ylabel(text("Risk above enumerated optimum", "比穷举最优值高出的风险", language))
    axes[1].set_ylabel(text("Fixed-subset fits (mean)", "固定选集拟合次数（均值）", language))
    axes[0].legend(frameon=False, fontsize=8)
    return save(fig, output, "search", language)


def anchor(summary, output, language):
    rows = groups(summary, suite="anchor", method="all_variables")
    if not rows:
        return None
    row = rows[0]
    metrics = ("upper", "ber_prob", "ber_hard", "error0_hard", "error1_hard")
    values = [measure(row, key)[0] for key in metrics]
    if any(value is None for value in values):
        return None
    values[0] /= 2
    labels = text(
        (
            "Training robust\nrisk / 2",
            "Test probability\nerror",
            "Test threshold\nerror",
            "Class 0\nthreshold error",
            "Class 1\nthreshold error",
        ),
        (
            "训练鲁棒\n风险 / 2",
            "测试概率\n错误率",
            "测试阈值\n错误率",
            "类别零\n阈值错误率",
            "类别一\n阈值错误率",
        ),
        language,
    )
    fig, ax = plt.subplots(figsize=(7.0, 3.3), layout="constrained")
    bars = ax.bar(
        range(5),
        100 * np.array(values),
        color=["#709079", "#315F7C", "#315F7C", "#B97B48", "#B97B48"],
        width=0.62,
    )
    ax.set(
        xticks=range(5),
        xticklabels=labels,
        ylim=(0, 110),
        ylabel=text("Percent (%)", "百分比（%）", language),
    )
    ax.axhline(50, color="0.6", linewidth=0.8, linestyle="--")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{100 * value:.1f}",
            ha="center",
            fontsize=10,
        )
    return save(fig, output, "anchor", language)


def build_figures(summary, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figures = {key: {} for key in ("main", "public", "shift", "search", "anchor")}
    for language in ("en", "zh"):
        style(language)
        for key, function in (
            ("main", synthetic),
            ("public", public),
            ("shift", shift),
            ("search", search),
            ("anchor", anchor),
        ):
            figures[key][language] = function(summary, output, language)
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
