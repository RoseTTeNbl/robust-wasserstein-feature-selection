"""Matched English/Chinese paper figures from the frozen experimental evidence."""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("report_builder", Path(__file__).with_name("build_report.py"))
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)
METHODS = ("risk", "greedy", "mmd", "wang", "l1", "random")
MECHANISMS = ("mean", "variance", "dependence", "mixture")
COLORS = {"risk": "#28698E", "risk_no_root": "#79A7BB", "risk_ratio": "#1C465F",
          "greedy": "#C98743", "mmd": "#4E846E", "wang": "#8A7594",
          "l1": "#887C69", "random": "#94999C", "box": "#848C93", "triangle": "#C98743",
          "signal": "#42484B", "all": "#B19B73"}
MARKERS = dict(zip(METHODS, ("o", "s", "^", "D", "v", "P")))
MARKERS.update(signal="*", all="X")
EN = {"risk": "Robust global", "greedy": "Robust greedy", "mmd": "MMD screening",
      "wang": "Additive MMD", "l1": "L1 logistic", "random": "Random",
      "signal": "True support", "all": "All features",
      "risk_no_root": "Risk, no roots", "box": "Box", "triangle": "Triangle",
      "mean": "Mean", "variance": "Variance", "dependence": "Dependence", "mixture": "Mixture"}
ZH = {"risk": "鲁棒全局", "greedy": "鲁棒贪心", "mmd": "MMD 筛选", "wang": "加性 MMD",
      "l1": "L1 逻辑回归", "random": "随机", "signal": "真实信号变量", "all": "全部变量",
      "risk_no_root": "风险（无根节点）", "box": "逐项价格界", "triangle": "总价格界",
      "mean": "均值", "variance": "方差", "dependence": "相关结构", "mixture": "混合"}


def words(language, en, zh):
    return zh if language == "zh" else en


def label(key, language):
    return (ZH if language == "zh" else EN).get(key, key)


def style(language):
    plt.rcParams.update({"font.family": ["Times New Roman", "SimSun"], "font.size": 10,
                         "axes.labelsize": 10, "axes.titlesize": 11, "xtick.labelsize": 9,
                         "ytick.labelsize": 10, "legend.fontsize": 10, "mathtext.fontset": "stix",
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .8, "savefig.bbox": "tight", "axes.unicode_minus": False})


def save(fig, out, stem, language):
    stem += "_zh" if language == "zh" else ""
    for extension in ("pdf", "svg", "png"):
        fig.savefig(out / (stem + "." + extension), dpi=220)
    plt.close(fig)


def clean(ax, axis="y"):
    ax.grid(axis=axis, alpha=.17, linewidth=.65)
    ax.set_axisbelow(True)


def interval(row, field):
    mean = REPORT.number(row, field + "_mean")
    if mean is None:
        return None
    lo, hi = REPORT.number(row, field + "_lo"), REPORT.number(row, field + "_hi")
    return mean, np.array([[max(0., mean-(lo if lo is not None else mean))],
                           [max(0., (hi if hi is not None else mean)-mean)]])


def _support_panels(e, field, methods, language, xlabel, xlim):
    """One question per figure; common axes and explicit diagnostic rows."""
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.0), layout="constrained", sharey=True)
    for index, dataset in enumerate(MECHANISMS):
        ax = axes.flat[index]
        for j, method in enumerate(methods):
            summary = e.lookup("statistical", dataset, method)
            value = interval(summary, field) if summary else None
            if value is None:
                continue
            mean, err = value
            ax.errorbar(mean, j, xerr=err, marker=MARKERS[method], color=COLORS[method],
                        markersize=4.5, capsize=2, linewidth=1.1)
        ax.set(title=label(dataset, language), yticks=range(len(methods)),
               ylim=(len(methods)-.5, -.5), xlim=xlim, xlabel=xlabel)
        if index % 2 == 0:
            ax.set_yticklabels([label(method, language) for method in methods])
        ax.tick_params(labelleft=index % 2 == 0)
        if len(methods) > len(METHODS):
            ax.axhline(len(METHODS)-.5, color="#BFC3C6", linewidth=.65)
        clean(ax, "x")
    return fig, axes


def testing(e, out, language):
    fig, _ = _support_panels(e, "power", (*METHODS, "signal", "all"), language,
                            words(language, "Testing power (higher is better)", "检验功效（越大越好）"),
                            (-.02, 1.03))
    save(fig, out, "power_and_calibration", language)


def recovery_prediction(e, out, language):
    fig, axes = _support_panels(e, "fdp", METHODS, language,
                               words(language, "FDP = NDP (lower is better)", "FDP = NDP（越小越好）"),
                               (-.02, 1.03))
    for ax in axes.flat:
        ax.axvline(0, color="#42484B", linestyle=":", linewidth=1.)
        ax.axvline(.75, color="#94999C", linestyle="--", linewidth=.85)
    save(fig, out, "recovery_and_prediction", language)

    fig, axes = _support_panels(e, "svm_balanced_error", METHODS, language,
                               words(language, "Balanced error (lower is better)", "平衡错误率（越小越好）"),
                               (-.02, .57))
    for ax in axes.flat:
        ax.axvline(.5, color="#94999C", linestyle=":", linewidth=1.)
    save(fig, out, "prediction_error", language)


def optimization(e, out, language):
    methods = ("box", "triangle", "risk_no_root", "risk")
    markers = ("s", "^", "D", "o")
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.0), layout="constrained")
    for column, dataset in enumerate(("mean", "mixture")):
        for method, marker in zip(methods, markers):
            rows = sorted(e.select("computational", dataset, method), key=lambda r: int(r["dimension"]))
            xs, times, lows, highs, solved = [], [], [], [], []
            for r in rows:
                v = interval(r, "selection_seconds")
                if v is None:
                    continue
                mean, err = v
                counts = json.loads(r["status_counts"]) if isinstance(r["status_counts"], str) else r["status_counts"]
                xs.append(int(r["dimension"]))
                times.append(mean)
                lows.append(err[0, 0])
                highs.append(err[1, 0])
                solved.append(counts.get("optimal", 0) / int(r["n_total"]))
            if not xs:
                continue
            positions = np.arange(len(xs))
            axes[0, column].errorbar(positions, times, yerr=np.array([lows, highs]),
                                    marker=marker, markersize=4, capsize=2, linewidth=1.15,
                                    color=COLORS[method], label=label(method, language))
            axes[1, column].plot(positions, solved, marker=marker, markersize=4,
                                 linewidth=1.15, color=COLORS[method])
            for ax in axes[:, column]:
                ax.set_xticks(positions, xs)
        axes[0, column].set(title=label(dataset, language), ylabel=words(language, "Charged time (s)", "累计时间（秒）"))
        axes[0, column].set_yscale("log")
        axes[1, column].set(ylabel=words(language, "Solved fraction", "解出比例"), xlabel=words(language, "Dimension D", "维度 D"), ylim=(-.03, 1.05))
        for ax in axes[:, column]:
            clean(ax)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside upper center", ncol=4, frameon=False)
    save(fig, out, "optimization_comparison", language)


def confirmation(e, out, language):
    if e.confirmation is None:
        return
    stage = e.confirmation
    ordered = sorted(stage["raw"], key=lambda case: (
        {"overlap": 0, "logistic": 1}[stage["raw"][case]["case"]["loss"]],
        stage["raw"][case]["case"]["dimension"],
        stage["raw"][case]["case"]["seed"], tuple(stage["raw"][case]["case"]["theta"])))
    by_case = {case: {r["method"]: r for r in stage["rows"] if r["case"] == case}
               for case in ordered}
    fig, axes = plt.subplots(2, 1, figsize=(7.1, 4.3), layout="constrained", sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1.]})
    palette = {"master": "#92999C", "baseline": "#C98743", "curvature": "#28698E"}
    symbols = {"master": "^", "baseline": "o", "curvature": "D"}
    names = {"master": words(language, "Master each step", "每步主问题"),
             "baseline": words(language, "Original GP", "原 GP"),
             "curvature": words(language, "Stored fractions", "保留分式")}
    for method, shift in (("master", -.12), ("baseline", 0.), ("curvature", .12)):
        ys = [by_case[case][method]["completed_evaluations"] for case in ordered]
        axes[0].scatter(np.arange(1, len(ordered)+1)+shift, ys, s=23, marker=symbols[method],
                        color=palette[method], label=names[method], zorder=4)
    for i, case in enumerate(ordered, 1):
        a, b = by_case[case]["curvature"], by_case[case]["baseline"]
        axes[0].plot([i, i+.12], [b["completed_evaluations"], a["completed_evaluations"]],
                     color="#647B88", linewidth=.8, alpha=.6, zorder=2)
        axes[1].scatter(i, a["seconds"]/b["seconds"], marker="D", color=palette["curvature"],
                        s=25, zorder=4)
    axes[0].set(ylabel=words(language, "Complete evaluations", "完整评估次数"), ylim=(0, None))
    axes[0].legend(loc="upper left", ncol=3, frameon=False)
    axes[1].axhline(1., color="#5C6063", linewidth=1., linestyle=":")
    axes[1].set(ylabel=words(language, "Time / original GP", "时间 / 原 GP"),
                xlabel=words(language, "Paired case", "配对实例"), xticks=range(1, len(ordered)+1),
                xlim=(.5, len(ordered)+.5), ylim=(.5, 1.55))
    for ax in axes:
        ax.axvline(8.5, color="#A6ADB0", linestyle="--", linewidth=.8)
        clean(ax)
    fig.supxlabel(words(language, "Half hinge: cases 1--8     |     Logistic: cases 9--16",
                        "半合页：实例 1--8     |     逻辑损失：实例 9--16"), fontsize=9)
    save(fig, out, "search_confirmation", language)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--root", type=Path, default=PROJECT / "results/research_20261003")
    parser.add_argument("--output", type=Path, default=PROJECT / "reports/figures")
    parser.add_argument("--language", choices=("en", "zh", "both"), default="both")
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args(argv)
    evidence = REPORT.Evidence(args.project_root.resolve(), args.root.resolve(), args.draft)
    args.output.mkdir(parents=True, exist_ok=True)
    for language in (("en", "zh") if args.language == "both" else (args.language,)):
        style(language)
        testing(evidence, args.output, language)
        recovery_prediction(evidence, args.output, language)
        optimization(evidence, args.output, language)
        if args.validation_root:
            validation = REPORT.SearchEvidence(args.validation_root.resolve(), args.draft)
            confirmation(validation, args.output, language)
    print(json.dumps({"figures": str(args.output), "reference_records": len(evidence.rows), "language": args.language}))


if __name__ == "__main__":
    main()
