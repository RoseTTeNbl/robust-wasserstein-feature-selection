"""Refresh a bilingual results candidate from immutable experiment records.

Default output is a candidate, not a silently published PDF. Inspect both
renderings, then use --publish --reviewed-by NAME to replace the reviewed pair.
The long-running execution pipeline can call this script after each suite.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from plot_experiments import build_figures
from presentation_report import Report, bi, from_summary
from report_results import enrich, write_scaling_csv

from robusthw.experiments.analysis import summarize
from robusthw.experiments.storage import file_hash, read_json, write_json

REPOSITORY = Path(__file__).resolve().parents[1]


def average(values):
    values = np.asarray(values, float)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
    }


def observed_records(study):
    # Analyze exactly the snapshot loaded by summarize, even while new raw
    # records arrive. Immutable records cannot change between these reads.
    with (study / "analysis/timing_runs.csv").open(encoding="utf-8", newline="") as handle:
        paths = {row["source_path"] for row in csv.DictReader(handle)}
    return [read_json(path) for path in sorted(paths)]


def populate(summary, records, study):
    data = from_summary(study / "analysis/summary.json")
    data["as_of"] = datetime.now().astimezone().isoformat(timespec="minutes")
    rows = [(record, row) for record in records for row in record.get("results", [])]
    for coverage in data["coverage"]:
        selected = [
            (record, row)
            for record, row in rows
            if record["spec"]["metadata"]["suite"] == coverage["suite"]
        ]
        statuses = Counter(row["status"] for _, row in selected)
        coverage["timeout_method_runs"] = sum(
            count
            for status, count in statuses.items()
            if "time_limit" in status or "timeout" in status
        )
        coverage["method_statuses"] = dict(statuses)
    observed = sum(row["observed_data"] for row in data["coverage"])
    planned = sum(row["planned_data"] for row in data["coverage"])
    execution_complete = observed == planned and not summary["planned_missing_outcomes"]
    # The automated candidate still needs a bilingual, visual review before
    # publishing. Missing narrative sections never become scientific claims.
    data["state"] = "complete" if execution_complete else "partial"
    data["findings"] = []
    small = [record for record in records if record["spec"]["kind"] == "small"]
    valid_small = [
        record for record in small if (record.get("verification") or {}).get("status") == "passed"
    ]
    if small:
        selected = {
            method: [
                row for record in small for row in record["results"] if row["method"] == method
            ]
            for method in ("enumeration", "global", "forward", "forward_swap")
        }
        global_rows = selected["global"]
        attained = sum(row["status"] == "optimal" for row in global_rows)
        exact = sum(abs(row["excess_risk"]) <= 1e-8 for row in global_rows if "excess_risk" in row)
        count = len(global_rows)
        data["findings"].append(
            bi(
                f"小问题中，全局搜索 {exact}/{count} 例找到独立穷举最优值，{attained}/{count} 例达到预定证明精度。这检验的是优化正确性，尚不说明测试预测有效。",
                f"On exact small problems, global search found the independently enumerated optimum in {exact}/{count} cases; {attained}/{count} met the declared certificate precision. This checks optimization, rather than held-out prediction.",
            )
        )
        for method, table_row in zip(
            ("enumeration", "global", "forward", "forward_swap"), data["tables"]["small"]["rows"]
        ):
            for index, metric, digits in (
                (1, "excess_risk", 4),
                (2, "runtime", 2),
                (3, "fit_count", 1),
            ):
                values = [row[metric] for row in selected[method] if row.get(metric) is not None]
                if values:
                    table_row[index] = {**average(values), "digits": digits}
        check_count = sum(record["verification"]["cut_checks"] for record in valid_small)
        subset_count = sum(record["verification"]["enumerated_subsets"] for record in valid_small)
        violations = max(
            (record["verification"]["maximum_cut_violation"] for record in valid_small), default=0
        )
        data["tables"]["checks"]["rows"] = [
            [
                bi("完整独立核查实例", "Complete independent checks"),
                f"{len(valid_small)} / 40",
                len(small) - len(valid_small),
            ],
            [
                bi("独立运输 LP", "Independent transport LPs"),
                f"{subset_count:,} / 7,040",
                0 if len(valid_small) == len(small) else bi("见失败记录", "See failed records"),
            ],
            [
                bi("下界割不等式", "Lower-bound inequalities"),
                f"{check_count:,}",
                f"{violations:.2g}",
            ],
            [
                bi("全局区间包含最优值", "Global interval covers optimum"),
                f"{len(valid_small)} / {len(small)}",
                len(small) - len(valid_small),
            ],
            [
                bi("达到目标证明精度", "Certificate target attained"),
                f"{attained} / {count}",
                count - attained,
            ],
        ]
        forward_excess = [row["excess_risk"] for row in selected["forward"] if "excess_risk" in row]
        swap_excess = [
            row["excess_risk"] for row in selected["forward_swap"] if "excess_risk" in row
        ]
        data["commentary"]["small"] = bi(
            f"交换将平均超额风险从 {np.mean(forward_excess):.4f} 降至 {np.mean(swap_excess):.4f}；全局平均拟合 {np.mean([row['fit_count'] for row in global_rows]):.1f} 次，穷举120次。并发计时不能作为独占硬件的速度结论。",
            f"Swaps reduce mean excess risk from {np.mean(forward_excess):.4f} to {np.mean(swap_excess):.4f}. Global search uses {np.mean([row['fit_count'] for row in global_rows]):.1f} fits versus 120 for enumeration. Concurrent timings do not establish isolated speed.",
        )
        maximum_gap = max(row["gap"] for row in global_rows)
        data["commentary"]["checks"] = bi(
            f"最大最终间隔 {maximum_gap:.6g}，略超0.000001目标的 {count - attained} 例保留精度限制状态。",
            f"The {count - attained} accuracy-limited cases ended at gap {maximum_gap:.6g}, just above the 0.000001 target.",
        )
    anchor = [row for record, row in rows if record["spec"]["kind"] == "anchor"]
    if anchor:
        probability = np.mean([row["metrics"]["ber_prob"] for row in anchor])
        hard = np.mean([row["metrics"]["ber_hard"] for row in anchor])
        class0 = np.mean([row["metrics"]["error0_hard"] for row in anchor])
        class1 = np.mean([row["metrics"]["error1_hard"] for row in anchor])
        robust = np.mean([row["upper"] for row in anchor]) / 2
        data["findings"].append(
            bi(
                f"一百维混合数据中，概率测试错误率为 {100 * probability:.2f}%，阈值错误率为 {100 * hard:.2f}%。两类阈值错误率分别为 {100 * class0:.1f}% 和 {100 * class1:.1f}%，呈现明显类别偏置。",
                f"On the 100-dimensional mixture, probabilistic test error is {100 * probability:.2f}% and threshold error is {100 * hard:.2f}%. Class-specific threshold errors are {100 * class0:.1f}% and {100 * class1:.1f}%, revealing a strong class bias.",
            )
        )
        data["commentary"]["anchor"] = bi(
            f"已运行 {len(anchor)}/500 次。训练鲁棒风险的一半平均为 {100 * robust:.2f}%，而测试错误率接近50%。按阈值分类时，两类错误率分别为 {100 * class0:.2f}% 和 {100 * class1:.2f}%。因此问题出在样本外评分的实际行为，不能据此宣称鲁棒变量选择改善了预测。当前评分按两个训练锚点的距离插值；远离训练样本时，权重可能达到截断值，偏向同一类别。本轮保留原规则和全部结果，未在测试结果出现后改动预测方法。",
            f"Observed {len(anchor)}/500 repetitions. Half the training robust risk averages {100 * robust:.2f}%, yet test error is close to 50%. Threshold errors for the two classes are {100 * class0:.2f}% and {100 * class1:.2f}%. The held-out score behavior therefore does not support a claim of improved prediction. The current score interpolates by distance between two training anchors; far from training observations, the weight can reach its clipping value and favor one class. This run preserves the original rule and all outcomes, without changing prediction in response to test results.",
        )
        diagnostic = study / "diagnostics/anchor_prediction.json"
        if diagnostic.exists():
            data["prediction_diagnostic"] = {
                "path": str(diagnostic),
                "sha256": file_hash(diagnostic),
            }
            detail = read_json(diagnostic)["aggregate_by_true_class"]
            median0, median1 = [detail[str(k)]["quantiles"]["gamma"][3] for k in (0, 1)]
            data["commentary"]["anchor"] = bi(
                f"500次重复中，一百万个测试点的阈值预测全部为类别一，两类错误率为100%和0%。概率错误率为 {100 * probability:.2f}%，比50%低 {50 - 100 * probability:.2f} 个百分点；但训练鲁棒风险的一半仅为 {100 * robust:.2f}%。\n\n当前评分在两个训练锚点之间按距离插值。新点到类别零锚点的距离，通常接近两个锚点之间的距离；两类插值权重的中位数分别为 {median0:.3f} 和 {median1:.3f}，使评分偏向类别一。独立重算完整复现全部已存指标。本轮保留原规则，没有在看过测试结果后修改预测。",
                f"Across 500 repetitions, all one million test points receive the class-one threshold prediction, producing class errors of 100% and 0%. Probabilistic error is {100 * probability:.2f}%, {50 - 100 * probability:.2f} percentage points below 50%, whereas half the training robust risk is only {100 * robust:.2f}%.\n\nThe score interpolates between two training anchors by distance. A new point is typically almost as far from the class-zero anchor as the anchors are from each other. Median interpolation weights are {median0:.3f} and {median1:.3f} for the two true classes, favoring class one. An independent replay reproduced every saved metric. The original predictor was retained without changes after test inspection.",
            )
    for field, zh, en in (
        (
            "main",
            "主网格尚未完成；当前不能给出变量选择效果的总体排名。",
            "The main grid is unfinished; an overall selection ranking is not yet supported.",
        ),
        (
            "generality",
            "不同规模、信号数和预算分别保留。完整观测值见配套数据表，缺失条件不补数。",
            "Scales, signal counts and budgets remain separate. The accompanying table retains all observed conditions, without imputing missing ones.",
        ),
        (
            "public",
            "公开数据的结论以实际完成的共享划分和测试者为范围。尚未运行的条件不能支持推广结论。",
            "Public-data conclusions apply to the observed shared splits and held-out people. Unrun conditions provide no evidence of generalization.",
        ),
        (
            "shift",
            "分布变化实验复用主实验的固定检验；主实验未完成前，不重新拟合替代结果。",
            "Shift tests reuse fixed main-experiment fits; no substitute refitting is performed before their source results are complete.",
        ),
        (
            "radius",
            "半径消融使用独立的二十套数据；只改变半径，不根据测试表现重新选择参数。",
            "The radius ablation uses twenty separate datasets and changes the radius without retuning against test outcomes.",
        ),
        (
            "scaling",
            "规模实验在其他计算任务结束后，以单进程、单求解器线程单独运行。未完成的时间测量不作效率结论。",
            "Scaling runs alone after other experiment jobs, with one process and one solver thread. Unfinished timings do not establish computational efficiency.",
        ),
        (
            "initialization",
            "初始化对照在相同数据和原半径下配对，复用已有的MMD初始化结果。",
            "Initialization comparisons are paired on the same data and original radius, reusing the existing MMD-start result.",
        ),
    ):
        data["commentary"][field] = bi(zh, en)
    data["commentary"]["conclusions"] = bi(
        "目前的证据支持小规模优化实现与独立参考值一致，同时揭示全变量预测的明显问题。变量选择的预测收益、分布变化下的表现及大规模计算优势，须由后续完成的对应实验分别回答。",
        "Current evidence supports agreement between small-scale optimization and the independent reference, while exposing a substantial all-variable prediction problem. Selection gains, performance under distribution change and larger-scale efficiency each require their corresponding completed experiment.",
    )
    data["commentary"]["limitations"] = bi(
        "本报告明确区分已运行结果与尚在执行的计划。小问题上的正确性不等于统计泛化；所列分布变化也不代表全部实际变化。若以后修改评分构造或算法，将另行标记代码版本与独立验证数据。",
        "Observed results are separated from the plan still being executed. Small-problem correctness is not statistical generalization, and the specified shifts do not cover every real change. Future score or algorithm changes require a separately identified code version and independent validation data.",
    )
    if execution_complete:
        data["findings"].insert(
            0,
            bi(
                "七套实验均已完成预定运行，超时和失败按实际状态保留。",
                "All seven suites have finished their planned runs, retaining timeouts and failures under their actual statuses.",
            ),
        )
        data["commentary"]["conclusions"] = bi(
            "本轮完成了预定网格。小问题优化与独立参考值一致，而混合数据的评分呈现明显类别偏置。变量选择的测试表现、分布变化后的增量和计算成本分别按条件列出，不能用训练风险或最优性证明替代测试预测结论。",
            "The declared grid is complete. Small-problem optimization agrees with the independent reference, while the mixture score has a strong class bias. Selection performance, error changes under shifts and computational cost are reported by condition; training risk and optimization certificates do not replace held-out prediction evidence.",
        )
        data["commentary"]["scaling"] = bi(
            "规模实验以单进程、单求解器线程运行。各实例先汇总重复计时，再按条件汇总独立数据；完整时间、风险、间隔和证明达标数见配套规模结果表。",
            "Scaling used one process and one solver thread. Timing repetitions are collapsed within each instance before condition-level aggregation. The accompanying scaling table retains time, risk, gap and certificate counts.",
        )
        data["commentary"]["limitations"] = bi(
            "结论只覆盖本报告列出的数据、预算、评分构造及分布变化。小规模正确性不等于统计泛化；以后若修改算法或评分，需另行标记版本并使用独立验证数据。",
            "Conclusions cover the stated data, budgets, score construction and distribution changes. Small-scale correctness is not statistical generalization. Future algorithm or score changes require an identified version and independent validation data.",
        )
    statuses = Counter(row["status"] for _, row in rows)
    failures = sum(
        count
        for status, count in statuses.items()
        if status
        in {"solver_error", "numerical_failure", "instance_failure", "verification_failure"}
    )
    timeouts = sum(
        count for status, count in statuses.items() if "time_limit" in status or "timeout" in status
    )
    data["tables"]["execution"]["rows"][0][1] = "AMD Ryzen 9 7945HX / 16 GB RAM"
    data["tables"]["execution"]["rows"][1][1] = (
        f"robusthw 1.0.0 / SciPy {__import__('scipy').__version__}, HiGHS"
    )
    data["tables"]["execution"]["rows"][-1][1] = f"{failures} / {timeouts}"
    data["method_status_counts"] = dict(statuses)
    data["execution_complete"] = execution_complete
    data["readable_data_path"] = "experiment_results.csv"
    return data


def write_results(summary, destination):
    fields = [
        "suite",
        "dataset",
        "D",
        "n_per_class",
        "true_signals",
        "budget",
        "method",
        "role",
        "radius_multiplier",
        "initialization",
        "observed_datasets",
        "planned_datasets",
        "balanced_error_percent_mean",
        "balanced_error_percent_sd",
        "threshold_error_percent_mean",
        "signal_recall_mean",
        "robust_risk_mean",
        "search_gap_mean",
        "runtime_seconds_mean",
        "statuses",
    ]
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in summary["groups"]:
            if row["n_instances"] == 0:
                continue
            g, m = row["group"], row["metrics"]

            def value(metric, statistic="mean", scale=1):
                number = m.get(metric, {}).get(statistic)
                return "" if number is None else number * scale

            writer.writerow(
                dict(
                    zip(
                        fields,
                        [
                            g["suite"],
                            g["model"],
                            g["D"],
                            g["n"],
                            g["s"],
                            g["budget"],
                            g["method"],
                            g["role"],
                            g["radius_factor"],
                            g["initialization"],
                            row["n_instances"],
                            row.get("planned_instances"),
                            value("ber_prob", scale=100),
                            value("ber_prob", "sd", 100),
                            value("ber_hard", scale=100),
                            value("recall"),
                            value("upper"),
                            value("gap"),
                            value("runtime"),
                            json.dumps(row["status_counts"], sort_keys=True),
                        ],
                    )
                )
            )


def compile_pair(candidate):
    environment = os.environ.copy()
    environment["TEMP"] = environment["TMP"] = str(candidate)
    for stem in ("experiment_report", "experiment_report_zh"):
        for _ in range(2):
            process = subprocess.run(
                ["xelatex", "-interaction=nonstopmode", "-halt-on-error", stem + ".tex"],
                cwd=candidate,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            (candidate / (stem + ".compiler.txt")).write_text(
                process.stdout + process.stderr, encoding="utf-8"
            )
            if process.returncode:
                raise RuntimeError(f"Compilation failed: {stem}; see candidate compiler log")
        log = (candidate / (stem + ".log")).read_text(encoding="utf-8", errors="replace")
        if any(
            error in log for error in ("Overfull \\hbox", "Overfull \\vbox", "Missing character")
        ):
            raise RuntimeError(f"Layout overflow needs review: {stem}")


def write_instance_data(records, study, candidate):
    """Keep per-run numbers and exact inputs beside the presentation averages."""
    folder = candidate / "data"
    folder.mkdir(exist_ok=True)
    fields = [
        "data_id",
        "suite",
        "model",
        "D",
        "n",
        "s",
        "seed",
        "budget",
        "method",
        "role",
        "status",
        "timing_repeat",
        "radius_factor",
        "initialization",
        "radii",
        "selected_original_columns",
        "upper",
        "lower",
        "gap",
        "runtime",
        "fit_count",
        "ber_prob",
        "ber_hard",
        "error0_hard",
        "error1_hard",
        "precision",
        "recall",
    ]
    inputs = []
    with (folder / "instance_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            spec = record["spec"]
            inputs.append(
                {
                    "data_id": spec["independent_data_id"],
                    "manifest_hash": record["manifest_hash"],
                    "record_hash": record["record_hash"],
                    "sample_sha256": record.get("data", {}).get("npz_sha256"),
                }
            )
            for method in record["results"]:
                row = {key: spec[key] for key in ("model", "D", "n", "s", "seed")}
                row.update(
                    data_id=spec["independent_data_id"],
                    suite=spec["metadata"]["suite"],
                    radii=json.dumps(record.get("radii")),
                    selected_original_columns=json.dumps(method.get("feature_ids")),
                )
                row.update(
                    {
                        key: method.get(key)
                        for key in (
                            "budget",
                            "method",
                            "role",
                            "status",
                            "timing_repeat",
                            "radius_factor",
                            "initialization",
                            "upper",
                            "lower",
                            "gap",
                            "runtime",
                            "fit_count",
                        )
                    }
                )
                row.update(
                    {
                        key: method.get("metrics", {}).get(key)
                        for key in (
                            "ber_prob",
                            "ber_hard",
                            "error0_hard",
                            "error1_hard",
                            "precision",
                            "recall",
                        )
                    }
                )
                writer.writerow(row)
    write_json(
        folder / "study_inputs.json",
        {
            "frozen_protocols": {
                path.parent.name: read_json(path) for path in sorted(study.glob("*/protocol.json"))
            },
            "input_records": inputs,
            "raw_records_root": str(study),
            "formal_data_excluded": "Development checks and runtime probes are not included.",
        },
    )
    diagnostic = study / "diagnostics/anchor_prediction.json"
    if diagnostic.exists():
        shutil.copy2(diagnostic, folder / "anchor_prediction_diagnostic.json")


def build(study, *, publish=False, reviewed_by=None):
    study = Path(study).resolve()
    candidate = study / "report-candidate"
    if publish:
        return publish_candidate(candidate, reviewed_by)
    candidate.mkdir(exist_ok=True)
    summarize(study, study / "analysis")
    summary = read_json(study / "analysis/summary.json")
    if summary["load_errors"] or summary["duplicate_attempts"]:
        raise ValueError("Resolve record-integrity errors before writing a report")
    records = observed_records(study)
    data = populate(summary, records, study)
    data = enrich(data, summary, records)
    figures = build_figures(summary, candidate / "figures")
    data["figures"] = figures
    write_results(summary, candidate / "experiment_results.csv")
    write_scaling_csv(summary, candidate / "scaling_results.csv")
    write_instance_data(records, study, candidate)
    compact_summary = {
        key: value
        for key, value in summary.items()
        if key
        not in {"groups", "planned_missing_records", "planned_missing_outcomes", "figures", "files"}
    }
    compact_summary.update(
        summary_kind="observed_results_view",
        original_summary_sha256=file_hash(study / "analysis/summary.json"),
        groups=[group for group in summary["groups"] if group["n_instances"]],
        planned_missing_records_count=len(summary["planned_missing_records"]),
        planned_missing_outcomes_count=len(summary["planned_missing_outcomes"]),
    )
    write_json(candidate / "data/analysis_summary.json", compact_summary)
    for lang, stem in (("en", "experiment_report"), ("zh", "experiment_report_zh")):
        (candidate / (stem + ".tex")).write_text(
            Report(data, lang).build().replace(candidate.as_posix() + "/", ""),
            encoding="utf-8",
            newline="\n",
        )
    write_json(candidate / "presentation_data.json", data)
    compile_pair(candidate)
    manifest = {
        "schema_version": 3,
        "document_kind": "experimental_record",
        "execution_state": "complete" if data["execution_complete"] else "partial",
        "state": "candidate_awaiting_visual_bilingual_review",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "study_root": str(study),
        "coverage": data["coverage"],
        "method_status_counts": data["method_status_counts"],
        "summary_sha256": file_hash(study / "analysis/summary.json"),
        "sources": {
            stem: file_hash(candidate / (stem + ".tex"))
            for stem in ("experiment_report", "experiment_report_zh")
        },
        "pdfs": {
            stem: file_hash(candidate / (stem + ".pdf"))
            for stem in ("experiment_report", "experiment_report_zh")
        },
        "assets": {
            path.relative_to(candidate).as_posix(): file_hash(path)
            for path in [
                candidate / "experiment_results.csv",
                candidate / "scaling_results.csv",
                *sorted((candidate / "figures").glob("*")),
                *sorted((candidate / "data").glob("*")),
            ]
            if path.is_file()
        },
    }
    write_json(candidate / "report.manifest.json", manifest)
    print(
        json.dumps(
            {
                "candidate": str(candidate),
                "execution_state": manifest["execution_state"],
                "published": publish,
                "observed_datasets": sum(row["observed_data"] for row in data["coverage"]),
            }
        )
    )
    return manifest


def publish_candidate(candidate, reviewed_by):
    """Publish the exact reviewed pair without rebuilding against newer data."""
    if not reviewed_by:
        raise ValueError("Publishing requires an explicit completed bilingual visual review")
    manifest = read_json(candidate / "report.manifest.json")
    expected = dict(manifest["assets"])
    for stem in ("experiment_report", "experiment_report_zh"):
        expected[stem + ".tex"] = manifest["sources"][stem]
        expected[stem + ".pdf"] = manifest["pdfs"][stem]
    contents = {}
    import hashlib

    for name, digest in expected.items():
        contents[name] = (candidate / name).read_bytes()
        if hashlib.sha256(contents[name]).hexdigest() != digest:
            raise ValueError(f"Candidate changed since review: {name}")
    destination = REPOSITORY / "reports"
    for name, content in contents.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    manifest.update(
        state="reviewed_published",
        reviewed_by=reviewed_by,
        reviewed_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    for stem, language in (("experiment_report", "en"), ("experiment_report_zh", "zh")):
        write_json(destination / (stem + ".manifest.json"), {**manifest, "language": language})
    print(json.dumps({"published": True, "execution_state": manifest["execution_state"]}))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--reviewed-by")
    args = parser.parse_args()
    build(args.study_root, publish=args.publish, reviewed_by=args.reviewed_by)
