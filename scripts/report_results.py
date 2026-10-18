"""Fill presentation tables from observed, condition-specific study results.

This helper never fits a model or changes records. ``enrich`` mutates only the
presentation dictionary; ``write_scaling_csv`` exports the tested analysis's
per-instance timing summaries, with repetitions kept out of dataset counts.
"""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

MODELS = ("mean", "variance", "correlation", "mixture")
LABELS = {
    "mean": ("均值", "Mean"),
    "variance": ("波动", "Variance"),
    "correlation": ("相关", "Correlation"),
    "mixture": ("混合", "Mixture"),
    "global": ("全局", "Global"),
    "forward_swap": ("前向加交换", "Forward + swaps"),
    "joint_mmd": ("联合 MMD", "Joint MMD"),
    "marginal_mmd": ("MMD 初始化", "MMD start"),
    "random": ("随机初始化", "Random start"),
    "translation": ("平移", "Translation"),
    "gaussian_noise": ("噪声", "Noise"),
}
CONDITION = ("manifest_hash", "model", "D", "n", "s", "budget")
DISPLAY_METHODS = ("global", "forward_swap", "joint_mmd")


def _bi(zh, en):
    return {"zh": zh, "en": en}


def _label(name, language):
    return LABELS.get(name, (name, name))[language == "en"]


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _stats(values, *, percent=False, digits=None):
    values = [float(value) for value in values if _finite(value)]
    if not values:
        return None
    return {
        "mean": mean(values),
        "sd": stdev(values) if len(values) > 1 else None,
        "n": len(values),
        "scale": 100 if percent else 1,
        "digits": (1 if percent else 4) if digits is None else digits,
    }


def _cell(group, metric="ber_prob", *, percent=True, digits=None):
    measure = (group or {}).get("metrics", {}).get(metric, {})
    if not _finite(measure.get("mean")) or not measure.get("n", 0):
        return None
    return {
        **{key: measure.get(key) for key in ("mean", "sd", "n")},
        "scale": 100 if percent else 1,
        "digits": (1 if percent else 4) if digits is None else digits,
    }


def _observed(summary, suite, **criteria):
    return [
        item
        for item in summary.get("groups", [])
        if item.get("n_instances", 0)
        and item["group"].get("suite") == suite
        and item["group"].get("role") == "comparison"
        and all(item["group"].get(key) == value for key, value in criteria.items())
    ]


def _key(group):
    return tuple(group.get(name) for name in CONDITION)


def _sort(key):
    _, model, D, n, s, budget = key
    return (MODELS.index(model) if model in MODELS else 99, D, n, s, budget, key[0] or "")


def _condition(group):
    return _bi(
        f"{_label(group['model'], 'zh')} D={group['D']}, n={group['n']}, s=d={group['s']}",
        f"{_label(group['model'], 'en')} D={group['D']}, n={group['n']}, s=d={group['s']}",
    )


def _fallback(row):
    fit = row.get("fit") or {}
    return bool(fit.get("diagnostics", {}).get("fallback")) or (
        row.get("fallback") == "constant_zero_score"
    )


def _generality(data, summary):
    conditions = defaultdict(dict)
    for item in _observed(summary, "main", radius_factor=1, initialization="marginal_mmd"):
        group = item["group"]
        if group["budget"] == group["s"] and group["method"] in DISPLAY_METHODS:
            conditions[_key(group)][group["method"]] = item
    rows, counts = [], []
    for key, methods in sorted(conditions.items(), key=lambda item: _sort(item[0])):
        cells = [_cell(methods.get(method)) for method in DISPLAY_METHODS]
        if not any(cells):
            continue
        group = next(iter(methods.values()))["group"]
        label = _condition(group)
        repetitions = [cell["n"] for cell in cells if cell]
        planned = [
            methods[method].get("planned_instances", 30)
            for method in DISPLAY_METHODS
            if method in methods
        ]
        if repetitions and len(set(repetitions)) == 1 and len(set(planned)) == 1:
            label["zh"] += f"；重复{repetitions[0]}/{planned[0]}"
            label["en"] += f"; {repetitions[0]}/{planned[0]} repeats"
        rows.append([label, *cells])
        counts.append(
            {
                "condition": dict(zip(CONDITION, key)),
                "methods": {
                    method: {
                        "observed": cell["n"] if cell else 0,
                        "planned": methods.get(method, {}).get("planned_instances", 30),
                    }
                    for method, cell in zip(DISPLAY_METHODS, cells)
                },
            }
        )
    data["tables"]["generality"]["rows"] = rows
    data["result_counts"]["generality"] = counts
    if rows:
        complete = sum(
            all(value["observed"] == value["planned"] for value in item["methods"].values())
            for item in counts
        )
        data["commentary"]["generality"] = _bi(
            f"表中已有 {len(rows)}/20 个预定条件，其中 {complete} 个条件的三种方法均记录完整。每行固定数据模型、候选维度、每类训练数及信号数；只比较预算等于信号数的结果。其余预算保留在完整数值表中，缺失观测不填零。",
            f"The table contains {len(rows)}/20 declared conditions; {complete} have complete observations for all three methods. Each row fixes the model, candidate dimension, training count per class and signal count, using a budget equal to the signal count. Other budgets remain in the complete numeric table; missing observations are not zero errors.",
        )


def _shift(data, records):
    values, observed = defaultdict(list), defaultdict(set)
    for record in records:
        spec = record["spec"]
        if spec.get("metadata", {}).get("suite", spec.get("kind")) != "shift":
            continue
        for row in record.get("results", []):
            if (
                row.get("role", "comparison") != "comparison"
                or row["method"] not in DISPLAY_METHODS
            ):
                continue
            for kind in ("translation", "gaussian_noise"):
                key = (record["manifest_hash"], spec["model"], kind)
                observed[key, row["method"]].add(spec["independent_data_id"])
                endpoints = {
                    shift["delta_factor"]: shift.get("metrics", {}).get("ber_prob")
                    for shift in row.get("shifts", [])
                    if shift["kind"] == kind and shift["delta_factor"] in (0, 2)
                }
                if all(_finite(endpoints.get(factor)) for factor in (0, 2)):
                    values[key, row["method"]].append(endpoints[2] - endpoints[0])
    keys = sorted(
        {key for key, _ in values}, key=lambda key: (MODELS.index(key[1]), key[2], key[0])
    )
    rows, counts, sentences = [], [], [[], []]
    for key in keys:
        _, model, kind = key
        cells = [_stats(values[key, method], percent=True) for method in DISPLAY_METHODS]
        rows.append(
            [
                _bi(*[f"{_label(model, lang)} / {_label(kind, lang)}" for lang in ("zh", "en")]),
                *cells,
            ]
        )
        counts.append(
            {
                "manifest_hash": key[0],
                "model": model,
                "change": kind,
                "methods": {
                    method: {
                        "paired": len(values[key, method]),
                        "observed": len(observed[key, method]),
                    }
                    for method in DISPLAY_METHODS
                },
            }
        )
        if cells[0]:
            increase, count = 100 * cells[0]["mean"], cells[0]["n"]
            sentences[0].append(
                f"{_label(model, 'zh')} / {_label(kind, 'zh')}：全局增量 {increase:+.2f} 个百分点（{count}/30 对）。"
            )
            sentences[1].append(
                f"{_label(model, 'en')} / {_label(kind, 'en')}: global change {increase:+.2f} percentage points ({count}/30 pairs)."
            )
    data["tables"]["shift"]["rows"] = rows
    data["result_counts"]["shift"] = counts
    if rows:
        data["commentary"]["shift"] = _bi(
            "".join(sentences[0])
            + "正数表示退化，负数表示在该变化下错误率下降；每对先作差，再计算均值和标准差，未配齐两个端点的观测不进入增量。",
            " ".join(sentences[1])
            + " Positive changes indicate degradation; negative changes indicate lower error under that change. Compute differences within pairs before their mean and SD; observations missing an endpoint do not enter the increase.",
        )


def _ablations(data, summary, records):
    radius, initialization = defaultdict(dict), defaultdict(dict)
    for item in _observed(summary, "ablation"):
        group = item["group"]
        if (group["D"], group["n"], group["s"], group["budget"]) != (50, 20, 4, 4):
            continue
        key = (group["manifest_hash"], group["model"], group["method"])
        if group["initialization"] == "marginal_mmd" and group["method"] in (
            "global",
            "forward_swap",
        ):
            radius[key][group["radius_factor"]] = item
        if group["method"] == "global" and group["radius_factor"] == 1:
            initialization[key][group["initialization"]] = item
    raw = [
        (record, row)
        for record in records
        if record["spec"].get("metadata", {}).get("suite", record["spec"].get("kind")) == "ablation"
        for row in record.get("results", [])
    ]
    radius_rows, radius_counts, radius_text = [], [], [[], []]
    for key, factors in sorted(radius.items()):
        cells = [_cell(factors.get(factor)) for factor in (0.5, 1, 2)]
        if not any(cells):
            continue
        _, model, method = key
        radius_rows.append(
            [
                _bi(*[f"{_label(model, lang)} / {_label(method, lang)}" for lang in ("zh", "en")]),
                *cells,
            ]
        )
        fallbacks = [
            sum(
                _fallback(row)
                for record, row in raw
                if record["manifest_hash"] == key[0]
                and record["spec"]["model"] == model
                and row["method"] == method
                and row.get("radius_factor", 1) == factor
                and row.get("initialization", "marginal_mmd") == "marginal_mmd"
            )
            for factor in (0.5, 1, 2)
        ]
        radius_counts.append(
            {
                "manifest_hash": key[0],
                "model": model,
                "method": method,
                "observed": [cell["n"] if cell else 0 for cell in cells],
                "fallback_runs": fallbacks,
            }
        )
        for index, lang in enumerate(("zh", "en")):
            shown = ", ".join(
                f"×{factor:g}: {100 * cell['mean']:.2f}% ({cell['n']}/10)"
                for factor, cell in zip((0.5, 1, 2), cells)
                if cell
            )
            radius_text[index].append(f"{_label(model, lang)} / {_label(method, lang)} — {shown}.")
    data["tables"]["radius"]["rows"] = radius_rows
    data["result_counts"]["radius"] = radius_counts
    if radius_rows:
        fallback_count = sum(sum(row["fallback_runs"]) for row in radius_counts)
        data["commentary"]["radius"] = _bi(
            " ".join(radius_text[0])
            + f" 原始记录明确标记的零评分兜底共 {fallback_count} 次；错误率50%本身不用于判定兜底。",
            " ".join(radius_text[1])
            + f" Raw records explicitly mark {fallback_count} zero-score fallbacks; error of 50% alone is not used to identify them.",
        )
    init_rows, init_text = [], [[], []]
    for key, starts in sorted(initialization.items()):
        _, model, _ = key
        for start in ("marginal_mmd", "random"):
            item = starts.get(start)
            if item is None:
                continue
            init_rows.append(
                [
                    _bi(
                        *[f"{_label(model, lang)} / {_label(start, lang)}" for lang in ("zh", "en")]
                    ),
                    _cell(item, "upper", percent=False),
                    _cell(item, "runtime", percent=False, digits=2),
                    _cell(item, "gap", percent=False),
                ]
            )
        pairs = defaultdict(dict)
        for record, row in raw:
            if (
                record["manifest_hash"] == key[0]
                and record["spec"]["model"] == model
                and row["method"] == "global"
                and row.get("radius_factor", 1) == 1
            ):
                pairs[record["spec"]["independent_data_id"]][
                    row.get("initialization", "marginal_mmd")
                ] = row
        differences = [
            right["random"]["upper"] - right["marginal_mmd"]["upper"]
            for right in pairs.values()
            if all(
                start in right and _finite(right[start].get("upper"))
                for start in ("marginal_mmd", "random")
            )
        ]
        if differences:
            init_text[0].append(
                f"{_label(model, 'zh')}：随机减MMD初始化的平均最终风险差为 {mean(differences):+.4f}（{len(differences)}/10 对）。"
            )
            init_text[1].append(
                f"{_label(model, 'en')}: mean final-risk difference, random minus MMD start, is {mean(differences):+.4f} ({len(differences)}/10 pairs)."
            )
    data["tables"]["initialization"]["columns"][0] = _bi("模型 / 初始化", "Model / initialization")
    data["tables"]["initialization"]["rows"] = init_rows
    if init_rows:
        data["commentary"]["initialization"] = _bi(
            "".join(init_text[0]) + "两种模型分别汇总；正的风险差表示随机起点返回的风险更高。",
            " ".join(init_text[1])
            + "The two models are summarized separately; a positive risk difference means the random start returned higher risk.",
        )


def _central_commentary(data, summary, suite):
    sentences = [[], []]
    for item in _observed(
        summary, suite, method="global", radius_factor=1, initialization="marginal_mmd"
    ):
        group, measure = item["group"], item["metrics"].get("ber_prob", {})
        central = (
            (group["D"], group["n"], group["s"], group["budget"]) == (50, 20, 4, 4)
            if suite == "main"
            else group["n"] == (40 if group["model"] == "har" else 20)
            and group["budget"] == (16 if group["model"] == "har" else 4)
        )
        if not central or not _finite(measure.get("mean")):
            continue
        for index, lang in enumerate(("zh", "en")):
            label = _label(group["model"], lang)
            condition = f"D={group['D']}, n={group['n']}, d={group['budget']}"
            count = measure["n"]
            if lang == "zh":
                unit = (
                    "名测试者"
                    if group["kind"] == "har"
                    else "个划分"
                    if suite == "public"
                    else "套数据"
                )
                sentences[index].append(
                    f"{label}（{condition}）：全局概率错误率 {100 * measure['mean']:.2f}%（{count}{unit}）。"
                )
            else:
                unit = (
                    "test people"
                    if group["kind"] == "har"
                    else "splits"
                    if suite == "public"
                    else "datasets"
                )
                sentences[index].append(
                    f"{label} ({condition}): global probabilistic error {100 * measure['mean']:.2f}% ({count} {unit})."
                )
    if sentences[0]:
        data["commentary"][suite] = _bi(
            "".join(sentences[0])
            + "这些数值只对应所列条件；各方法的差异和误差范围见表，不能用训练目标替代测试结果。",
            " ".join(sentences[1])
            + "These values apply to the listed conditions. Consult the table for method differences and variability; the training objective does not replace test performance.",
        )


def enrich(data, summary, records):
    """Add observed supplementary tables without pooling unrelated conditions."""
    data.setdefault("result_counts", {})
    _generality(data, summary)
    _shift(data, records)
    _ablations(data, summary, records)
    for suite in ("main", "public"):
        _central_commentary(data, summary, suite)
    scaling = _observed(summary, "scaling")
    if scaling:
        data["commentary"]["scaling"] = _bi(
            f"已观测 {len(scaling)} 个模型、规模、预算与方法组合，完整结果另列于 scaling_results.csv。每套数据的重复计时先取中位数，再在同一条件内计算均值和标准差；重复计时不增加独立数据数，尚未达到证明精度的运行仍保留。",
            f"Observed {len(scaling)} model, scale, budget and method combinations are listed in scaling_results.csv. Timing repetitions are first collapsed to a median within each dataset, then summarized by mean and SD within the same condition. Timing repeats add no independent datasets; runs missing the certificate target remain included.",
        )
    return data


def write_scaling_csv(summary, destination):
    """Export all scaling conditions; costs already use within-instance medians."""
    fields = [
        *CONDITION,
        "method",
        "observed_datasets",
        "planned_datasets",
        "observed_timing_runs",
        "planned_timing_runs",
        "status_counts",
    ]
    metrics = (
        "runtime",
        "first_best_time",
        "time_to_gap",
        "upper",
        "gap",
        "fit_count",
        "master_count",
        "gap_at_10s",
        "gap_at_60s",
        "gap_at_300s",
    )
    fields += [f"{metric}_{statistic}" for metric in metrics for statistic in ("mean", "sd", "n")]
    fields += ["certificate_reached_timing_runs", "certificate_unreached_timing_runs"]
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in summary.get("groups", []):
            group = item["group"]
            if group.get("suite") != "scaling" or group.get("role") != "comparison":
                continue
            row = {name: group.get(name) for name in (*CONDITION, "method")}
            row.update(
                observed_datasets=item["n_instances"],
                planned_datasets=item.get("planned_instances"),
                observed_timing_runs=item.get("n_timing_runs"),
                planned_timing_runs=item.get("planned_timing_runs"),
                status_counts=json.dumps(item.get("status_counts", {}), sort_keys=True),
            )
            for metric in metrics:
                for statistic in ("mean", "sd", "n"):
                    row[f"{metric}_{statistic}"] = (
                        item.get("metrics", {}).get(metric, {}).get(statistic)
                    )
            times = item.get("metrics", {}).get("time_to_gap", {})
            row["certificate_reached_timing_runs"] = times.get("valid_timing_runs")
            row["certificate_unreached_timing_runs"] = times.get("missing_timing_runs")
            writer.writerow(row)
    return destination
