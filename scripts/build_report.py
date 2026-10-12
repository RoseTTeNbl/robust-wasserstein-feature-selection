"""Fill the report template from audited records; incomplete builds require --draft.

No observed result is hardcoded in this file. Summary intervals resample training
seeds; paired differences are recomputed from per-trial values on common seeds.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from report_evidence import AblationEvidence, SearchEvidence

PRIMARY = ("risk", "greedy", "mmd", "wang", "l1", "random")
GATES = ("box", "triangle", "risk_no_root", "risk", "enumeration")
MECHANISMS = ("mean", "variance", "dependence", "mixture")
LABELS = {
    "risk": "Global risk", "risk_ratio": "Complete ratios", "greedy": "Robust greedy",
    "mmd": "MMD screening", "wang": "Additive MMD", "l1": "L1 logistic",
    "random": "Random", "signal": "Known signal", "all": "All features",
    "box": "Box", "triangle": "Triangle", "risk_no_root": "Risk, no root",
    "enumeration": "Enumeration",
}
TITLES = {
    "mean": "Mean difference", "variance": "Variance difference",
    "dependence": "Equal marginals, different dependence", "mixture": "Symmetric mixture",
    "wdbc": "Wisconsin diagnostic breast cancer", "ionosphere": "Ionosphere",
}
SUITES = ("statistical", "computational", "real", "sensitivity", "ratio")
CONFIG_FILES = {suite: suite for suite in SUITES} | {"ratio": "ratio_cuts"}
MARKERS = ("OVERVIEW", "SETUP_TABLE", "SEARCH_TABLE", "COMPUTATIONAL_TABLE",
           "STATISTICAL_TABLE", "REAL_TABLE", "SENSITIVITY_TABLE", "RECOVERY_TABLE", "REMEDIATION")


def escaped(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
                    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))



def number(row, field):
    try:
        value = float(row.get(field, ""))
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None



def fmt(value, digits=3, scientific=False):
    if value is None or not math.isfinite(value):
        return "---"
    if scientific and 0 < abs(value) < 0.001:
        mantissa, exponent = f"{value:.1e}".split("e")
        return rf"\({mantissa}\times10^{{{int(exponent)}}}\)"
    return f"{value:.{digits}f}"



def estimate(row, metric, *, real=False, digits=3):
    mean = number(row, metric + "_mean")
    if mean is None:
        return "---"
    if real:
        sd = number(row, metric + "_sd")
        return fmt(mean, digits) + (rf"\(\pm\){fmt(sd, digits)}" if sd is not None else "")
    lo, hi = number(row, metric + "_lo"), number(row, metric + "_hi")
    if lo is None or hi is None:
        return fmt(mean, digits)
    return (r"\shortstack{" + fmt(mean, digits) + r"\\{"
            + f"[{fmt(lo, digits)}, {fmt(hi, digits)}]" + "}}")



def key(row, *, group=False):
    fields = (row["suite"], row["dataset"], int(row["dimension"]), int(row["budget"]),
              round(float(row["theta"]), 12), row["method"])
    return fields if group else fields + (int(row["seed"]),)



def bootstrap(values):
    values = np.asarray(values, float)
    if not len(values):
        return None, None, None
    mean = float(values.mean())
    if len(values) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(20261003)
    means = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return mean, float(lo), float(hi)



def paired(rows, suite, dataset, metric, a="risk", b="mmd", dimension=None, theta=None):
    by_seed = defaultdict(dict)
    for row in rows:
        if row["suite"] != suite or row["dataset"] != dataset or row["method"] not in (a, b):
            continue
        if dimension is not None and int(row["dimension"]) != dimension:
            continue
        if theta is not None and abs(float(row["theta"]) - theta) > 1e-12:
            continue
        value = number(row, metric)
        if value is not None:
            by_seed[int(row["seed"])][row["method"]] = value
    differences = [v[a] - v[b] for v in by_seed.values() if a in v and b in v]
    return len(differences), bootstrap(differences)



def interval_cell(triple, digits=3):
    mean, lo, hi = triple
    if mean is None:
        return "---"
    return (r"\shortstack{" + fmt(mean, digits) + r"\\{"
            + f"[{fmt(lo, digits)}, {fmt(hi, digits)}]" + "}}")



def expected_trials(project):
    """Derive exact trial keys and hashes from the five frozen configurations."""
    expected = {}
    configs = {}
    for suite in SUITES:
        config = json.loads((project / "configs" / f"{CONFIG_FILES[suite]}.json").read_text(encoding="utf-8"))
        configs[suite] = config
        settings = config.get("settings")
        if settings is None:
            settings = []
            if suite == "real":
                manifest = json.loads((project / "docs/data_sources.json").read_text(encoding="utf-8"))
                available = {r["name"]: int(r["features"]) for r in manifest}
                dimensions = {Path(path).stem: available[Path(path).stem] for path in config["real_paths"]}
            else:
                dimensions = {name: config["dimension"] for name in config["mechanisms"]}
            for theta in config.get("radii", [config.get("theta", .25)]):
                for dataset, dimension in dimensions.items():
                    methods = list(config["methods"])
                    if config.get("diagnostic_references"):
                        methods += ["signal", "all"]
                    settings.append({"mechanism": dataset, "dimension": dimension,
                                         "budget": config["budget"], "seeds": config["seeds"],
                                         "gates": methods, "theta": theta})
        for setting in settings:
            theta = float(setting.get("theta", config.get("theta", .25)))
            current = {**config, "theta": theta} if suite == "sensitivity" else config
            digest = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()
            methods = list(setting["gates"])
            if setting.get("enumeration"):
                methods.append("enumeration")
            for method in methods:
                for seed in setting["seeds"]:
                    trial = (suite, setting["mechanism"], int(setting["dimension"]),
                             int(setting["budget"]), round(theta, 12), method, int(seed))
                    if trial in expected:
                        raise ValueError(f"Duplicate configured trial: {trial}")
                    expected[trial] = digest
    return expected, configs



def check_summary(rows, summaries):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row, group=True)].append(row)
    summary_groups = {key(row, group=True): row for row in summaries}
    if len(summary_groups) != len(summaries) or set(groups) != set(summary_groups):
        raise ValueError("Summary and per-trial group keys differ; run summarize.py again")
    for group_key, group in groups.items():
        summary = summary_groups[group_key]
        if int(summary["n_total"]) != len(group):
            raise ValueError(f"Stale summary count: {group_key}")
        if int(summary["n_complete"]) != sum(r["run_status"] == "complete" for r in group):
            raise ValueError(f"Stale summary completion count: {group_key}")
        if Counter(r["selection_status"] for r in group) != Counter(json.loads(summary["status_counts"])):
            raise ValueError(f"Stale summary status counts: {group_key}")
        for field, value in summary.items():
            if not field.endswith("_mean"):
                continue
            metric = field[:-5]
            vals = [number(r, metric) for r in group]
            vals = [v for v in vals if v is not None]
            if not vals or not math.isclose(float(value), float(np.mean(vals)), rel_tol=1e-9, abs_tol=1e-10):
                raise ValueError(f"Stale summary mean: {group_key}, {metric}")
            if int(summary.get(metric + "_n", -1)) != len(vals):
                raise ValueError(f"Stale effective metric count: {group_key}, {metric}")
            if metric + "_sd" in summary:
                sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                if not math.isclose(float(summary[metric + "_sd"]), sd, rel_tol=1e-9, abs_tol=1e-10):
                    raise ValueError(f"Stale summary SD: {group_key}, {metric}")
            if metric + "_lo" in summary or metric + "_hi" in summary:
                _, lo, hi = bootstrap(vals)
                if any(not math.isclose(float(summary.get(metric + suffix, float('nan'))), endpoint,
                                        rel_tol=1e-9, abs_tol=1e-10)
                       for suffix, endpoint in (("_lo", lo), ("_hi", hi))):
                    raise ValueError(f"Stale summary interval: {group_key}, {metric}")



def check_raw_record(row, record):
    """Reject a stale CSV even when its own aggregate summary is consistent."""
    identity = {**record, "suite": row["suite"], "theta": record.get("theta", .25)}
    if key(identity) != key(row) or record.get("config_sha256") != row["config_sha256"]:
        raise ValueError(f"CSV identity/configuration differs from raw record: {key(row)}")
    if record.get("run_status") != row["run_status"]:
        raise ValueError(f"CSV evaluation status differs from raw record: {key(row)}")
    selection = record.get("selection", {})
    if selection.get("status", "failure") != row["selection_status"]:
        raise ValueError(f"CSV selection status differs from raw record: {key(row)}")
    values = {field: record.get(field) for field in ("seed", "dimension", "budget", "theta")}
    values["theta"] = record.get("theta", .25)
    values.update({"selection_" + field: selection[field]
                   for field in ("lower", "upper", "gap", "seconds", "oracle_calls", "visited")
                   if field in selection})
    diagnostics = selection.get("diagnostics", {})
    values.update({field: diagnostics[field] for field in ("root_bound", "oracle_seconds", "master_seconds")
                   if field in diagnostics})
    values.update(record.get("metrics", {}))
    for stress in record.get("stress", []):
        values[f"stress_ber_{stress['rho']:.2f}"] = stress["balanced_error"]
        values[f"stress_randomized_ber_{stress['rho']:.2f}"] = stress["randomized_balanced_error"]
    for field in set(values) | set(row):
        saved, summarized = number(values, field), number(row, field)
        if saved is None and summarized is None:
            continue
        if (saved is None or summarized is None
                or not math.isclose(saved, summarized, rel_tol=1e-9, abs_tol=1e-10)):
            raise ValueError(f"CSV numeric value differs from raw record: {key(row)}, {field}")



class Evidence:
    def __init__(self, project, root, draft):
        self.project, self.root, self.draft = project, root, draft
        self.expected, self.configs = expected_trials(project)
        with (root / "per_trial.csv").open(encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        self.summaries = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        self.groups = {key(r, group=True): r for r in self.summaries}
        actual = [key(r) for r in self.rows]
        if len(set(actual)) != len(actual):
            raise ValueError("Duplicate per-trial key")
        extra = set(actual) - set(self.expected)
        if extra:
            raise ValueError(f"Unplanned trials in report input: {list(extra)[:3]}")
        self.missing = set(self.expected) - set(actual)
        if self.missing and not draft:
            raise ValueError(f"Final build rejected: {len(self.missing)} of {len(self.expected)} trials missing. "
                             "Wait for all suites, summarize again, or explicitly use --draft.")
        check_summary(self.rows, self.summaries)
        self.raw = {}
        for row in self.rows:
            trial = key(row)
            if row["config_sha256"] != self.expected[trial]:
                raise ValueError(f"Trial configuration hash differs from frozen design: {trial}")
            path = root / row["record_file"].replace("\\", "/")
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Record path leaves the results directory")
            record = json.loads(path.read_text(encoding="utf-8"))
            check_raw_record(row, record)
            self.raw[trial] = record
        saved = list(root.rglob("records/*.json"))
        if len(saved) != len(self.rows):
            raise ValueError("Saved records and CSV have different lengths; summarize.py must be rerun")
        if not draft and not (root / "remediation/initial_failures.json").is_file():
            raise ValueError("Final build requires published remediation/initial_failures.json, including an explicit empty array if there were no initial failures")
        originals = archived_records(self)
        for trial, original in originals.items():
            if trial not in self.expected or trial not in self.raw:
                if draft:
                    continue
                raise ValueError(f"Original failure lacks a planned final record: {trial}")
            final = self.raw[trial]
            for before, after, label in ((original.get("selection", {}), final.get("selection", {}), "selection"),
                                         (original, final, "total")):
                field = "seconds" if label == "selection" else "total_seconds"
                old, new = number(before, field), number(after, field)
                if old is not None and (new is None or new + 1e-9 < old):
                    raise ValueError(f"Original {label} time was erased from final record: {trial}")
            for field in ("oracle_seconds", "master_seconds"):
                old = number(original.get("selection", {}).get("diagnostics", {}), field)
                new = number(final.get("selection", {}).get("diagnostics", {}), field)
                if old is not None and (new is None or new + 1e-9 < old):
                    raise ValueError(f"Original component time was erased: {trial}, {field}")
            metadata = final.get("numerical_remediation", {})
            if metadata:
                for initial_field, restart_field, final_value in (
                    ("initial_selection_seconds", "restart_selection_seconds", number(final.get("selection", {}), "seconds")),
                    ("initial_total_seconds", "restart_total_seconds", number(final, "total_seconds")),
                ):
                    first, second = number(metadata, initial_field), number(metadata, restart_field)
                    if first is not None and second is not None and (
                            final_value is None or not math.isclose(first + second, final_value, rel_tol=1e-9, abs_tol=1e-8)):
                        raise ValueError(f"Cumulative remediation time is inconsistent: {trial}, {initial_field}")

    def lookup(self, suite, dataset, method, dimension=None, theta=.25):
        found = [r for r in self.summaries if r["suite"] == suite and r["dataset"] == dataset
                 and r["method"] == method and abs(float(r["theta"]) - theta) < 1e-12
                 and (dimension is None or int(r["dimension"]) == dimension)]
        if len(found) > 1:
            raise ValueError("Ambiguous summary lookup; specify dimension")
        return found[0] if found else None

    def select(self, suite, dataset=None, method=None, dimension=None):
        return [r for r in self.summaries if r["suite"] == suite
                and (dataset is None or r["dataset"] == dataset)
                and (method is None or r["method"] == method)
                and (dimension is None or int(r["dimension"]) == dimension)]



def computational_check(evidence):
    rows = evidence.rows
    references = {key(r)[:-2] + (int(r["seed"]),): r for r in rows
                  if r["suite"] == "computational" and r["method"] == "enumeration"
                  and r["selection_status"] == "optimal"}
    differences, violations = [], 0
    for row in rows:
        if row["suite"] != "computational" or row["method"] == "enumeration":
            continue
        ref = references.get(key(row)[:-2] + (int(row["seed"]),))
        if ref is None:
            continue
        lo, hi = number(row, "selection_lower"), number(row, "selection_upper")
        reflo, refhi = number(ref, "selection_lower"), number(ref, "selection_upper")
        tol = 2 * (1e-6 + 1e-5 * abs(refhi))
        if lo is not None and hi is not None and (lo > refhi + tol or hi < reflo - tol):
            violations += 1
        if row["selection_status"] == "optimal" and hi is not None:
            differences.append(abs(hi - refhi))
    return len(differences), max(differences, default=None), violations



def solved_count(row):
    values = row.get("status_counts", {})
    if isinstance(values, str):
        values = json.loads(values)
    return int(values.get("optimal", 0))



def archived_records(e):
    published = e.root / "remediation/initial_failures.json"
    if published.exists():
        records = json.loads(published.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("Published original failure provenance must be an array")
        result = {key(r): r for r in records}
        if len(result) != len(records):
            raise ValueError("Duplicate original failure provenance")
        return result
    archive = e.project / "_archive" / "numerical_failures_20261003"
    records = {}
    if archive.exists():
        for path in archive.rglob("records/*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            if "selection" not in record or "dataset" not in record:
                continue
            suite = path.relative_to(archive).parts[0]
            row = {**record, "suite": suite}
            records[key(row)] = record
    return records



LABELS_ZH = {
    "risk": "全局风险", "risk_ratio": "完整分式", "greedy": "鲁棒贪心",
    "mmd": "MMD 筛选", "wang": "Wang 加性 MMD", "l1": "L1 逻辑回归",
    "random": "随机", "signal": "已知信号", "all": "全部变量",
    "box": "逐项价格界", "triangle": "总价格界", "risk_no_root": "风险（无根节点）",
    "enumeration": "组合枚举",
}


TITLES_ZH = {"mean": "均值变化", "variance": "方差变化", "dependence": "相关结构变化",
             "mixture": "对称混合", "wdbc": "WDBC", "ionosphere": "Ionosphere"}


FAILURES = {"failure", "master_failure", "oracle_failure", "numerical_failure",
            "numerical_inconsistency"}


def tr(language, english, chinese):
    return chinese if language == "zh" else english



def method_label(method, language):
    return (LABELS_ZH if language == "zh" else LABELS).get(method, escaped(method))



def title_label(dataset, language):
    short = {"mean": "Mean", "variance": "Variance", "dependence": "Dependence", "mixture": "Mixture"}
    return (TITLES_ZH if language == "zh" else short).get(dataset, escaped(dataset))



def report_table(caption, headers, rows, *, label, columns=None):
    """Small paper tables, with no shrinking or hidden overflow correction."""
    if not rows:
        return ""
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Table row length differs from its header")
    columns = columns or "@{}p{28mm}" + "r" * (len(headers) - 1) + "@{}"
    return (r"\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3.5pt}" + "\n"
            + r"\caption{" + caption + r"}\label{" + label + "}\n"
            + r"\begin{tabularx}{\linewidth}{" + columns + "}\n"
            + r"\toprule " + " & ".join(headers) + r" \\" + "\n"
            + r"\midrule" + "\n"
            + "\n".join(" & ".join(map(str, row)) + r" \\" for row in rows) + "\n"
            + r"\bottomrule\end{tabularx}\end{table}" + "\n"
            + r"\FloatBarrier" + "\n")



def concise_overview(e, language, validation=None):
    prefix = (rf"\textbf{{DRAFT SNAPSHOT: {len(e.rows)}/{len(e.expected)} reference records; "
              rf"{len(e.missing)} missing.}} " if e.draft else "")
    text = tr(language,
        "We study whether choosing a few variables by minimizing a robust testing loss preserves the difference between two populations. Four synthetic mechanisms distinguish testing power from signal recovery; independent prediction and two real datasets provide further checks. The recorded robust-risk optimum does not consistently give the best testing power or prediction. ",
        "本报告考察：通过最小化鲁棒检验损失选择少量变量，能否保留两个总体的差异。四种合成机制分别评价检验功效与信号恢复，独立预测和两个真实数据集提供进一步检验。记录中的鲁棒风险最优并没有一致带来最好的功效或预测表现。")
    if validation and getattr(validation, "ablation", None):
        a, b = (validation.ablation["totals"][m] for m in ("cuts_fraction", "cuts_tangent"))
        reduction = 100 * (1 - a["completed_evaluations"]["total"] / b["completed_evaluations"]["total"])
        cost = 100 * (a["seconds"]["total"] / b["seconds"]["total"] - 1)
        text += tr(language,
            f"On reused small-case settings, retaining fraction bounds changes full subset evaluations by {-reduction:+.1f}\\% and total median-summed time by {cost:+.1f}\\%. This is an ablation, not independent confirmation.",
            f"在再次使用的小规模配置上，保留分式下界使完整组合评价次数变化 {-reduction:+.1f}\\%，逐实例中位时间总和变化 {cost:+.1f}\\%。这是消融对照，并非独立确认。")
    return prefix + text





def setup_report(e, language):
    stat = e.configs.get("statistical", {})
    real = e.configs.get("real", {})
    sources_path = e.project / "docs/data_sources.json"
    sources = json.loads(sources_path.read_text(encoding="utf-8")) if sources_path.exists() else []
    rows = []
    if stat:
        rows.append([tr(language, "Synthetic", "合成数据"),
                     f"{stat['dimension']}/{stat['budget']}",
                     f"{stat['n_select']}/{stat['n_head']}",
                     f"{stat['n_classification']}" + tr(language, "/class", "/类"),
                     len(stat["seeds"])])
    for data in sources:
        if not real:
            continue
        held_out = data["rows"] - 2 * (real["n_select"] + real["n_head"])
        rows.append(["WDBC" if data["name"] == "wdbc" else "Ionosphere",
                     f"{data['features']}/{real['budget']}",
                     f"{real['n_select']}/{real['n_head']}",
                     str(held_out) + tr(language, " total", " 总数"), len(real["seeds"])])
    text = report_table(
        tr(language, "Reference statistical data. Selection and classifier fitting counts are per class; final counts show their unit. Repeats are training seeds or overlapping real-data splits.",
           "参考统计数据。选择与分类器拟合样本数均为每类数量；最终样本列注明单位。重复次数为训练种子数或重叠的真实数据分割数。"),
        [tr(language, "Data", "数据"), "$D/d$",
         tr(language, r"\shortstack{Selection / fitting\\per class}", r"\shortstack{选择 / 拟合\\每类样本数}"),
         tr(language, r"\shortstack{Final\\observations}", r"\shortstack{最终\\观测}"),
         tr(language, "Repeats", "重复次数")], rows, label="tab:data", columns="@{}p{25mm}rYrr@{}")
    settings = e.configs.get("computational", {}).get("settings", [])
    seen, rows = set(), []
    for setting in settings:
        setting_key = (setting["dimension"], setting["budget"], setting["n_select"], len(setting["seeds"]))
        if setting_key in seen:
            continue
        seen.add(setting_key)
        rows.append([str(setting_key[0]), str(setting_key[1]), str(setting_key[2]),
                     str(setting_key[3]), tr(language, "Mean, mixture", "均值、混合")])
    text += report_table(tr(language, "Reference optimization settings. Each row is evaluated separately for mean and mixture alternatives.",
                           "参考优化参数。每行分别在均值与混合备择上评价。"),
                         ["$D$", "$d$", tr(language, "Atoms/class", "源点/类"),
                          tr(language, "Seeds", "种子数"), tr(language, "Alternatives", "备择")],
                         rows, label="tab:computation-settings", columns="@{}rrrrY@{}")
    if stat:
        text += tr(language,
                   f"The primary class radii are both {stat['theta']:.2f}. Each synthetic seed has {stat['test_replicates']} alternative and {stat['test_replicates']} null batches, with {stat['n_testing']} observations per class in a batch. ",
                   f"主实验两类半径均为 {stat['theta']:.2f}。每个合成种子分别使用 {stat['test_replicates']} 个备择和 {stat['test_replicates']} 个零假设批次，每批每类 {stat['n_testing']} 个观测。")
        text += tr(language, f"Radius sensitivity uses {len(e.configs.get('sensitivity', {}).get('seeds', []))} seeds per mechanism and radius.\n",
                   f"半径分析在每个机制和半径上使用 {len(e.configs.get('sensitivity', {}).get('seeds', []))} 个种子。\n")
    return text



def reference_computation(e, language):
    n, difference, violations = computational_check(e)
    text = tr(language,
              f"Across {n} solved gate/enumeration pairs, the largest recorded feasible-objective difference is {fmt(difference, scientific=True)}, with {violations} incompatible intervals at the combined solver tolerances. ",
              f"在 {n} 对已解出的约束形式/枚举结果中，最大记录可行目标差为 {fmt(difference, scientific=True)}；按合并求解容差核查，{violations} 个区间不相容。") if n else ""
    rows = []
    for dataset in ("mean", "mixture"):
        for method in GATES[:-1]:
            r = e.lookup("computational", dataset, method, dimension=12)
            if r:
                rows.append([title_label(dataset, language), method_label(method, language),
                             f"{solved_count(r)}/{r['n_total']}", estimate(r, "selection_seconds", digits=2),
                             fmt(number(r, "selection_visited_mean"), 1)])
        primary = [r for r in e.rows if r["suite"] == "computational" and r["dataset"] == dataset
                   and r["method"] == "risk" and int(r["dimension"]) >= 20]
        if primary:
            limits = sum(r["selection_status"] == "time_limit" for r in primary)
            text += tr(language,
                       f"At $D\\ge20$, global risk reaches the limit in {limits}/{len(primary)} {escaped(dataset)} instances. ",
                       f"在 $D\\ge20$ 时，全局风险选择在{title_label(dataset, language)}的 {len(primary)} 个实例中有 {limits} 个达到时间上限。")
    text += "\n" + report_table(
        tr(language, "$D=12$ reference comparison. Time is charged seconds with 95\\% seed-bootstrap intervals; supports are distinct integer subsets, excluding fractional root calls.",
           "$D=12$ 的参考比较。时间为累计秒数及 95\\% 种子自助区间；组合数仅计不同整数子集，不含分数根节点调用。"),
        [tr(language, "Alternative", "备择"), tr(language, "Search", "搜索"),
         tr(language, "Solved", "解出数"), tr(language, "Time (s)", "时间（秒）"),
         tr(language, "Supports", "组合数")], rows, label="tab:reference-computation", columns="@{}p{22mm}Yrrr@{}")
    text += report_figure("optimization_comparison", tr(language,
        "Earlier implementation: charged time and solved fraction as dimension grows. Lower time and higher solved fraction are preferable; unsuccessful runs remain included.",
        "较早实现：随维度增加的实际时间与解出比例。时间越低、解出比例越高越好；未成功运行仍保留。"), "fig:computation", language)
    return text



def reference_ratio(e, language):
    rows = []
    for dataset in ("mean", "mixture"):
        for dimension in (8, 12, 20):
            a = e.lookup("ratio", dataset, "risk", dimension=dimension)
            b = e.lookup("ratio", dataset, "risk_ratio", dimension=dimension)
            if not a or not b:
                continue
            _n, difference = paired(e.rows, "ratio", dataset, "selection_seconds", a="risk_ratio", b="risk", dimension=dimension)
            rows.append([title_label(dataset, language), dimension,
                         f"{solved_count(a)}/{a['n_total']}", f"{solved_count(b)}/{b['n_total']}",
                         interval_cell(difference, digits=2)])
    if not rows:
        return ""
    text = tr(language,
              "A separately declared complete-fraction comparison uses different seeds and is not pooled with the primary timing cohort. The full fractions are stronger value bounds, but their product-variable integer master can cost more.\n",
              "另行声明的完整分式比较使用不同种子，不与主时间批次合并。完整分式提供更强的值下界，但含乘积变量的整数主问题可能需要更多时间。\n")
    text += report_table(
        tr(language, "Earlier full-fraction supplement. The final column is paired full-fraction minus tangent time (seconds, 95\\% seed-bootstrap interval); a positive value is slower.",
           "较早的完整分式补充比较。末列为配对的完整分式减切线时间（秒，95\\% 种子自助区间），正值表示更慢。"),
        [tr(language, "Alternative", "备择"), "$D$", tr(language, "Tangent solved", "切线解出"),
         tr(language, "Fraction solved", "分式解出"), tr(language, r"\shortstack{Paired time\\difference}", r"\shortstack{配对时间\\差}")],
        rows, label="tab:reference-ratio", columns="@{}p{24mm}rrrr@{}")
    return text



def statistical_report(e, language):
    text = tr(language,
        "Figure~\\ref{fig:power} measures detection after selection; higher power is better. The two references distinguish selection errors from testing difficulty and noise dilution.\n\n",
        "图~\\ref{fig:power} 评价选择后的检出能力，功效越高越好。两个参照帮助区分选择错误、检验困难与噪声干扰。\n\n")
    risk, screen = e.lookup("statistical", "mean", "risk"), e.lookup("statistical", "mean", "mmd")
    if risk and screen:
        n, difference = paired(e.rows, "statistical", "mean", "power")
        mean, lo, hi = difference
        if mean is not None:
            text += tr(language,
                f"For mean differences, global power is {fmt(number(risk, 'power_mean'))}, versus {fmt(number(screen, 'power_mean'))} for individual-coordinate MMD screening; the paired difference is {mean * 100:.1f} percentage points [{lo * 100:.1f}, {hi * 100:.1f}] ($n={n}$). ",
                f"均值变化时，全局选择的功效为 {fmt(number(risk, 'power_mean'))}，单坐标 MMD 筛选为 {fmt(number(screen, 'power_mean'))}；配对差为 {mean * 100:.1f} 个百分点 [{lo * 100:.1f}, {hi * 100:.1f}]（$n={n}$）。")
        greedy = e.lookup("statistical", "mean", "greedy")
        if greedy and "n_total" in risk:
            text += tr(language,
                f"All {solved_count(risk)}/{risk['n_total']} mean-shift global searches meet the optimization tolerance, yet greedy power is {fmt(number(greedy, 'power_mean'))}. Solving the empirical robust objective accurately does not maximize this independent test's power.\n\n",
                f"均值变化的全局搜索有 {solved_count(risk)}/{risk['n_total']} 次达到优化容差，但贪心功效为 {fmt(number(greedy, 'power_mean'))}。精确求解经验鲁棒目标，并不等于让这个独立检验的功效最大。\n\n")
    text += report_figure("power_and_calibration", tr(language,
        "Common-test power (higher is better). The six selectors use three variables; Signal uses the true three and All uses twelve. Points and bars are means and 95\\% seed-bootstrap intervals.",
        "共同检验的功效，越大越好。六种选择方法使用三个变量；已知信号参照使用真实的三个变量，全部参照使用十二个。点与误差线为均值和 $95\\%$ 种子自助区间。"), "fig:power", language)
    variance, mixture = (e.lookup("statistical", d, "signal") for d in ("variance", "mixture"))
    if variance and mixture:
        va, ma = (e.lookup("statistical", d, "all") for d in ("variance", "mixture"))
        dependence = e.lookup("statistical", "dependence", "signal")
        text += tr(language,
            f"The known-signal powers are {fmt(number(variance, 'power_mean'))} for variance and {fmt(number(mixture, 'power_mean'))} for mixture differences, whereas using every variable gives {fmt(number(va, 'power_mean'))} and {fmt(number(ma, 'power_mean'))}. Keeping all the signal is not enough when irrelevant coordinates weaken the common test. In the dependence case, even the known-signal power is {fmt(number(dependence, 'power_mean'))}: the failure of practical selectors is accompanied by limited sensitivity of the 40-observation-per-class test.\n\n",
            f"方差与混合变化中，已知信号的功效分别为 {fmt(number(variance, 'power_mean'))} 和 {fmt(number(mixture, 'power_mean'))}，使用全部变量时则为 {fmt(number(va, 'power_mean'))} 和 {fmt(number(ma, 'power_mean'))}。保留全部信号并不充分：无关坐标仍可能削弱共同检验。相关结构变化中，即便已知信号的功效也只有 {fmt(number(dependence, 'power_mean'))}；实际方法选择困难的同时，每类四十个观测的检验本身也不够敏感。\n\n")
    text += tr(language,
        "Table~\\ref{tab:null} tests two samples from the same population. Their rejection rate measures false alarms, which should not systematically exceed 0.05; lower conservative rates are acceptable.\n",
        "表~\\ref{tab:null} 检验来自同一总体的两组样本。拒绝比例就是假警报率，不应系统性超过 $0.05$；较低的保守错误率可以接受。\n")
    rows = []
    for method in (*PRIMARY, "signal", "all"):
        records = [e.lookup("statistical", d, method) for d in MECHANISMS]
        if any(records):
            rows.append([method_label(method, language), *[fmt(number(r, "null_rejection_mean")) if r else "---" for r in records]])
    text += report_table(tr(language,
        "False-alarm rate of the common RBF test (declared level 0.05). Entries average testing batches within each training seed, then the 30 seeds. Known signal and all features are diagnostic references.",
        "共同 RBF 检验的假警报率，预定水平为 $0.05$。各训练种子内先对批次取平均，再对三十个种子取平均。已知信号与全部变量为诊断参照。"),
        [tr(language, "Selector", "选择方法"), *[title_label(d, language) for d in MECHANISMS]],
        rows, label="tab:null", columns="@{}Yrrrr@{}")
    text += r"\subsection{" + tr(language, "What changes when the learned weights are retained?", "保留学习到的权重后，检验表现有何变化？") + "}\n"
    text += tr(language,
        "The common test uses joint similarity on the selected coordinates. Wang's additive method instead combines coordinate-wise statistics, choosing weights through\n",
        "共同检验使用所选坐标的联合相似度。Wang 的加性方法则组合各坐标统计量，并通过下面的目标选择权重：\n")
    text += (r"\[\max_w\ t^\top w-\lambda w^\top\widehat\Sigma w,\qquad"
             r"\|w\|_2=1,\quad\|w\|_0\le d.\]" + "\n")
    text += tr(language,
        "Here $t_j$ is the training MMD statistic for coordinate $j$, $\\widehat\\Sigma$ estimates their joint variability, $w_j$ is its weight, and $d=3$ limits nonzero weights. The first term rewards a measured difference; the second penalizes variability, with fixed $\\lambda=0.1$. Weights may be negative, so their similarity combination need not be a positive-semidefinite kernel. For the same selected coordinates, common power drops the weights, while native power retains the signed additive weights. The native permutation test uses the raw weighted MMD, without the training variance penalty. This is a partial implementation, not a complete reproduction.\n",
        "其中 $t_j$ 为坐标 $j$ 的训练 MMD 统计量，$\\widehat\\Sigma$ 估计这些统计量的联合波动，$w_j$ 为权重，$d=3$ 限制非零权重数。第一项鼓励可测得的差异，第二项惩罚波动，固定 $\\lambda=0.1$。权重可为负，因此组合后的相似度未必是半正定核。对于同一组所选坐标，共同功效丢弃权重，原生功效保留有符号加性权重。原生置换检验使用原始带权 MMD，不包含训练时的方差惩罚。这里只实现了部分方法，并非完整复现。\n")
    rows = []
    for dataset in MECHANISMS:
        record = e.lookup("statistical", dataset, "wang")
        if record and number(record, "native_additive_power_mean") is not None:
            rows.append([title_label(dataset, language), int(record["native_additive_power_n"]),
                         estimate(record, "power"), estimate(record, "native_additive_power"),
                         estimate(record, "native_additive_null_rejection")])
    text += report_table(tr(language,
        "Two tests on the same Wang-selected support. Common RBF power uses only its coordinates; native power retains the signed additive weights. Entries are seed means and 95\\% intervals; native false alarms are also shown.",
        "Wang 所选同一组合的两种检验。共同 RBF 功效只使用坐标；原生功效保留有符号加性权重。数值为种子均值和 $95\\%$ 区间，并列出原生假警报率。"),
        [tr(language, "Difference", "差异"), "$n$", tr(language, "Common RBF power", "共同 RBF 功效"),
         tr(language, "Native power", "原生功效"), tr(language, "Native false alarms", "原生假警报率")],
        rows, label="tab:native", columns="@{}p{24mm}rrrr@{}")
    if rows:
        text += tr(language,
            "The native test is substantially stronger in the variance and mixture cases. Treating common RBF power as the performance of the complete additive method would understate that method. The common comparison instead isolates the usefulness of the selected coordinates.\n",
            "方差与混合变化中，原生检验明显更强。若把共同 RBF 功效当成整个加性方法的表现，会低估它；共同评价的目的，是单独比较所选坐标的作用。\n")
    return text



def sensitivity_report(e, language):
    rows = []
    for dataset in ("mean", "mixture"):
        for radius in e.configs.get("sensitivity", {}).get("radii", []):
            r = e.lookup("sensitivity", dataset, "risk", theta=radius)
            if r:
                rows.append([title_label(dataset, language), f"{radius:.2f}", int(r.get("power_n", 0)),
                             estimate(r, "power"), estimate(r, "robust_randomized_balanced_error"),
                             estimate(r, "robust_worst_stress_randomized_balanced_error")])
    text = report_table(tr(language, "Joint radius sensitivity. Entries are seed means and 95\\% intervals; errors are randomized balanced errors. Stress is the maximum over preset shifts within each seed.",
                            "完整过程的半径敏感性。数值为种子均值与 95\\% 区间；错误率为随机化平衡错误率，压力错误在每个种子内对预设平移取最大值。"),
                        [tr(language, "Alternative", "备择"), r"$\theta$", "$n$", tr(language, "Power", "功效"),
                         tr(language, "Fresh error", "新样本错误"), tr(language, "Shift maximum", "平移最大值")],
                        rows, label="tab:radius", columns="@{}p{22mm}rrrrr@{}")
    if rows:
        text += tr(language,
            "Larger radii improve the recorded power and randomized error in these two mechanisms. This is a result of the preset comparison, not a general monotonicity guarantee or an independently validated radius-selection rule.\n",
            "这两种机制的记录中，较大半径带来更高功效和较低随机化错误率。这是预设比较的结果，并非一般的单调性保证，也没有独立验证据此选择半径的规则。\n")
    return text



def real_report(e, language):
    rows = []
    for dataset in ("wdbc", "ionosphere"):
        for method in PRIMARY:
            r = e.lookup("real", dataset, method)
            if r:
                rows.append(["WDBC" if dataset == "wdbc" else "Ionosphere", method_label(method, language),
                             estimate(r, "robust_randomized_balanced_error", real=True),
                             estimate(r, "svm_balanced_error", real=True)])
    text = report_table(tr(language, "Independent common classifiers on real datasets. Entries are mean $\\pm$ overlapping-split SD. Robust error is randomized BER; SVM error is hard BER. These are incumbent supports when search is limited.",
                            "真实数据上的独立共同分类器。数值为均值 $\\pm$ 重叠分割标准差。鲁棒错误为随机化 BER，SVM 错误为确定性 BER；搜索受限时使用保留的可行组合。"),
                        [tr(language, "Data", "数据"), tr(language, "Selector", "选择器"),
                         tr(language, r"\shortstack{Robust\\randomized BER}", r"\shortstack{鲁棒\\随机化 BER}"),
                         "SVM BER"],
                        rows, label="tab:real", columns="@{}p{18mm}Yrr@{}")
    risk = [r for r in e.rows if r["suite"] == "real" and r["method"] == "risk"]
    if risk:
        limits = sum(r["selection_status"] == "time_limit" for r in risk)
        text += tr(language, f"Global search is time-limited in {limits}/{len(risk)} splits; its supports must not be called certified optima.\n",
                   f"全局搜索在 {limits}/{len(risk)} 次分割中达到时间上限，因此不能将这些组合称为已认证最优解。\n")
    return text



def concise_records(e, language):
    rows = []
    for suite in SUITES:
        source = [r for r in e.rows if r["suite"] == suite]
        counts = Counter(r["selection_status"] for r in source)
        failures = sum(counts[k] for k in FAILURES)
        other = len(source)-counts["optimal"]-counts["time_limit"]-failures
        names = {"statistical": ("Synthetic", "合成统计"), "computational": ("Optimization", "优化"),
                 "real": ("Real data", "真实数据"), "sensitivity": ("Radius", "半径"), "ratio": ("Fraction supplement", "分式补充")}
        rows.append([tr(language, *names[suite]), len(source), counts["optimal"], counts["time_limit"], failures, other])
    text = report_table(tr(language, "All frozen outcomes retained. Optimum refers to the robust criterion at its declared tolerance. Other includes baseline and diagnostic completion; it does not imply a robust global optimum.",
                            "保留的全部冻结运行结果。最优指在声明容差下解出鲁棒目标。其他包含基线与诊断完成，不代表鲁棒全局最优。"),
                        [tr(language, "Suite", "组别"), tr(language, "Records", "记录"), tr(language, "Optimum", "最优"),
                         tr(language, "Time limit", "时间上限"), tr(language, "Failure", "失败"), tr(language, "Other", "其他")],
                        rows, label="tab:outcomes", columns="@{}Yrrrrr@{}")
    originals = archived_records(e)
    outcomes = Counter(e.raw[k].get("selection", {}).get("status", "failure") for k in originals if k in e.raw)
    final_failures = sum(r.get("selection", {}).get("status", "failure") in FAILURES for r in e.raw.values())
    if originals:
        text += tr(language,
                   f"The {len(originals)} original failures were rerun within the unused portion of their original selection allowance: {outcomes['optimal']} ended optimal, {outcomes['time_limit']} time-limited and {sum(outcomes[k] for k in FAILURES)} failed. The final collection has {final_failures} failures, including later sensitivity outcomes. Original and replay costs remain charged.\n",
                   f"最初 {len(originals)} 次失败在原选择预算的剩余部分内重算：最终 {outcomes['optimal']} 次最优、{outcomes['time_limit']} 次达到时间上限，{sum(outcomes[k] for k in FAILURES)} 次失败。最终记录共有 {final_failures} 次失败，包含后来半径实验的结果。原尝试与重算成本全部保留。\n")
    environment = json.loads((e.root/"environment.json").read_text(encoding="utf-8")) if (e.root/"environment.json").exists() else {}
    hardware = json.loads((e.root/"hardware_cpu.json").read_text(encoding="utf-8")) if (e.root/"hardware_cpu.json").exists() else {}
    packages = environment.get("packages", {})
    engine = environment.get("linear_integer_backend", {}).get("highs_version", "unrecorded")
    total = sum(float(r.get("total_seconds", 0)) for r in e.raw.values())
    selection = sum(float(r.get("selection", {}).get("seconds", 0)) for r in e.raw.values())
    host = escaped(str(hardware.get("Name", "unrecorded host")).strip())
    unknown = sum(number(r.get("selection", {}), "seconds") is None for r in e.raw.values())
    if unknown:
        text += tr(language, f"Selection-only timing is unavailable for {unknown} records; they remain in outcome counts and are not imputed as zero in method means.\n",
                   f"{unknown} 条记录没有单独选择时间；它们保留在结果计数中，不在方法均值里填成零。\n")
    text += tr(language,
               f"Reference computations used {host}, one solver thread and SciPy-backed HiGHS {escaped(engine)} (SciPy {escaped(packages.get('scipy','unrecorded'))}). Logistic verification uses CVXPY {escaped(packages.get('cvxpy','unrecorded'))} and Clarabel {escaped(packages.get('clarabel','unrecorded'))}. Charged selection and end-to-end sums are {selection/3600:.2f} and {total/3600:.2f} hours, including unsuccessful costs; concurrent-run sums are not elapsed calendar time.\n",
               f"参考计算使用 {host}、一个求解线程，以及 SciPy 内置的 HiGHS {escaped(engine)}（SciPy {escaped(packages.get('scipy','unrecorded'))}）。逻辑损失核验使用 CVXPY {escaped(packages.get('cvxpy','unrecorded'))} 和 Clarabel {escaped(packages.get('clarabel','unrecorded'))}。累计选择时间与全部流程时间分别为 {selection/3600:.2f} 和 {total/3600:.2f} 小时，包含未成功尝试；并发运行时间的总和不是实际日历耗时。\n")
    return text



SEARCH_NAMES = {
    "lcb": ("Mean minus SD", "均值减标准差"),
    "ei": ("Expected improvement", "期望改进"),
    "guard": ("Mean minus SD, master priority", "均值减标准差，主问题优先"),
    "ei_guard": ("Improvement, master priority", "期望改进，主问题优先"),
    "baseline": ("Original GP search", "原 GP 搜索"),
    "screened": ("Ranked proposals", "增加排序候选"),
    "core": ("Dual-face tangent", "对偶面切线"),
    "curvature": ("Stored fractions", "保留分式"),
    "ratio": ("Fractions at every step", "每步分式主问题"),
    "master": ("Tangent master at every step", "每步切线主问题"),
}


def search_table(stage, language, label, caption):
    rows = []
    for method, total in stage["totals"].items():
        runs = stage.get("run_totals", stage["totals"])[method]
        def numeric(field, digits=0, total=total):
            return fmt(total[field]["total"], digits) if total.get(field) else "---"
        rows.append([tr(language, *SEARCH_NAMES[method]), numeric("completed_evaluations"),
                     numeric("continuous_calls"), numeric("masters"),
                     numeric("master_seconds", 3), numeric("seconds", 3),
                     f"{runs['statuses'].get('optimal', 0)}/{runs['cases']}"])
    headers = [tr(language, "Search", "搜索"), "$T$",
               tr(language, r"\shortstack{Continuous\\solves}", r"\shortstack{连续\\求解}"),
               tr(language, "Masters", "主问题"),
               tr(language, r"\shortstack{Master\\time (s)}", r"\shortstack{主问题\\时间（秒）}"),
               tr(language, r"\shortstack{Total\\time (s)}", r"\shortstack{总时间\\（秒）}"),
               tr(language, r"\shortstack{Optimal\\runs}", r"\shortstack{最优\\运行}")]
    return report_table(caption, headers, rows, label=label, columns="@{}Yrrrrrr@{}")


def development_table(validation, language):
    rows = []
    for i, stage in enumerate(validation.development, 1):
        for method, total in stage["totals"].items():
            count = total.get("continuous_calls")
            rows.append([str(i), tr(language, *SEARCH_NAMES[method]),
                         fmt(total["completed_evaluations"]["total"], 0),
                         fmt(count["total"], 0) if count else "---",
                         fmt(total["masters"]["total"], 0), fmt(total["seconds"]["total"], 3),
                         f"{total['statuses'].get('optimal', 0)}/{total['cases']}"])
    return report_table(
        tr(language, "All development comparisons, each on the same 26 paired cases. Stages use separate source snapshots and timing batches. A dash means the measurement was not recorded. An unfinished run remains in its denominator and its partial work is retained.",
           "全部开发对照，每个阶段均为相同的 26 个配对实例。各阶段使用各自代码快照与计时批次。横线表示未记录该项测量。未完成运行保留在分母中，其已发生工作也保留。"),
        [tr(language, "Stage", "阶段"), tr(language, "Search", "搜索"), "$T$",
         tr(language, r"\shortstack{Continuous\\solves}", r"\shortstack{连续\\求解}"),
         tr(language, "Masters", "主问题"), tr(language, "Time (s)", "时间（秒）"),
         tr(language, "Optimal", "最优")], rows, label="tab:search-development", columns="@{}rYrrrrr@{}")


def search_report(validation, language):
    """Historical October 10 comparison; never used for the current report body."""
    if validation is None or validation.confirmation is None:
        return tr(language, r"\textbf{DRAFT: confirmation is unavailable.}", r"\textbf{草稿：尚无确认结果。}")
    stage = validation.confirmation
    cases, repeats = len(stage["raw"]), stage["protocol"]["timing_repeats"]
    datasets = len({r["case"]["data_sha256"] for r in stage["raw"].values()})
    text = tr(language,
        "The revision keeps the existing rule for predicting which subset to evaluate next. It additionally retains the stronger dual-fraction lower bound already returned by each full solve, using that information to discard more unpromising subsets. No additional continuous solve is introduced, and every newly chosen binary subset is fully evaluated. We compare it with the original GP-guided tangent search and with refreshing a tangent master at every step.\n\n",
        "改进保持原本预测下一个评价组合的规则，只额外保留每次完整求解已经返回的较强对偶分式下界，用它排除更多不值得评价的组合。不增加额外连续求解，任何新选中的二元组合仍然完整评价。对照为原来的高斯过程引导切线搜索，以及每一步都刷新切线主问题的实现。\n\n")
    text += tr(language,
        f"The separate confirmation study has {cases} configurations on {datasets} generated datasets, with dimensions 7 or 10 and six observations per class. Each method is timed {repeats} times per configuration. Reported totals sum the median within each configuration; these settings share data and are not {cases} independent population trials. Table~\\ref{{tab:search-confirmation}} separates full evaluations from time spent in the integer master.\n",
        f"独立确认研究在 {datasets} 份生成数据上设置 {cases} 个配置，维度为七或十，每类六个观测。各方法在每个配置下计时 {repeats} 次，下面的总数先在各配置内取中位数再求和。这些设置共享数据，不是 {cases} 次独立的总体试验。表~\\ref{{tab:search-confirmation}} 将完整评价次数与整数主问题的时间分开。\n")
    rows = []
    for method in ("master", "baseline", "curvature"):
        total = stage["totals"][method]
        runs = stage["run_totals"][method]
        rows.append([tr(language, *SEARCH_NAMES[method]), fmt(total["completed_evaluations"]["total"], 0),
                     fmt(total["master_seconds"]["total"]), fmt(total["seconds"]["total"]),
                     f"{runs['statuses'].get('optimal', 0)}/{runs['cases']}"])
    text += report_table(tr(language,
        "Current confirmation comparison. Counts and seconds sum within-configuration medians; successful-run counts include all three timings. Every full evaluation makes two continuous solves.",
        "当前确认比较。次数与秒数是逐配置中位数的总和，成功运行数包含全部三次计时。每次完整评价执行两个连续求解。"),
        [tr(language, "Search", "搜索"), tr(language, r"\shortstack{Evaluated\\subsets}", r"\shortstack{评价\\组合数}"),
         tr(language, r"\shortstack{Master\\seconds}", r"\shortstack{主问题\\秒数}"),
         tr(language, r"\shortstack{Total\\seconds}", r"\shortstack{总\\秒数}"), tr(language, "Optimal runs", "最优运行")],
        rows, label="tab:search-confirmation", columns="@{}Yrrrr@{}")
    a, b = stage["totals"]["curvature"], stage["totals"]["baseline"]
    reduction = 100 * (1 - a["completed_evaluations"]["total"] / b["completed_evaluations"]["total"])
    increase = 100 * (a["seconds"]["total"] / b["seconds"]["total"] - 1)
    pairs = validation.paired()
    fewer = sum(x["completed_evaluations"] < y["completed_evaluations"] for _, x, y in pairs)
    equal = sum(x["completed_evaluations"] == y["completed_evaluations"] for _, x, y in pairs)
    more = len(pairs) - fewer - equal
    faster = sum(x["seconds"] < y["seconds"] for _, x, y in pairs)
    text += tr(language,
        f"The revised search reduces evaluations from {b['completed_evaluations']['total']:.0f} to {a['completed_evaluations']['total']:.0f}, a {reduction:.1f}\\% reduction. Fewer/equal/more evaluations occur in {fewer}/{equal}/{more} cases. Total time changes from {b['seconds']['total']:.3f} to {a['seconds']['total']:.3f} s ({increase:+.1f}\\%), with faster runs in {faster}/{len(pairs)} cases. The integer master costs more, so saved inner evaluations do not yield an overall speedup here. The unchanged GP cannot be credited as the cause of the evaluation reduction.\n",
        f"改进使评价次数从 {b['completed_evaluations']['total']:.0f} 降到 {a['completed_evaluations']['total']:.0f}，减少 {reduction:.1f}\\%。评价次数减少/相同/增加的配置分别为 {fewer}/{equal}/{more}。总时间从 {b['seconds']['total']:.3f} 变为 {a['seconds']['total']:.3f} 秒（{increase:+.1f}\\%），在 {faster}/{len(pairs)} 个配置上更快。整数主问题更昂贵，所以减少的内层评价在这里没有带来整体提速。未改变的 GP 不能被视为这次评价减少的原因。\n")
    text += tr(language,
        "Figure~\\ref{fig:search-confirmation} shows the paired cases rather than only the totals: fewer evaluations are preferable, while a time ratio above one means slower.\n",
        "图~\\ref{fig:search-confirmation} 展示配对配置，而不仅给出总数：评价次数越少越好，时间比大于一则表示更慢。\n")
    text += report_figure("search_confirmation", tr(language,
        "Full subset evaluations (upper panel) and revised/original time (lower panel). The dotted line at one means equal time. Cases 1--8 use half hinge and 9--16 logistic loss; within each block, cases are ordered by dimension, seed, and radius pair.",
        "完整组合评价次数（上图）与改进/原始时间比（下图）。虚线一表示时间相同。配置 1--8 使用半合页损失，9--16 使用逻辑损失；各组内依次按维度、种子和半径对排列。"), "fig:search-confirmation", language)
    maximum = max(result["upper"] - record["reference"]["lower"] for record in stage["raw"].values()
                  for repeated in record["repeats"] for result in repeated["methods"].values())
    all_runs = sum(v["cases"] for v in stage["run_totals"].values())
    verified = sum(v["statuses"].get("optimal", 0) for v in stage["run_totals"].values())
    text += tr(language,
        f"The confirmation solves {verified}/{all_runs} runs to their numerical target; the largest feasible-risk excess over the exhaustive lower reference is {fmt(maximum, scientific=True)}. Development included one reported master infeasibility; after the common numerical repair, the fixed rule passes 26/26 integration cases. All development comparisons and the failed run are retained in \\path{{docs/experiment_development_record.md}}. The new efficiency evidence is limited to these small cases.\n",
        f"确认实验有 {verified}/{all_runs} 次运行达到数值目标，相对全部组合枚举下界的最大可行风险超出为 {fmt(maximum, scientific=True)}。开发中曾有一次主问题被报告为不可行；采用共同的数值修复后，固定规则通过了 26/26 个集成实例。全部开发对照和失败运行保留在 \\path{{docs/experiment_development_record.md}} 中。新的效率证据仅限于这些小规模实例。\n")
    return text


def ablation_report(evidence, language):
    if evidence is None:
        return tr(language, r"\textbf{DRAFT: the new ablation has not been recorded.}",
                  r"\textbf{草稿：新的消融结果尚未记录。}")
    stage = evidence.ablation
    cases, repeats = len(stage["raw"]), stage["protocol"]["timing_repeats"]
    datasets = len({record["case"]["data_sha256"] for record in stage["raw"].values()})
    text = tr(language,
        f"We rerun {cases} existing configurations on {datasets} generated datasets, with {repeats} timings per method. Both current variants solve the global master every step; only retaining the fraction bounds differs. Counts and seconds sum within-configuration medians. This reused-setting ablation is not independent confirmation.\n",
        f"在 {datasets} 份生成数据的 {cases} 个已有配置上重新对照，各方法计时 {repeats} 次。两个当前实现都每步求解全局主问题，区别仅为是否保留分式下界。次数与秒数先在配置内取中位数再求和。这是复用配置的消融，不是独立确认。\n")
    rows = []
    names = {"cuts_tangent": ("Tangent", "切线"), "cuts_fraction": ("Tangent + fraction", "切线＋分式"),
             "gp_tangent": ("Earlier prediction + tangent", "较早预测＋切线"),
             "gp_fraction": ("Earlier prediction + fraction", "较早预测＋分式")}
    for method in names:
        total, runs = stage["totals"][method], stage["run_totals"][method]
        rows.append([tr(language, *names[method]), fmt(total["completed_evaluations"]["total"], 0),
                     fmt(total["master_seconds"]["total"]), fmt(total["seconds"]["total"]),
                     f"{runs['statuses'].get('optimal', 0)}/{runs['cases']}"])
    text += report_table(tr(language,
        f"Subset-search ablation ({repeats} timings per configuration). The first two rows compare bounds within the current algorithm; the last two compare earlier complete search procedures.",
        f"组合搜索消融，每配置计时 {repeats} 次。前两行比较当前算法的下界，后两行对照较早的完整搜索流程。"),
        [tr(language, "Search", "搜索"), tr(language, "Evaluations", "评价次数"),
         tr(language, "Master (s)", "主问题（秒）"), tr(language, "Total (s)", "总计（秒）"),
         tr(language, "Optimal runs", "最优运行")], rows, label="tab:search-ablation", columns="@{}Yrrrr@{}")
    a, b = (stage["totals"][method] for method in ("cuts_fraction", "cuts_tangent"))
    queries = 100 * (a["completed_evaluations"]["total"] / b["completed_evaluations"]["total"] - 1)
    seconds = 100 * (a["seconds"]["total"] / b["seconds"]["total"] - 1)
    text += tr(language,
        f"Retaining fractions changes full evaluations by {queries:+.1f}\\% and total time by {seconds:+.1f}\\%. The earlier procedures also differ in candidate pools and periodic master updates, so their comparison cannot isolate prediction alone. Figure~\\ref{{fig:search-ablation}} displays the current paired comparison.\n",
        f"保留分式使完整评价次数变化 {queries:+.1f}\\%，总时间变化 {seconds:+.1f}\\%。较早流程还使用不同的候选池与定期主问题更新，因此不能单独归因于预测。图~\\ref{{fig:search-ablation}} 展示当前两种实现的配对比较。\n")
    if seconds > 0 and a["master_seconds"]["total"] > b["master_seconds"]["total"]:
        text += tr(language,
            "The extra integer-master cost offsets the saved inner work. ",
            "整数主问题增加的开销抵消了内层节省。")
    if max(stage["totals"][method]["seconds"]["total"] for method in ("gp_tangent", "gp_fraction")) < min(
            total["seconds"]["total"] for total in (a, b)):
        text += tr(language,
            "The earlier complete procedures are faster in this cohort. Removing prediction simplifies the algorithm; these timings do not show that prediction has no value.\n",
            "较早的完整流程在这组实例上更快。移除预测是为了简化算法；这些计时不能证明预测没有价值。\n")
    text += report_figure("search_ablation", tr(language,
        "Fraction/tangent ratios for full evaluations and total time. Below one is better; one means equal cost. Each point is a within-configuration median ratio.",
        "分式实现相对切线实现的完整评价次数比与总时间比。小于一表示更省，一表示相同；各点为配置内中位数的比。"), "fig:search-ablation", language)
    maximum = max(result["upper"] - record["reference"]["lower"] for record in stage["raw"].values()
                  for repeated in record["repeats"] for result in repeated["methods"].values())
    runs = sum(total["cases"] for total in stage["run_totals"].values())
    optimal = sum(total["statuses"].get("optimal", 0) for total in stage["run_totals"].values())
    text += tr(language,
        f"{optimal}/{runs} runs meet the numerical target; the maximum feasible-risk excess over the exhaustive lower reference is {fmt(maximum, scientific=True)}. These small cases establish neither large-instance speed nor new statistical performance. Earlier failures and prediction comparisons remain in \\path{{docs/experiment_development_record.md}}.\n",
        f"{optimal}/{runs} 次运行达到数值目标，相对全部组合枚举下界的最大可行风险超出为 {fmt(maximum, scientific=True)}。这些小实例不能证明大规模提速或新的统计表现。较早的失败与预测对照保留于 \\path{{docs/experiment_development_record.md}}。\n")
    return text


def report_figure(stem, caption, label, language):
    suffix = "_zh" if language == "zh" else ""
    return (rf"\ResearchFigure{{{stem}{suffix}}}{{{caption}}}{{{label}}}" + "\n"
            + r"\FloatBarrier" + "\n")

def recovery_report(e, language):
    text = tr(language,
        "Figure~\\ref{fig:recovery} now asks whether those useful coordinates are the true informative ones. Smaller FDP/NDP is better. Values near the random expectation 0.75 indicate little recovery beyond uniformly choosing three coordinates.\n",
        "图~\\ref{fig:recovery} 接着考察这些坐标是否就是真正的信号变量。FDP/NDP 越小越好；接近随机期望 $0.75$ 时，说明恢复表现与均匀选择三个坐标相差不大。\n")
    comparisons = []
    for dataset in ("mean", "mixture"):
        risk, screen = (e.lookup("statistical", dataset, m) for m in ("risk", "mmd"))
        if risk and screen:
            comparisons.append((dataset, number(risk, "fdp_mean"), number(screen, "fdp_mean")))
    for dataset, risk, screen in comparisons:
        text += tr(language,
            f"For {escaped(dataset)} differences, global FDP/NDP is {fmt(risk)}, versus {fmt(screen)} for MMD screening. ",
            f"{title_label(dataset, language)}中，全局选择的 FDP/NDP 为 {fmt(risk)}，MMD 筛选为 {fmt(screen)}。")
    if comparisons:
        text += tr(language,
            "Thus a high rejection probability can coexist with selecting irrelevant variables: the testing and recovery questions are different.\n",
            "因此，较高的检出概率仍可能伴随选入无关变量；检验与恢复回答的是不同问题。\n")
    text += report_figure("recovery_and_prediction", tr(language,
        "Variable-recovery errors: smaller is better. FDP and NDP coincide for the six three-variable selectors. The lines show perfect recovery (0) and the expected random error (0.75); the observed random selector is a separate point. Points and intervals use training seeds.",
        "变量恢复错误，越小越好。六种三变量选择方法的 FDP 与 NDP 相等。参考线为完全恢复的零值和随机期望错误 $0.75$；实际随机方法另画为一个点。点和区间以训练种子为单位。"), "fig:recovery", language)
    return text

def classifier_report(e, language):
    text = tr(language,
        "Figure~\\ref{fig:prediction} compares the same independently fitted SVM on the six practical supports. Lower balanced error means better prediction. This does not reuse the robust selector's fitted test.\n",
        "图~\\ref{fig:prediction} 比较六种实际组合上独立拟合的同一种 SVM。平衡错误率越低，预测越好；这里没有直接沿用鲁棒选择阶段拟合出的检验。\n")
    for dataset in ("mean", "mixture"):
        risk, screen = (e.lookup("statistical", dataset, m) for m in ("risk", "mmd"))
        if risk and screen and number(risk, "svm_balanced_error_mean") is not None:
            text += tr(language,
                f"For {escaped(dataset)} differences, errors are {fmt(number(risk, 'svm_balanced_error_mean'))} for global selection and {fmt(number(screen, 'svm_balanced_error_mean'))} for MMD screening. ",
                f"{title_label(dataset, language)}中，全局选择与 MMD 筛选的错误率分别为 {fmt(number(risk, 'svm_balanced_error_mean'))} 和 {fmt(number(screen, 'svm_balanced_error_mean'))}。")
    text += tr(language,
        "The robust selection objective does not consistently produce the best inputs for this different prediction model.\n",
        "鲁棒选择目标并没有一致地为这个不同的预测模型提供最好的输入变量。\n")
    text += report_figure("prediction_error", tr(language,
        "Independent SVM balanced error: smaller is better and 0.5 is chance performance. All six selectors use the same fitting samples and model settings. Points are seed means and bars are 95\\% intervals.",
        "独立 SVM 的平衡错误率，越小越好，$0.5$ 为随机判断水平。六种选择方法使用相同的拟合样本和模型参数。点为种子均值，误差线为 $95\\%$ 区间。"), "fig:prediction", language)
    text += r"\subsection{" + tr(language, "How the fitted robust bound compares with fresh-data error", "拟合的鲁棒上界与新样本错误率有何不同") + "}\n"
    text += tr(language,
        "A separate robust classifier is also refitted on each selected support. Let $p_0(x)$ be its probability of assigning class zero. A randomized decision draws its class from that probability. Its balanced error and the SVM's hard-decision balanced error are\n",
        "每个组合还用独立样本重新拟合一个鲁棒分类器。记 $p_0(x)$ 为将观测判为类别零的概率；随机化判断按这个概率抽取类别。它的平衡错误率与 SVM 确定性判断的平衡错误率分别为\n")
    text += r"\begin{equation}\label{eq:errors}\begin{aligned}" + "\n"
    text += r"\operatorname{BER}_{\rm hard}&=\tfrac12(\operatorname{FPR}+\operatorname{FNR}),\\" + "\n"
    text += r"\operatorname{BER}_{\rm random}&=\tfrac12\{\mathbb E_0[1-p_0(X)]+\mathbb E_1[p_0(Y)]\}." + "\n"
    text += r"\end{aligned}\end{equation}" + "\n"
    text += tr(language,
        "FPR and FNR are the two classwise wrong-decision rates. Unlike a hard classifier, a confident but non-deterministic probability still contributes randomized error. These two error definitions are not interchangeable. Table~\\ref{tab:head} compares the robust classifier's fitted bound with its randomized error on new data and under three shifts toward the other class. The shifts have magnitudes 0, 0.25 and 0.50; their maximum is a measured stress error, not a supremum over every distribution in a Wasserstein ball.\n",
        "FPR 与 FNR 是两类各自的错误判断比例。与确定性分类不同，只要输出概率不是零或一，随机化判断仍可能出错，二者不能混用。表~\\ref{tab:head} 比较鲁棒分类器的拟合上界、新样本随机化错误率，以及向另一类平移后的错误率。平移大小为 $0$、$0.25$ 和 $0.50$；其中最大值是实际测得的压力错误，不是对 Wasserstein 球内所有分布取上确界。\n")
    rows, all_pairs, all_above = [], 0, 0
    for dataset in MECHANISMS:
        source = [r for r in e.rows if r["suite"] == "statistical" and r["dataset"] == dataset and r["method"] == "risk"]
        pairs = [(number(r, "head_robust_upper") / 2, number(r, "robust_randomized_balanced_error"),
                  number(r, "robust_worst_stress_randomized_balanced_error")) for r in source
                 if number(r, "head_robust_upper") is not None and number(r, "robust_randomized_balanced_error") is not None]
        if not pairs:
            continue
        above = sum(error > bound + 1e-8 for bound, error, _ in pairs)
        stress = [value for _, _, value in pairs if value is not None]
        rows.append([title_label(dataset, language), len(pairs), fmt(np.mean([v[0] for v in pairs])),
                     fmt(np.mean([v[1] for v in pairs])), fmt(np.mean(stress)) if stress else "---", f"{above}/{len(pairs)}"])
        all_pairs += len(pairs)
        all_above += above
    if rows:
        text += report_table(tr(language,
            "Refitted robust classifier on globally selected variables. All errors are randomized balanced errors. The fitted bound is the classwise sum-risk upper bound divided by two. Stress is the within-seed maximum over three specified shifts; final columns count paired trials exceeding the fitted bound.",
            "全局选择变量上重新拟合的鲁棒分类器。错误均为随机化平衡错误率。拟合界为两类总风险上界的一半。压力值为每个种子三次规定平移的最大错误，末列按配对运行统计超过拟合界的次数。"),
            [tr(language, "Difference", "差异"), "$n$", tr(language, "Fitted bound", "拟合界"),
             tr(language, "Fresh error", "新样本错误"), tr(language, "Shift maximum", "平移最大值"), tr(language, "Above bound", "超过拟合界")],
            rows, label="tab:head", columns="@{}p{23mm}rrrrr@{}")
        text += tr(language,
            f"Fresh errors exceed the fitted empirical bound in {all_above}/{all_pairs} paired trials. This is not a certificate failure: the bound concerns the fitted empirical distribution ball, and the unknown population has not been shown to belong to it. A low empirical worst-case loss cannot be read as a demonstrated population-error guarantee.\n",
            f"{all_above}/{all_pairs} 次配对运行的新样本错误率超过拟合的经验界。这不能称为证书失效：该界针对拟合的经验分布球，并未证明未知总体包含在球内。较低的经验最坏损失不能直接理解成已经验证的总体错误率保证。\n")
    text += r"\subsection{" + tr(language, "Changing the uncertainty radius", "改变不确定性半径") + "}\n"
    text += tr(language,
        "Table~\\ref{tab:radius} changes the radius used both for selection and for refitting the robust classifier. It checks the whole procedure, rather than holding the selected variables fixed. The three radii are preset alternatives, not choices made from the final test outcomes.\n",
        "表~\\ref{tab:radius} 同时改变选择阶段与重新拟合鲁棒分类器的半径，考察整个过程，而不是固定变量后只改分类器。三个半径预先设置，并非根据最终测试结果选择。\n")
    text += sensitivity_report(e, language)
    return text

def implementation_report(e, language, validation=None, ablation=None):
    stat = e.configs.get("statistical", {})
    text = tr(language,
        "Centering and sample-standard-deviation scaling use pooled selection observations and are then fixed. The common RBF uses $b^2$ equal to their median squared distance. SVMs use $C=1$ and a median bandwidth from their own fitting sample. The robust classifier uses the selection loss and radius.\n\n",
        "中心化与样本标准差缩放用合并的选择观测拟合，之后固定。共同 RBF 的 $b^2$ 为其平方距离中位数。SVM 使用 $C=1$，中位带宽来自自身拟合样本。鲁棒分类器使用选择阶段的损失与半径。\n\n")
    text += tr(language,
        "Additive MMD uses fixed $\\lambda=0.1$, without cross-validation. Supports are enumerated at $D\\le12$; larger cases use forward selection and at most ten improving swaps. Testing bandwidths never use final labels. L1 logistic ranking uses liblinear, $C=1$, and at most 2,000 iterations. At its time limit, greedy keeps its last completed stage and fills any remaining slots by MMD screening.\n\n",
        "加性 MMD 固定 $\\lambda=0.1$，未做交叉验证。$D\\le12$ 时枚举组合，更大问题使用前向选择与至多十次改进交换。测试带宽不使用最终标签。L1 逻辑排序使用 liblinear、$C=1$ 与至多 $2{,}000$ 次迭代。贪心达到时间上限时保留最后完成的阶段，并用 MMD 筛选补足剩余变量。\n\n")
    if stat:
        text += tr(language,
            f"The global and greedy robust selectors each have a {stat['time_limit']}-s selection allowance. The robust radius is {stat['theta']:.2f}; numerical gap tolerance $10^{{-6}}+10^{{-5}}|\\mathrm{{UB}}|$ is used only by global search. The remaining selectors have no enforced wall-clock cutoff. The earlier global search starts from MMD screening and uses five fractional starting rounds. Randomized robust predictions use their fitted class probabilities; SVM hard decisions use threshold zero, with ties assigned to class zero.\n\n",
            f"全局与鲁棒贪心选择各有 {stat['time_limit']} 秒的选择时间限制。鲁棒半径为 {stat['theta']:.2f}；数值间隙容差 $10^{{-6}}+10^{{-5}}|\\mathrm{{UB}}|$ 仅用于全局搜索。其余选择方法未设置硬时间截止。较早的全局搜索以 MMD 筛选启动，并使用五轮分数起始求解。鲁棒随机化预测使用拟合的类别概率；SVM 的确定性判断以零为阈值，并列时判为类别零。\n\n")
    if ablation:
        settings = ablation.ablation["protocol"]["settings"]
        text += report_table(tr(language, "Fixed settings of the reused-configuration search ablation.", "复用配置的搜索消融固定设置。"),
            [tr(language, "Item", "项目"), tr(language, "Setting", "设置")],
            [[tr(language, "Cases and data", "配置与数据"), tr(language,
                "16 configurations on 4 datasets; $(D,d)=(7,2),(10,3)$; 6 observations/class; seeds 19,23.",
                "四份数据上的十六个配置；$(D,d)=(7,2),(10,3)$；每类六个观测；种子 19、23。")],
             [tr(language, "Loss and radius", "损失与半径"), tr(language,
                "Half hinge or logistic; radii $(0.12,0.20)$ or $(0.30,0.45)$.",
                "半合页或逻辑损失；半径为 $(0.12,0.20)$ 或 $(0.30,0.45)$。")],
             [tr(language, "Generated coordinates", "生成坐标"), tr(language,
                "Gaussian class zero; two class-one means shifted by 1.4,0.9; nuisance scales 0.3--3; no fitted standardization.",
                "类别零为高斯；类别一两个均值平移 1.4、0.9；噪声尺度 0.3--3；不做拟合标准化。")],
             [tr(language, "Common starting subset", "共同起始组合"), r"$|\bar x_{0j}-\bar x_{1j}|/(s_{0j}+s_{1j}+10^{-12})$; " + tr(language,
                "sample SD with ddof=0, index ties; no true signal identities.", "标准差用总体分母，并列按索引；不使用真实信号身份。")],
             [tr(language, "Time and precision", "时间与精度"), rf"{settings['time_limit']:.0f} " + tr(language,
                "s, one thread; $\\varepsilon=2\\times10^{-5}$, relative tolerance zero; inner/master targets $\\varepsilon/4$.",
                "秒，一个线程；$\\varepsilon=2\\times10^{-5}$，相对容差为零；内层/主问题目标为 $\\varepsilon/4$。")],
             [tr(language, "Current subset choice", "当前组合选择"), tr(language,
                "Global integer master at every step; tangent bounds, with or without stored fractions; no prediction, pool, or fractional root calls.",
                "每步求解全局整数主问题；使用切线下界，比较是否保留分式；无预测、候选池或分数根节点调用。")]],
            label="tab:current-settings", columns="@{}p{36mm}Y@{}")
        text += tr(language,
            "Counts use distinct fully evaluated binary subsets; each current evaluation targets $\\varepsilon/4$. Physical refinements and master time remain in the raw traces. Three timings per configuration rotate execution order. Earlier prediction settings remain documented with their source snapshots.\n\n",
            "次数统计完整评价过的不同二元组合，每次当前评价的精度目标为 $\\varepsilon/4$。实际精度重算与主问题时间保留在原始轨迹中。每配置计时三次，轮换执行顺序。较早预测设置保留在对应文档与源码快照中。\n\n")
    environment = json.loads((e.root / "environment.json").read_text(encoding="utf-8")) if (e.root / "environment.json").exists() else {}
    hardware = json.loads((e.root / "hardware_cpu.json").read_text(encoding="utf-8")) if (e.root / "hardware_cpu.json").exists() else {}
    packages = environment.get("packages", {})
    engine = environment.get("linear_integer_backend", {}).get("highs_version", "unrecorded")
    text += tr(language,
        f"Reference computations use {escaped(str(hardware.get('Name', 'unrecorded host')).strip())}, one solver thread and SciPy-backed HiGHS {escaped(engine)}; logistic checks use CVXPY {escaped(packages.get('cvxpy', 'unrecorded'))} and Clarabel {escaped(packages.get('clarabel', 'unrecorded'))}.\n\n",
        f"参考计算使用 {escaped(str(hardware.get('Name', '未记录主机')).strip())}、一个求解线程和 SciPy 内置的 HiGHS {escaped(engine)}；逻辑损失核查使用 CVXPY {escaped(packages.get('cvxpy', '未记录'))} 与 Clarabel {escaped(packages.get('clarabel', '未记录'))}。\n\n")
    failures = sum(r["selection_status"] in FAILURES for r in e.rows)
    limits = sum(r["selection_status"] == "time_limit" for r in e.rows)
    text += tr(language,
        f"The earlier study's {len(e.rows)} records include {failures} failures and {limits} time limits; complete metrics, supports and traces remain in \\path{{results/research_20261003}}. Historical comparisons and failures are indexed in \\path{{docs/experiment_development_record.md}}. The new ablation is separate under \\path{{reports/verification/20261012/no_prediction}}; manifests identify all included inputs.\n",
        f"较早研究的 {len(e.rows)} 条记录包含 {failures} 次失败和 {limits} 次达到时间上限；完整指标、组合与轨迹保留于 \\path{{results/research_20261003}}。历史对照与失败索引见 \\path{{docs/experiment_development_record.md}}。新消融单独存于 \\path{{reports/verification/20261012/no_prediction}}；清单记录全部使用的输入。\n")
    return text


def render(evidence, template, *, language="en", validation=None, ablation=None):
    blocks = {"OVERVIEW": concise_overview(evidence, language, ablation),
              "SETUP_TABLE": setup_report(evidence, language), "SEARCH_TABLE": ablation_report(ablation, language),
              "COMPUTATIONAL_TABLE": reference_computation(evidence, language),
              "STATISTICAL_TABLE": statistical_report(evidence, language), "REAL_TABLE": real_report(evidence, language),
              "SENSITIVITY_TABLE": classifier_report(evidence, language), "RECOVERY_TABLE": recovery_report(evidence, language),
              "REMEDIATION": implementation_report(evidence, language, validation, ablation)}
    for marker in MARKERS:
        token = "%%" + marker + "%%"
        if template.count(token) != 1:
            raise ValueError(f"Template must have exactly one {token}")
        template = template.replace(token, blocks[marker])
    if re.search(r"%%[A-Z_]+%%", template):
        raise ValueError("Unreplaced report marker")
    return "\n".join(line.rstrip() for line in template.splitlines()) + "\n"



def input_fingerprint(path, project, canonical=None):
    """Rebase a known snapshot only when its bytes match the canonical source."""
    path, project = path.resolve(), project.resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.is_relative_to(project):
        return path.relative_to(project).as_posix(), digest
    if canonical is not None:
        candidate = (project / canonical).resolve()
        if (candidate.is_relative_to(project) and candidate.is_file()
                and hashlib.sha256(candidate.read_bytes()).hexdigest() == digest):
            return candidate.relative_to(project).as_posix(), digest
    return str(path), digest


def main(argv=None):
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument("--root", type=Path, default=project / "results/research_20261003")
    parser.add_argument("--template", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figure-root", type=Path)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--ablation-root", type=Path)
    parser.add_argument("--language", choices=("en", "zh"), default="en")
    parser.add_argument("--draft", action="store_true", help="Explicitly mark incomplete evidence as a draft")
    args = parser.parse_args(argv)
    suffix = "_zh" if args.language == "zh" else ""
    if args.template is None:
        args.template = args.project_root / "reports" / ("report_template" + suffix + ".tex")
    if args.figure_root is None:
        args.figure_root = args.project_root / "reports/figures"
    if args.output is None:
        args.output = ((Path(tempfile.mkdtemp(prefix="robusthw-report-draft-")) if args.draft
                        else args.project_root / "reports") / ("experiment_report" + suffix + ".tex"))
    if args.draft:
        temporary = Path(tempfile.gettempdir()).resolve()
        if not args.output.resolve().is_relative_to(temporary) or args.output.parent.resolve() == temporary:
            raise ValueError("Draft output must be inside a task-specific system TEMP directory")
    evidence = Evidence(args.project_root.resolve(), args.root.resolve(), args.draft)
    if not args.draft and args.ablation_root is None:
        raise ValueError("Final bilingual report requires sealed current-search ablation evidence; reference-only builds must use --draft")
    validation = SearchEvidence(args.validation_root.resolve(), args.draft) if args.validation_root else None
    ablation = AblationEvidence(args.ablation_root.resolve(), args.draft) if args.ablation_root else None
    source = render(evidence, args.template.read_text(encoding="utf-8"), language=args.language,
                    validation=validation, ablation=ablation)
    files = [args.root / "summary.json", args.root / "per_trial.csv", args.template,
             Path(__file__), Path(__file__).with_name("plot_results.py"),
             Path(__file__).with_name("report_evidence.py")]
    files += [args.project_root / "configs" / f"{CONFIG_FILES[suite]}.json" for suite in SUITES]
    files.append(args.project_root / "docs/data_sources.json")
    for metadata in ("environment.json", "hardware_cpu.json", "verification/tests.txt"):
        path = args.root / metadata
        if path.is_file():
            files.append(path)
    files += [args.root / row["record_file"].replace("\\", "/") for row in evidence.rows]
    provenance = args.root / "remediation/initial_failures.json"
    if provenance.exists():
        files.append(provenance)
    elif args.draft:
        archive = args.project_root / "_archive/numerical_failures_20261003"
        if archive.exists():
            files += list(archive.rglob("records/*.json"))
    if validation is not None:
        files += validation.input_files
    if ablation is not None:
        files += ablation.input_files
    figure_stems = set(re.findall(r"\\ResearchFigure\{([a-z_]+)\}", source))
    figure_stems.update(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([a-z_]+)\.pdf\}", source))
    source_aliases = {
        args.template.resolve(): "reports/report_template" + suffix + ".tex",
        Path(__file__).resolve(): "scripts/build_report.py",
        Path(__file__).with_name("plot_results.py").resolve(): "scripts/plot_results.py",
        Path(__file__).with_name("report_evidence.py").resolve(): "scripts/report_evidence.py",
    }
    input_hashes = {}
    for path in files:
        label, digest = input_fingerprint(path, args.project_root, source_aliases.get(path.resolve()))
        input_hashes[label] = digest
    for stem in sorted(figure_stems):
        asset = args.figure_root / (stem + ".pdf")
        if asset.is_file():
            input_hashes["reports/figures/" + stem + ".pdf"] = hashlib.sha256(asset.read_bytes()).hexdigest()
        elif not args.draft:
            raise ValueError(f"Required figure PDF is missing: {asset}")
    manifest = {"state": "draft" if args.draft else "complete", "language": args.language,
                    "reference_records": len(evidence.rows), "expected_reference_records": len(evidence.expected),
                    "missing_reference_records": len(evidence.missing),
                    "records": len(evidence.rows), "expected_records": len(evidence.expected),
                    "missing_records": len(evidence.missing),
                    "confirmation_cases": validation.confirmation_cases if validation else 0,
                    "ablation_cases": ablation.ablation_cases if ablation else 0,
                    "evidence_cohorts_separated": True, "inputs": input_hashes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"source": str(args.output), **{k: v for k, v in manifest.items() if k != "inputs"}}))



if __name__ == "__main__":
    main()
