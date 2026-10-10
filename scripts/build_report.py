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
from report_evidence import SearchEvidence

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
           "STATISTICAL_TABLE", "REAL_TABLE", "SENSITIVITY_TABLE", "RATIO_TABLE", "REMEDIATION")


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
            + r"\bottomrule\end{tabularx}\end{table}" + "\n")



def concise_overview(e, language, validation=None):
    prefix = (rf"\textbf{{DRAFT SNAPSHOT: {len(e.rows)}/{len(e.expected)} reference records; "
              rf"{len(e.missing)} missing.}} " if e.draft else "")
    en = ("We evaluate robust feature selection using a separate validation of subset search "
          "and a frozen study of statistical utility. ")
    zh = "本报告分别验证变量组合搜索的计算开销，并整理冻结研究中的统计效用。"
    if validation and validation.confirmation:
        totals = validation.confirmation["totals"]
        a, b = totals["curvature"], totals["baseline"]
        reduction = 100*(1-a["completed_evaluations"]["total"]/b["completed_evaluations"]["total"])
        extra_time = 100*(a["seconds"]["total"]/b["seconds"]["total"]-1)
        en += (f"On {validation.confirmation_cases} predeclared confirmation configurations, storing "
               f"the dual fractions already obtained by full evaluation reduces complete subset solves "
               f"by {reduction:.1f}\\%, while summed median runtime changes by {extra_time:+.1f}\\%. ")
        zh += (f"在 {validation.confirmation_cases} 个预先声明的确认配置上，保留完整评估已经得到的对偶分式，"
               f"使完整组合求解次数减少 {reduction:.1f}\\%，总中位运行时间变化为 {extra_time:+.1f}\\%。")
    en += ("The reference study compares four synthetic alternatives and two public datasets using "
           "independent selection, classifier fitting and testing. Global empirical-risk optimality "
           "does not yield uniformly better testing power or prediction. All comparisons, time limits "
           "and failures are retained; results from earlier search versions remain labeled separately.")
    zh += ("参考研究比较四种合成备择与两个公开数据集，变量选择、分类器拟合与测试相互独立。"
           "经验风险的全局最优并不带来一致更好的检验功效或预测。全部对照、时间上限和失败均保留，"
           "早期搜索版本的结果单独标明。")
    return prefix + tr(language, en, zh)





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
    text = ""
    for dataset in MECHANISMS:
        risk = e.lookup("statistical", dataset, "risk")
        screen = e.lookup("statistical", dataset, "mmd")
        if not risk or not screen:
            continue
        n, diff = paired(e.rows, "statistical", dataset, "power")
        dmean, lo, hi = diff
        if dmean is None:
            continue
        text += tr(language,
                   f"For {escaped(dataset)} data, global power is {fmt(number(risk, 'power_mean'))}, versus {fmt(number(screen, 'power_mean'))} for marginal screening; the paired difference is {dmean*100:.1f} percentage points [{lo*100:.1f}, {hi*100:.1f}] ($n={n}$). ",
                   f"在{title_label(dataset, language)}上，全局选择的功效为 {fmt(number(risk, 'power_mean'))}，MMD 筛选为 {fmt(number(screen, 'power_mean'))}；配对差为 {dmean*100:.1f} 个百分点 [{lo*100:.1f}, {hi*100:.1f}]（$n={n}$）。")
    risk, greedy = e.lookup("statistical", "mean", "risk"), e.lookup("statistical", "mean", "greedy")
    if risk and greedy:
        text += tr(language,
                   f"All {solved_count(risk)}/{risk['n_total']} mean-shift global searches meet the optimization tolerance, yet greedy power is {fmt(number(greedy, 'power_mean'))} versus {fmt(number(risk, 'power_mean'))}. Empirical-risk optimality does not maximize downstream power.\n",
                   f"均值变化的全局搜索中有 {solved_count(risk)}/{risk['n_total']} 次达到优化容差，但贪心功效为 {fmt(number(greedy, 'power_mean'))}，全局选择为 {fmt(number(risk, 'power_mean'))}。经验风险最优不等于后续检验功效最大。\n")
    rows = []
    for dataset in MECHANISMS:
        r = e.lookup("statistical", dataset, "wang")
        if r and number(r, "native_additive_power_mean") is not None:
            rows.append([title_label(dataset, language), int(r["native_additive_power_n"]),
                         estimate(r, "power"), estimate(r, "native_additive_power"),
                         estimate(r, "native_additive_null_rejection")])
    text += report_table(
        tr(language, "Two evaluators on the Wang-selected supports. Common power uses only the chosen coordinates; native power retains the signed additive weights. Entries are seed means and 95\\% intervals.",
           "Wang 所选组合的两种评价。共同功效仅使用所选坐标；原生功效还保留有符号加性权重。数值为种子均值与 95\\% 区间。"),
        [tr(language, "Alternative", "备择"), "$n$", tr(language, "Common power", "共同功效"),
         tr(language, "Native power", "原生功效"), tr(language, "Native null", "原生零假设")],
        rows, label="tab:native", columns="@{}p{25mm}rrrr@{}")
    rows, all_pairs, all_above = [], 0, 0
    for dataset in MECHANISMS:
        source = [r for r in e.rows if r["suite"] == "statistical" and r["dataset"] == dataset and r["method"] == "risk"]
        pairs = [(number(r, "head_robust_upper")/2, number(r, "robust_randomized_balanced_error"),
                  number(r, "robust_worst_stress_randomized_balanced_error")) for r in source
                 if number(r, "head_robust_upper") is not None and number(r, "robust_randomized_balanced_error") is not None]
        if not pairs:
            continue
        above = sum(y > bound + 1e-8 for bound, y, _ in pairs)
        stress = [v for _, _, v in pairs if v is not None]
        rows.append([title_label(dataset, language), len(pairs), fmt(np.mean([v[0] for v in pairs])),
                     fmt(np.mean([v[1] for v in pairs])), fmt(np.mean(stress)) if stress else "---",
                     f"{above}/{len(pairs)}"])
        all_pairs += len(pairs)
        all_above += above
    if rows:
        text += tr(language,
                   f"For global selection, {all_above}/{all_pairs} independent randomized errors exceed their own refitted empirical head bound divided by two. The classifier bound is a different fitted quantity from the selection objective. The population was not established to lie in the fitted ball, so this excess is not a certificate failure.\n",
                   f"全局选择有 {all_above}/{all_pairs} 次独立随机化错误率高于其重新拟合的经验分类器上界的一半。这个分类器界不是变量选择目标。总体是否属于拟合的球并未得到证明，因此该超出不能称为证书失效。\n")
        text += report_table(
            tr(language, "Global-selector refitted head and independent error. Values are paired-seed means; stress is the mean of each seed's maximum over the three preset shifts. Above-bound counts use tolerance $10^{-8}$.",
               "全局选择器的重新拟合分类器界与独立错误率。数值为配对种子均值；压力错误为每个种子三次预设平移最大值的平均。超出计数使用 $10^{-8}$ 容差。"),
            [tr(language, "Alternative", "备择"), "$n$", tr(language, "Fit bound / 2", "拟合界 / 2"),
             tr(language, "Fresh error", "新样本错误"), tr(language, "Shift maximum", "平移最大值"),
             tr(language, "Above bound", "超出界")], rows, label="tab:head", columns="@{}p{22mm}rrrrr@{}")
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
    values = [number(r, "empirical_certificate_violation") for r in e.rows]
    values = [v for v in values if v is not None]
    if values:
        text += tr(language,
                   f"Across all {len(values)} budget-feasible source-atom checks, {sum(v>1e-7 for v in values)} violations exceed $10^{{-7}}$ and the largest recorded nonnegative violation is {fmt(max(values), scientific=True)}. This direct check is separate from fresh-population performance.\n",
                   f"全部 {len(values)} 次预算可行源点核验中，{sum(v>1e-7 for v in values)} 次违反量超过 $10^{{-7}}$，最大记录非负违反量为 {fmt(max(values), scientific=True)}。这个直接核验与新总体样本表现分开。\n")
    return text



def real_report(e, language):
    rows = []
    for dataset in ("wdbc", "ionosphere"):
        for method in PRIMARY:
            r = e.lookup("real", dataset, method)
            if r:
                rows.append(["WDBC" if dataset == "wdbc" else "Ionosphere", method_label(method, language),
                             estimate(r, "robust_randomized_balanced_error", real=True),
                             estimate(r, "svm_balanced_error", real=True), estimate(r, "robust_auc", real=True)])
    text = report_table(tr(language, "Independent common classifiers on real datasets. Entries are mean $\\pm$ overlapping-split SD. Robust error is randomized BER; SVM error is hard BER. These are incumbent supports when search is limited.",
                            "真实数据上的独立共同分类器。数值为均值 $\\pm$ 重叠分割标准差。鲁棒错误为随机化 BER，SVM 错误为确定性 BER；搜索受限时使用保留的可行组合。"),
                        [tr(language, "Data", "数据"), tr(language, "Selector", "选择器"),
                         tr(language, r"\shortstack{Robust\\randomized BER}", r"\shortstack{鲁棒\\随机化 BER}"),
                         "SVM BER", tr(language, "Robust AUC", "鲁棒 AUC")],
                        rows, label="tab:real", columns="@{}p{18mm}Yrrr@{}")
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
    if validation is None:
        return tr(language, r"\textbf{DRAFT: revised-search evidence is unavailable.}",
                  r"\textbf{草稿：尚无改进搜索的证据。}")
    text = tr(language,
        "The computational question is whether choosing and screening subsets more carefully reduces expensive full evaluations. Each selected binary subset is fully solved, gives an actual feasible test and contributes one tangent. No old test is transferred to a new subset.\n\n",
        "这里的计算问题是：更充分地筛选和选择组合，能否减少昂贵的完整评估？每个选中的二元组合都完整求解，产生实际可行检验，并贡献一条切线。不将旧检验转移到新组合。\n\n")
    text += tr(language,
        "The original search uses a fixed Gaussian process (GP) predictor to obtain mean risk and model standard deviation from completed evaluations, then ranks up to eight eligible proposals by mean minus standard deviation. The revision retains the same predictor, proposals and refresh interval. It also stores the safe dual fraction already returned by each full evaluation. A candidate is screened using the largest stored tangent or fraction value. The periodic master also retains these fractions. This uses existing certificates and adds no continuous oracle stage.\n\n",
        "原搜索使用固定的高斯过程（GP）预测器，根据已完成评估得到风险均值与模型标准差，再按均值减标准差排序，候选至多八个。改进保持预测器、候选和刷新间隔不变；仅额外保留每次完整评估已经得到的安全对偶分式。筛选候选时取已有切线与分式给出的最大下界，定期主问题也保留这些分式。这里使用的是已有证书，不增加连续内层求解阶段。\n\n")
    text += tr(language,
        "The implementation repeats four steps: (1) exclude proposals whose stored lower bound exceeds the incumbent minus half the target tolerance; (2) rank the remaining proposals; (3) fully solve the chosen subset and store its test, tangent and fraction; (4) refresh the global master after three completed evaluations or when no proposal survives. The actual test updates the upper bound; the master supplies the global lower bound and stopping test.\n\n",
        "实现反复执行四步：（1）排除已有下界超过当前最好风险减去一半目标容差的候选；（2）排序剩余候选；（3）完整求解选中的组合，保存其检验、切线与分式；（4）每完成三次评估，或没有候选通过筛选时，刷新全局主问题。实际检验更新上界；主问题提供全局下界并判断是否停止。\n\n")
    text += tr(language,
        "Here $T$ is the recorded support-evaluation count (the solver's visited counter). It is separate from the count meeting inner precision: a limited run may record a final evaluation that misses this target. Every recorded anchor in the current validation meets $\\varepsilon/4$, where $\\varepsilon$ is the target absolute risk tolerance. Physical oracle attempts may include refinements; continuous solves count the primal and dual solver calls inside them. Master calls and master time measure the separate integer-search cost. Numerical accuracy is checked against exhaustive support enumeration, not only the reported outer gap.\n\n",
        "这里 $T$ 是记录的组合评估次数（求解器的 visited 计数），与满足内层精度的次数分开；受限运行的末次评估可能没有达到该精度。当前验证中的每个记录锚点都达到 $\\varepsilon/4$，其中 $\\varepsilon$ 是目标绝对风险容差。实际内层尝试还可能包含精度重算；连续求解次数计入其中的原始与对偶求解。主问题次数与时间则衡量整数搜索的独立成本。数值精度与全部组合枚举结果核对，而不只检查算法自身报告的外层间隙。\n\n")
    text += r"\subsection{" + tr(language, "Development and the fixed rule", "开发过程与固定规则") + "}\n"
    text += tr(language,
        "The 26 development cases use two losses, dimensions 6 and 8 with budgets 2, class sizes 4 and 8, and seeds 3, 7 and 11; two additional cases use dimension 12, budget 3 and four observations per class. These cases use a shared deliberately poor tail-coordinate start.\n",
        "26 个开发实例包含两种损失，维度 6 与 8、预算 2、每类样本数 4 与 8，以及种子 3、7、11；另两个实例使用维度 12、预算 3 和每类四个样本。它们共用一个刻意选择的较差尾部坐标起点。\n")
    text += tr(language,
        "Stage 1 changes the prediction score or gives a refreshed master candidate priority. Stage 2 adds data-ranked proposals or chooses a tangent on the same point's dual optimal face; the latter incurs an extra continuous solve. Stage 3 retains fractions and solves a fraction master at every step. Stage 4 keeps those fractions but refreshes only periodically. Changing the score alone did not reduce evaluations. The last two stages reduce evaluations through stronger stored lower bounds, while their integer masters cost more per call.\n",
        "阶段 1 改变预测评分，或优先采用刚刷新的主问题候选。阶段 2 增加数据排序候选，或在同一点的对偶最优面上选切线；后者会额外求解一个连续问题。阶段 3 保留分式，并在每一步求解分式主问题。阶段 4 仍保留分式，但只定期刷新。单独修改评分没有减少评估次数；后两个阶段利用更强的已有下界减少评估，但每次整数主问题的成本更高。\n")
    text += development_table(validation, language)
    text += tr(language,
        "The stage-4 fraction variant has one solver-reported infeasible master and 25 of 26 optimal results; its partial total is not an all-success improvement claim. The returned incumbent satisfies the stored lower constraints. The implementation now verifies such a witness and retries the same master without presolve inside the remaining allowance. This numerical repair is applied to every compared method. The rule was fixed before confirmation; the following integration replay checks the repaired implementation on the old development cases.\n",
        "阶段 4 的分式变体有一次主问题被求解器判为不可行，26 次中只有 25 次数值最优；其部分工作总数不能作为全部成功的改进结论。返回的当前检验满足已保存的下界约束。实现现在会核验这个可行见证，并在剩余预算内关闭预处理重算同一个主问题；各对照共用该数值修复。规则在确认实验前固定；下面的集成重跑只在旧开发实例上核验修复后的实现。\n")
    if validation.integration:
        text += search_table(validation.integration, language, "tab:search-integration",
            tr(language, "Integration replay of the fixed rule after numerical repair: 26 cases, one run per method. Totals include every run; this is development verification, not a second confirmation cohort.",
               "数值修复后固定规则的集成重跑：26 个实例，每种方法运行一次。总数包含全部运行；这是开发核验，不是另一组确认数据。"))
    text += r"\FloatBarrier" + "\n"
    text += r"\subsection{" + tr(language, "Independent confirmation", "独立确认") + "}\n"
    if validation.confirmation is None:
        return text + tr(language, r"\textbf{DRAFT: confirmation is unavailable.}", r"\textbf{草稿：尚无确认结果。}")
    stage = validation.confirmation
    settings = stage["protocol"]["settings"]
    cases, repeats = len(stage["raw"]), stage["protocol"]["timing_repeats"]
    datasets = len({r["case"]["data_sha256"] for r in stage["raw"].values()})
    text += tr(language,
        f"The {cases} predeclared confirmation configurations use $(D,d)=(7,2)$ or $(10,3)$, six observations per class, seeds 19 and 23, radii $(0.12,0.20)$ or $(0.30,0.45)$, and half-hinge or logistic loss. They reuse {datasets} generated datasets across losses and radii. Generated coordinates are used directly, without the standardization fitted in the reference statistical study. Class zero is Gaussian; class one shifts its first two coordinates by 1.4 and 0.9. Nuisance scales span 0.3 to 3. Dimension and budget change the support space; unequal radii and the two losses check both asymmetric uncertainty and linear/conic solver costs. This is an optimization-effort study, not a new power study.\n",
        f"{cases} 个预先声明的确认配置使用 $(D,d)=(7,2)$ 或 $(10,3)$，每类六个样本，种子 19 与 23，半径 $(0.12,0.20)$ 或 $(0.30,0.45)$，以及半合页或逻辑损失。不同损失与半径会复用 {datasets} 份生成数据。这里直接使用生成坐标，不使用参考统计研究中的拟合标准化。类别零为高斯分布；类别一前两个坐标分别平移 1.4 与 0.9，噪声坐标的尺度从 0.3 到 3。维度与预算改变组合空间；不等半径和两种损失分别核查不对称不确定性与线性/锥求解成本。这是优化开销实验，不是新增功效实验。\n")
    text += tr(language,
        r"All methods use the same data-ranked start. The score numerator is the absolute class-mean difference; its denominator is the sum of both class standard deviations and $10^{-12}$. Standard deviations use \texttt{ddof=0}; ties are resolved by stable index order. No generating signal identity is used." + "" + "\n",
        r"各方法使用相同的数据排序起点。评分的分子是两类均值差的绝对值，分母是两类标准差与 $10^{-12}$ 三项之和。标准差使用 \texttt{ddof=0}，并列按索引稳定处理。排序不使用生成时的信号身份。" + "\n")
    text += tr(language,
        rf"The allowance is {settings['time_limit']:.0f} s per run, with one solver thread, no fractional root evaluations, absolute risk tolerance $\varepsilon=2\times10^{{-5}}$ and zero relative tolerance. Inner and master precision targets are $\varepsilon/4$. The normalized risk predictor has prior mean $1/2$, covariance $\frac14\exp[-\|z-z'\|_1/\max(1,d)]$ and ridge $\max\{{10^{{-8}},(\varepsilon/(4B))^2\}}$, where $B=1$ for half hinge and $B=2\log2$ for logistic. These parameters are fixed." + "" + "\n",
        rf"每次运行预算为 {settings['time_limit']:.0f} 秒，使用一个求解线程，不评估分数根节点；绝对风险容差为 $\varepsilon=2\times10^{{-5}}$，相对容差为零。内层与主问题精度目标均为 $\varepsilon/4$。归一化风险预测器的先验均值为 $1/2$，协方差为 $\frac14\exp[-\|z-z'\|_1/\max(1,d)]$，正则量为 $\max\{{10^{{-8}},(\varepsilon/(4B))^2\}}$，其中半合页损失的 $B=1$，逻辑损失的 $B=2\log2$。这些参数均固定。" + "\n")
    stable = all(row["query_range"][0] == row["query_range"][1] for row in stage["rows"] if row["query_range"])
    text += tr(language,
        f"Each method is timed {repeats} times per configuration, rotating execution order. Times below sum the within-case medians; query counts are unchanged across those repetitions. The {cases} configurations are paired settings on {datasets} datasets, so no independent-trial confidence interval is attached to these totals.\n" if stable else
        f"Each method is timed {repeats} times per configuration, rotating execution order. Totals below sum the within-case medians, and repeated query counts may vary. The settings reuse datasets, so totals are descriptive.\n",
        f"每个配置下各方法计时 {repeats} 次，并轮换执行顺序。下面的时间总数对每个实例先取中位数再求和；评估次数在重复间一致。{cases} 个配置是 {datasets} 份数据上的配对设置，因此不为这些总数标注独立试验置信区间。\n" if stable else
        f"每个配置下各方法计时 {repeats} 次，并轮换执行顺序。下面的总数对每个实例先取中位数再求和，重复间评估次数可能变化。配置会复用数据，因此总数仅作描述。\n")
    text += search_table(stage, language, "tab:search-confirmation",
        tr(language, "Confirmation totals of per-case medians. All physical oracle attempts equal completed evaluations $T$; each evaluation makes two continuous solves. Optimal-run counts include all timing repetitions.",
           "确认实验的逐实例中位数总计。所有实际内层尝试均等于完整评估次数 $T$；每次评估执行两个连续求解。最优运行计数包含全部计时重复。"))
    a, b = stage["totals"]["curvature"], stage["totals"]["baseline"]
    qt = 100 * (a["completed_evaluations"]["total"] / b["completed_evaluations"]["total"] - 1)
    rt = 100 * (a["seconds"]["total"] / b["seconds"]["total"] - 1)
    pairs = validation.paired()
    fewer = sum(x["completed_evaluations"] < y["completed_evaluations"] for _, x, y in pairs)
    equal = sum(x["completed_evaluations"] == y["completed_evaluations"] for _, x, y in pairs)
    more = len(pairs) - fewer - equal
    faster = sum(x["seconds"] < y["seconds"] for _, x, y in pairs)
    text += tr(language,
        f"Stored fractions reduce total evaluations from {b['completed_evaluations']['total']:.0f} to {a['completed_evaluations']['total']:.0f} ({-qt:.1f}\\%), with fewer/equal/more evaluations in {fewer}/{equal}/{more} cases. Their total median time is {a['seconds']['total']:.3f} versus {b['seconds']['total']:.3f} s ({rt:+.1f}\\%); they are faster in {faster}/{len(pairs)} cases. Fewer evaluations therefore do not establish an overall speedup. Master time increases because the stored fraction model is more costly, despite fewer master calls. The unchanged GP cannot be credited as the cause of this query reduction.\n",
        f"保留分式使总评估次数从 {b['completed_evaluations']['total']:.0f} 降到 {a['completed_evaluations']['total']:.0f}（减少 {-qt:.1f}\\%），评估次数减少/相同/增加的实例分别为 {fewer}/{equal}/{more}。总中位时间为 {a['seconds']['total']:.3f} 秒，原搜索为 {b['seconds']['total']:.3f} 秒（{rt:+.1f}\\%）；在 {faster}/{len(pairs)} 个实例上更快。因此，减少评估次数不能证明整体提速。分式模型的每次主问题更昂贵，尽管调用更少，主问题总时间仍增加。未改变的 GP 不能被视为这次评估减少的原因。\n")
    text += r"\ResearchFigure{search_confirmation" + ("_zh" if language == "zh" else "") + "}{" + tr(language,
        "Paired confirmation cases. Upper panel: full binary evaluations (smaller is better). Lower panel: stored-fraction time divided by original GP time, using the three-run median for each method. The dashed line at one means equal time. Cases 1--8 use half hinge and 9--16 logistic; each loss block is ordered by dimension 7/10, seed 19/23, then radius pair.",
        "配对确认实例。上图：完整二元评估次数，越小越好。下图：保留分式的时间除以原 GP 时间，每种方法均使用三次运行的中位数；虚线一表示时间相同。实例 1--8 为半合页，9--16 为逻辑损失；每个损失组内依次按维度 7/10、种子 19/23、半径对排列。") + r"}{fig:search-confirmation}" + "\n"
    maximum = max(result["upper"] - record["reference"]["lower"] for record in stage["raw"].values()
                  for repeated in record["repeats"] for result in repeated["methods"].values())
    all_runs = sum(v["cases"] for v in stage["run_totals"].values())
    verified = sum(v["statuses"].get("optimal", 0) for v in stage["run_totals"].values())
    text += tr(language,
        f"All {verified}/{all_runs} confirmation runs are numerically optimal and pass binary-only, owned-certificate and actual-incumbent checks. The largest feasible-risk excess over the enumeration lower reference is {fmt(maximum, scientific=True)}. These checks validate the empirical optimization target; they do not assert population robustness or universal runtime improvement.\n",
        f"全部 {verified}/{all_runs} 次确认运行均数值最优，并通过仅二元评估、证书归属与真实检验更新上界的核查。相对枚举下界参照的最大可行风险超出为 {fmt(maximum, scientific=True)}。这些核查验证经验优化目标，不断言总体鲁棒性或普遍运行时间改进。\n")
    return text


def render(evidence, template, *, language="en", validation=None):
    blocks = {"OVERVIEW": concise_overview(evidence, language, validation),
              "SETUP_TABLE": setup_report(evidence, language), "SEARCH_TABLE": search_report(validation, language),
              "COMPUTATIONAL_TABLE": reference_computation(evidence, language),
              "STATISTICAL_TABLE": statistical_report(evidence, language), "REAL_TABLE": real_report(evidence, language),
              "SENSITIVITY_TABLE": sensitivity_report(evidence, language), "RATIO_TABLE": reference_ratio(evidence, language),
              "REMEDIATION": concise_records(evidence, language)}
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
    if not args.draft and args.validation_root is None:
        raise ValueError("Final bilingual report requires separately frozen revised-search confirmation evidence; reference-only builds must use --draft")
    validation = SearchEvidence(args.validation_root.resolve(), args.draft) if args.validation_root else None
    source = render(evidence, args.template.read_text(encoding="utf-8"), language=args.language, validation=validation)
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
                    "evidence_cohorts_separated": True, "inputs": input_hashes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"source": str(args.output), **{k: v for k, v in manifest.items() if k != "inputs"}}))



if __name__ == "__main__":
    main()
