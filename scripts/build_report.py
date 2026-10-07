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
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

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
MARKERS = ("OVERVIEW", "COMPUTATIONAL_TABLE", "STATISTICAL_TABLE", "REAL_TABLE",
           "SENSITIVITY_TABLE", "RATIO_TABLE", "REMEDIATION", "APPENDIX_TABLES")
SHORT_STATUS = {"optimal": "O", "time_limit": "T", "master_failure": "M",
                "oracle_failure": "F", "oracle_accuracy_limit": "A",
                "master_accuracy_limit": "A", "iteration_limit": "I"}


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


def longtable(caption, headers, rows, *, first_width="31mm", label=None):
    if not rows:
        return r"\emph{No recorded outcomes are available for this block.}" + "\n"
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Table row length differs from its header")
    columns = "@{}p{" + first_width + "}" + "r" * (len(headers) - 1) + "@{}"
    header = " & ".join(headers) + r" \\"
    body = "\n".join(" & ".join(map(str, row)) + r" \\" for row in rows)
    if label in {"tab:gate-main", "tab:head-main", "tab:real-main"}:
        return (r"\begin{table}[tbp]\centering\small\setlength{\tabcolsep}{4pt}" + "\n"
                + r"\caption{" + caption + r"}\label{" + label + "}\n"
                + r"\begin{tabular}{" + columns + "}\n"
                + r"\toprule " + header + "\n" + r"\midrule" + "\n" + body + "\n"
                + r"\bottomrule\end{tabular}\end{table}" + "\n")
    return (r"\begingroup\small\setlength{\tabcolsep}{4pt}" + "\n"
            + r"\begin{longtable}{" + columns + "}\n"
            + r"\caption{" + caption + "}" + (r"\label{" + label + "}" if label else "") + r"\\" + "\n"
            + r"\toprule " + header + "\n" + r"\midrule\endfirsthead" + "\n"
            + r"\multicolumn{" + str(len(headers)) + r"}{l}{\emph{Continued}}\\" + "\n"
            + r"\toprule " + header + "\n" + r"\midrule\endhead" + "\n"
            + r"\midrule\multicolumn{" + str(len(headers))
            + r"}{r}{\emph{Continued on next page}}\\\endfoot" + "\n"
            + r"\bottomrule\endlastfoot" + "\n" + body + "\n"
            + r"\end{longtable}\endgroup" + "\n")


def statuses(row, *, abbreviated=False):
    values = row.get("status_counts", {})
    if isinstance(values, str):
        values = json.loads(values)
    if abbreviated:
        counts = Counter()
        for name, count in values.items():
            counts[SHORT_STATUS.get(name, name)] += count
        return escaped(", ".join(f"{v}{k}" for k, v in sorted(counts.items())))
    return escaped(", ".join(f"{count} {name.replace('_', ' ')}" for name, count in sorted(values.items())))


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
                    settings.append(dict(mechanism=dataset, dimension=dimension,
                                         budget=config["budget"], seeds=config["seeds"],
                                         gates=methods, theta=theta))
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


def run_metadata(e):
    rows = []
    for suite in SUITES:
        actual = [r for r in e.rows if r["suite"] == suite]
        expected = sum(k[0] == suite for k in e.expected)
        rows.append([escaped(suite.replace("_", " ")), expected, len(actual),
                     sum(r["run_status"] == "complete" for r in actual)])
    prefix = ""
    if e.draft:
        prefix = (r"\begin{center}\fbox{\parbox{.92\linewidth}{\textbf{DRAFT SNAPSHOT.} "
                  + f"{len(e.rows)} of {len(e.expected)} planned records are present; "
                  + f"{len(e.missing)} are missing. This is not the completed study."
                  + r"}}\end{center}" + "\n")
    prefix += longtable("Recorded outcome inventory. Complete denotes completed evaluation, not certified optimal selection.",
                        ["Suite", "Planned", "Present", "Evaluated"], rows, first_width="47mm")
    tested = e.root / "verification" / "tests.txt"
    match = re.search(r"(\d+) passed(?: in ([\d.]+)s)?", tested.read_text(encoding="utf-8") if tested.exists() else "")
    checktext = f"The original study verification reports {match[1]} passed tests" if match else "The original study has no recorded verification count"
    if match and match[2]:
        checktext += f" in {match[2]} seconds"
    cpu = e.root / "hardware_cpu.json"
    hardware = json.loads(cpu.read_text(encoding="utf-8")) if cpu.exists() else {}
    model = str(hardware.get("Name", "hardware recorded in the run manifest")).strip()
    total = sum(float(r.get("total_seconds", 0)) for r in e.raw.values())
    selection = sum(float(r.get("selection", {}).get("seconds", 0)) for r in e.raw.values())
    prefix += (f"{escaped(checktext)}. The host is {escaped(model)}. "
               f"Summed charged selection time is {selection / 3600:.2f} hours; "
               f"summed end-to-end trial time, including evaluation, is {total / 3600:.2f} hours. "
               "These sums include unsuccessful outcomes and recorded remediation costs; they are not elapsed calendar time for concurrent runners.\n\n")
    unknown_selection = sum(number(r.get("selection", {}), "seconds") is None for r in e.raw.values())
    if unknown_selection:
        prefix += (f"Selection-only timing is unavailable for {unknown_selection} records; they remain in status denominators and contribute their recorded end-to-end time. "
                   "Missing selection times are not imputed as zero in a method mean.\n\n")
    environment_path = e.root / "environment.json"
    if environment_path.exists():
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        packages = environment.get("packages", {})
        engine = environment.get("linear_integer_backend", {}).get("highs_version")
        if engine:
            prefix += (f"The LP and MILP engine is SciPy-backed HiGHS {escaped(engine)} "
                       f"(SciPy {escaped(packages.get('scipy', 'version not recorded'))}). ")
        else:
            prefix += "The LP/MILP backend is SciPy; its bundled HiGHS version was not recorded in this snapshot. "
        prefix += (f"Logistic correctness checks use CVXPY {escaped(packages.get('cvxpy', 'version not recorded'))} "
                   f"with Clarabel {escaped(packages.get('clarabel', 'version not recorded'))}. "
                   "\n\n")
    git = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=e.project,
                         capture_output=True, text=True, check=False)
    version = git.stdout.strip() if git.returncode == 0 else "uncommitted workspace"
    prefix += "Build revision: " + escaped(version) + ". Input and configuration SHA-256 hashes are saved with the generated report manifest.\n\n"
    n, difference, violations = computational_check(e)
    prefix += (f"The completed small-instance comparisons contain {n} paired, numerically solved gate/reference outcomes. "
               f"Their largest absolute feasible-objective difference is {fmt(difference, scientific=True)}; "
               f"{violations} recorded gate intervals fail the paired enumeration compatibility check at the combined declared tolerances. "
               "Agreement validates the shared integer target, rather than a statistical performance gain.\n")
    return prefix


def full_computational_tables(e, suite="computational"):
    out = []
    method_order = GATES if suite == "computational" else ("risk", "risk_ratio")
    for dataset in ("mean", "mixture"):
        groups = e.select(suite, dataset)
        dimensions = sorted({int(r["dimension"]) for r in groups})
        rows = []
        for dimension in dimensions:
            for method in method_order:
                r = e.lookup(suite, dataset, method, dimension)
                if r is None:
                    continue
                solved = round(r.get("solved_fraction", 0) * r["n_total"])
                rows.append([f"${dimension},{r['budget']}$", LABELS[method], f"{solved}/{r['n_total']}",
                             estimate(r, "selection_seconds", digits=2),
                             fmt(number(r, "selection_gap_mean"), scientific=True),
                             fmt(number(r, "selection_visited_mean"), 1), statuses(r, abbreviated=True)])
        caption = (f"{TITLES[dataset]}: " + ("primary gate comparison" if suite == "computational" else "paired exploratory master comparison")
                   + ". Time is charged mean seconds [95\\% seed-bootstrap interval], with all outcomes retained. "
                   "Gap is mean absolute upper-minus-lower bound; visits count distinct integer supports. "
                   "O: numerical optimum; T: time limit; M: master failure; F: oracle failure; A: accuracy limit.")
        out.append(longtable(caption, ["$(D,d)$", "Method", "Solved / $n$", "Time (s)",
                                      "Gap", "Visits", "Status"], rows, first_width="12mm"))
        if suite == "ratio":
            paired_rows = []
            for dimension in dimensions:
                n, interval = paired(e.rows, suite, dataset, "selection_seconds", a="risk_ratio", b="risk", dimension=dimension)
                paired_rows.append([f"$D={dimension}$", n, interval_cell(interval, 2)])
            out.append(longtable("Within-seed ratio-minus-tangent charged time. Negative differences favor the supplement; these are descriptive paired intervals, not significance tests.",
                                 ["Dimension", "Paired $n$", "$\\Delta$ time (s) [95\\% CI]"], paired_rows))
    if suite == "computational":
        n, difference, violations = computational_check(e)
        out.append(f"On {n} solved gate/enumeration pairs, the maximum absolute upper-objective difference is "
                   f"{fmt(difference, scientific=True)}. Interval incompatibilities at combined numerical tolerances: {violations}. "
                   "Enumeration references are paired by mechanism, dimension, budget, radius, and seed. "
                   "A comparison with no completed reference is not counted as exact objective agreement.\n")
    else:
        out.append("These observations use seeds 10--14 in the separately frozen supplement. "
                   "The tangent comparator is from this suite itself; primary-suite timings are not substituted. "
                   "The component-time tables in the appendix identify whether a difference comes from master or oracle work.\n")
    return "\n".join(out)


def solved_count(row):
    values = row.get("status_counts", {})
    if isinstance(values, str):
        values = json.loads(values)
    return int(values.get("optimal", 0))


def computational_readout(e, suite, dataset):
    if suite == "computational":
        comparison = [e.lookup(suite, dataset, method, dimension=12)
                      for method in ("risk", "triangle", "box")]
        pieces = []
        if all(comparison):
            descriptions = []
            for method, row in zip(("risk", "triangle", "box"), comparison):
                seconds = fmt(number(row, "selection_seconds_mean"), 2)
                descriptions.append(f"{LABELS[method]} solves {solved_count(row)}/{row['n_total']} "
                                    f"recorded instances, with mean charged time {seconds} s")
            pieces.append("At $D=12$, " + "; ".join(descriptions) + ".")
        scaling = [r for r in e.rows if r["suite"] == suite and r["dataset"] == dataset
                   and r["method"] == "risk" and int(r["dimension"]) >= 20]
        if scaling:
            timed_out = sum(r["selection_status"] == "time_limit" for r in scaling)
            pieces.append(f"Across the recorded $D\\ge20$ risk-method instances, {timed_out}/{len(scaling)} "
                          f"end at the time limit ({100 * timed_out / len(scaling):.1f}\\%). "
                          "These time and solved-count comparisons retain unsuccessful outcomes; "
                          "a stronger continuous gate has not removed the integer-search cost.")
        if not pieces:
            return ""
        return (r"\paragraph{" + escaped(TITLES[dataset]) + " --- computational readout.} "
                + " ".join(pieces) + "\n")
    pieces = []
    dimensions = sorted({int(r["dimension"]) for r in e.select(suite, dataset)})
    for dimension in dimensions:
        tangent = e.lookup(suite, dataset, "risk", dimension)
        ratio = e.lookup(suite, dataset, "risk_ratio", dimension)
        if not tangent or not ratio:
            continue
        n, (difference, _, _) = paired(e.rows, suite, dataset, "selection_seconds",
                                       a="risk_ratio", b="risk", dimension=dimension)
        if not n:
            continue
        pieces.append(f"At $D={dimension}$, the observed mean ratio-minus-tangent charged time "
                      f"is {difference:+.2f} s over {n} paired seeds; complete ratios solve "
                      f"{solved_count(ratio)}/{ratio['n_total']} recorded instances and tangents "
                      f"{solved_count(tangent)}/{tangent['n_total']}.")
    if not pieces:
        return ""
    return (r"\paragraph{" + escaped(TITLES[dataset]) + " --- supplementary readout.} "
            + " ".join(pieces)
            + " Negative time differences favor the supplement on these paired instances. "
            + "The table gives seed-level uncertainty and achieved gaps; no universal speed or superiority claim follows.\n")


def mechanism_readout(e, dataset):
    if any(k[0] == "statistical" and k[1] == dataset for k in e.missing):
        return ""
    global_row = e.lookup("statistical", dataset, "risk")
    screen_row = e.lookup("statistical", dataset, "mmd")
    if not global_row or not screen_row:
        return ""
    gp, sp = number(global_row, "power_mean"), number(screen_row, "power_mean")
    gr, sr = number(global_row, "recall_mean"), number(screen_row, "recall_mean")
    if any(v is None for v in (gp, sp, gr, sr)):
        return ""
    n, (difference, lo, hi) = paired(e.rows, "statistical", dataset, "power")
    if not n:
        return ""
    text = (f"For {escaped(dataset)} data, global selection has common-test power {gp:.3f} "
            f"versus {sp:.3f} for marginal screening. The paired difference is "
            f"{100 * difference:+.1f} percentage points (95\\% CI "
            f"[{100 * lo:+.1f}, {100 * hi:+.1f}]; $n={n}$). "
            f"Signal recall is {gr:.3f} versus {sr:.3f}. ")
    greedy = e.lookup("statistical", dataset, "greedy")
    greedy_power = number(greedy or {}, "power_mean")
    if greedy_power is not None and dataset == "mean":
        text += (f"All {solved_count(global_row)}/{global_row['n_total']} global runs meet the optimization tolerance, "
                 f"but greedy power is {greedy_power:.3f}. ")
    wang = e.lookup("statistical", dataset, "wang")
    native_power = number(wang or {}, "native_additive_power_mean")
    native_null = number(wang or {}, "native_additive_null_rejection_mean")
    common_power = number(wang or {}, "power_mean")
    if (dataset in ("variance", "mixture")
            and all(v is not None for v in (native_power, native_null, common_power))):
        text += (f"Wang's native signed additive statistic has power {native_power:.3f} "
                 f"and null rejection {native_null:.3f}, compared with common support-only power "
                 f"{common_power:.3f} ($n={wang.get('native_additive_power_n', 0)}$). ")
    return text + "\n\n"


def training_bound_readout(e):
    """Compare each separately fitted head's empirical bound with independent BER."""
    table_rows = []
    total_pairs, total_above, source_values = 0, 0, []
    for dataset in MECHANISMS:
        records = [r for r in e.rows if r["suite"] == "statistical"
                   and r["dataset"] == dataset and r["method"] == "risk"]
        pairs = [(number(r, "head_robust_upper"), number(r, "robust_randomized_balanced_error"),
                  number(r, "empirical_certificate_violation")) for r in records]
        pairs = [(upper / 2, error, violation) for upper, error, violation in pairs
                 if upper is not None and error is not None]
        if not pairs:
            continue
        above = sum(error > bound + 1e-8 for bound, error, _ in pairs)
        violations = [violation for _, _, violation in pairs if violation is not None]
        table_rows.append([escaped(TITLES[dataset]), len(pairs),
                           fmt(float(np.mean([p[0] for p in pairs]))),
                           fmt(float(np.mean([p[1] for p in pairs]))),
                           f"{above}/{len(pairs)}", fmt(max(violations, default=None), scientific=True)])
        total_pairs += len(pairs)
        total_above += above
        source_values += violations
    if not table_rows:
        return ""
    text = (f"For the separately fitted robust classifier, {total_above}/{total_pairs} independent "
            "randomized errors exceed their own empirical head bound divided by two "
            "(Table~\\ref{tab:head-main}). "
            f"The corresponding {len(source_values)} valid source-atom checks have "
            f"{sum(v > 1e-7 for v in source_values)} violations above $10^{{-7}}$. "
            "This empirical-ball certificate is distinct from the selection objective. "
            "The evaluated population was not established to lie in that ball; the held-out excess "
            "is therefore not a certificate failure or a population-coverage result.\n\n")
    text += longtable("Refitted empirical head risk and independent randomized error for global selection. Means use paired training seeds. Above-bound counts compare each seed at tolerance $10^{-8}$; the final column is the maximum budget-feasible source-atom violation.",
                      ["Mechanism", "$n$", "Head risk / 2", "Test error", "Above bound", "Source violation"],
                      table_rows, first_width="35mm", label="tab:head-main")
    return text


def full_statistical_tables(e):
    out = []
    metrics = ["power", "null_rejection", "recall"]
    for dataset in MECHANISMS:
        rows = []
        for method in PRIMARY:
            r = e.lookup("statistical", dataset, method)
            if r:
                rows.append([LABELS[method], r["n_total"], *[estimate(r, m) for m in metrics]])
        out.append(longtable(TITLES[dataset] + ": common-support testing and recovery. Means [95\\% training-seed intervals]. All methods use budget three; the null level is 0.05.",
                             ["Selector", "$n$", "Power", "Null rejection", "Recall"], rows))
        classrows = []
        for method in PRIMARY:
            r = e.lookup("statistical", dataset, method)
            if r:
                classrows.append([LABELS[method], r["n_total"],
                                  *[estimate(r, m) for m in ["robust_randomized_balanced_error", "robust_balanced_error", "robust_auc", "svm_balanced_error"]]])
        out.append(longtable(TITLES[dataset] + ": independently refitted common heads. Randomized and thresholded balanced error are distinct; AUC evaluates ranking. Values are means [95\\% seed intervals].",
                             ["Selector", "$n$", "Random BER $\\downarrow$", "BER $\\downarrow$", "AUC $\\uparrow$", "SVM BER $\\downarrow$"], classrows))
    differences = []
    for dataset in MECHANISMS:
        n, triple = paired(e.rows, "statistical", dataset, "power")
        differences.append([escaped(TITLES[dataset]), n, interval_cell(triple)])
    out.append(longtable("Paired global-risk-minus-marginal-screening power. Negative values mean lower observed power for global robust-risk selection; the interval resamples training-seed differences.",
                         ["Mechanism", "Paired $n$", "$\\Delta$ power [95\\% CI]"], differences, first_width="67mm"))
    out.append("All six selectors above buy the same number of coordinates. Known-signal and all-feature diagnostics are reserved for the appendix. "
               "Within-seed two-sample batches were averaged before uncertainty was computed; conditional batches are not counted as independent selections. "
               "Valid predictive metrics from a time-limited or failed global search describe its retained support and do not certify optimal selection.\n")
    return "\n".join(out)


def full_real_tables(e):
    out = []
    for dataset in ("wdbc", "ionosphere"):
        rows = []
        for method in PRIMARY:
            r = e.lookup("real", dataset, method)
            if r:
                rows.append([LABELS[method], r["n_complete"],
                             *[estimate(r, m, real=True) for m in ["robust_randomized_balanced_error", "robust_balanced_error", "robust_auc", "svm_balanced_error", "robust_worst_stress_balanced_error"]]])
        out.append(longtable(TITLES[dataset] + ": mean $\\pm$ split SD, not an independent-population confidence interval. Worst stress is the maximum over the three prescribed shifts. Effective metric counts and selection statuses are retained in the appendix.",
                             ["Selector", "$n$", "Random BER $\\downarrow$", "BER $\\downarrow$", "AUC $\\uparrow$", "SVM BER $\\downarrow$", "Stress BER $\\downarrow$"], rows))
        selrows = []
        for method in PRIMARY:
            r = e.lookup("real", dataset, method)
            if r:
                selrows.append([LABELS[method], r["n_total"], estimate(r, "selection_seconds", real=True, digits=2),
                                fmt(number(r, "selection_lower_mean")), fmt(number(r, "selection_upper_mean")),
                                statuses(r, abbreviated=method == "risk")])
        # Non-robust baseline status strings are explained by the method definition;
        # keep long status prose outside the numeric columns.
        for row in selrows:
            if row[0] != LABELS["risk"]:
                row[-1] = "baseline"
        out.append(longtable(TITLES[dataset] + ": charged selection time and original-risk bounds. A dash means no global original-objective certificate, not zero risk. Risk statuses use O/T/M/F/A as in the computational tables.",
                             ["Selector", "$n$", "Time (s) $\\downarrow$", "Mean LB", "Mean UB", "Status"], selrows))
    return "\n".join(out)


def full_sensitivity_tables(e):
    out = []
    for dataset in ("mean", "mixture"):
        rows = []
        for theta in e.configs["sensitivity"]["radii"]:
            r = e.lookup("sensitivity", dataset, "risk", theta=theta)
            if r:
                rows.append([fmt(theta, 2), r["n_total"], estimate(r, "power"),
                             estimate(r, "robust_randomized_balanced_error"),
                             estimate(r, "robust_worst_stress_balanced_error"),
                             statuses(r, abbreviated=True)])
        out.append(longtable(TITLES[dataset] + ": radius sensitivity, means [95\\% training-seed intervals]. The primary radius remains 0.25. Supports are refitted at each radius; no result-selected setting is declared optimal.",
                             ["Radius", "$n$", "Power $\\uparrow$", "Random BER $\\downarrow$", "Stress BER $\\downarrow$", "Status"], rows, first_width="19mm"))
    return "\n".join(out)


def sensitivity_readout(e, dataset):
    radii = sorted(e.configs["sensitivity"]["radii"])
    rows = [e.lookup("sensitivity", dataset, "risk", theta=radius) for radius in radii]
    if not rows or any(r is None for r in rows):
        return ""
    metrics = ("power", "robust_randomized_balanced_error", "robust_worst_stress_balanced_error")
    endpoints = [(number(rows[0], m + "_mean"), number(rows[-1], m + "_mean")) for m in metrics]
    if any(v is None for pair in endpoints for v in pair):
        return ""
    power = [number(row, "power_mean") for row in rows]
    trend = ""
    if all(v is not None for v in power):
        changes = np.diff(power)
        trend = ("The observed power averages increase across these settings. " if np.all(changes >= 0)
                 else "The observed power averages decrease across these settings. " if np.all(changes <= 0)
                 else "The observed power averages are nonmonotone across these settings. ")
    return (f"For {escaped(dataset)} data, from radius {radii[0]:.2f} to {radii[-1]:.2f}, power changes from "
            + f"{endpoints[0][0]:.3f} to {endpoints[0][1]:.3f}, randomized balanced error from "
            + f"{endpoints[1][0]:.3f} to {endpoints[1][1]:.3f}, and worst measured thresholded stress error from "
            + f"{endpoints[2][0]:.3f} to {endpoints[2][1]:.3f}. " + trend
            + "The support and robust classifier are both refitted, so these changes describe their joint sensitivity.\n")


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


def remediation(e):
    archived = archived_records(e)
    failed_statuses = {"master_failure", "oracle_failure"}
    current_failed = {k for k, r in e.raw.items() if r.get("selection", {}).get("status", "failure") in failed_statuses}
    rerun = [k for k in archived if k in e.raw]
    retries = sum(int(r.get("selection", {}).get("diagnostics", {}).get("master_retries", 0)) for r in e.raw.values())
    outcomes = Counter(e.raw[k].get("selection", {}).get("status", "failure") for k in rerun)
    outcome_labels = {
        "optimal": ("numerical optimum", "numerical optima"),
        "time_limit": ("termination at the time limit", "terminations at the time limit"),
        "master_failure": ("master failure", "master failures"),
        "oracle_failure": ("oracle failure", "oracle failures"),
    }
    outcome_items = [f"{count} {outcome_labels.get(name, (name, name))[count != 1]}"
                     for name, count in sorted(outcomes.items())]
    outcome_text = (", ".join(outcome_items[:-1]) + ", and " + outcome_items[-1]
                    if len(outcome_items) > 2 else " and ".join(outcome_items))
    extra = current_failed - set(archived)
    out = (r"\subsection{Numerical failures and replay}" + "\n"
           + f"The frozen original failure inventory contains {len(archived)} runs. "
           + f"The {len(rerun)} replays ended in "
           + escaped(outcome_text) + ". "
           + f"The final collection has {len(current_failed)} failures, including {len(extra)} "
           + "outside the original replay cohort. "
           + f"Final diagnostics record {retries} identical-model master retries.\n\n")
    original_selection = sum(float(r["selection"].get("seconds", 0)) for r in archived.values())
    original_total = sum(float(r.get("total_seconds", 0)) for r in archived.values())
    if archived:
        out += (f"Archived failed attempts charged {original_selection:.2f} selection seconds and {original_total:.2f} end-to-end seconds. "
                "The final record timings include those original costs in addition to rerun costs. The rerun receives only the original 30-second selection allowance minus the original charged selection time; both attempts share that allowance, subject to modest call/finalization overhead. "
                "The appendix lists the retained original and final statuses rather than suppressing the first outcomes.\n")
    else:
        out += "No archived rerun provenance is present in this snapshot. The current failed statuses remain in every reported denominator.\n"
    out += ("Only status-4 solver errors trigger one same-model retry with presolve disabled. "
            "An error return contributes no optimization bound. Original and replayed oracle/master "
            "costs are cumulative; replay receives only the unused original selection allowance. "
            "The independent numerical probe is not a performance observation.\n")
    failure_rows = []
    for trial, old in sorted(archived.items()):
        current = e.raw.get(trial)
        failure_rows.append([rf"\shortstack[l]{{{escaped(trial[0])}, {escaped(trial[1])}\\"
                             + rf"$D={trial[2]}$, seed {trial[-1]}; {LABELS[trial[-2]]}}}",
                             escaped(old.get("selection", {}).get("status", "failure").replace("_", " ")),
                             escaped(current.get("selection", {}).get("status", "failure").replace("_", " ")) if current else "---",
                             fmt(number(old.get("selection", {}), "seconds"), 2),
                             fmt(number(current.get("selection", {}), "seconds"), 2) if current else "---"])
    out += longtable("Original replay cohort and final outcomes. Charged time includes the first attempt and replay.",
                     ["Trial", "Original status", "Final status", "Original (s)", "Charged (s)"], failure_rows,
                     first_width="43mm")
    return out


def supplementary_tables(e):
    out = [r"\subsection{Component time and effort}" + "\n"]
    for suite in ("computational", "ratio"):
        for dataset in ("mean", "mixture"):
            groups = e.select(suite, dataset)
            order = GATES if suite == "computational" else ("risk", "risk_ratio")
            rows = []
            for dimension in sorted({r["dimension"] for r in groups}):
                for method in order:
                    r = e.lookup(suite, dataset, method, dimension)
                    if not r:
                        continue
                    rows.append([f"${dimension}$", LABELS[method], r["n_total"],
                                 fmt(number(r, "oracle_seconds_mean"), 2), fmt(number(r, "master_seconds_mean"), 2),
                                 fmt(number(r, "selection_oracle_calls_mean"), 1), fmt(number(r, "root_bound_mean")),
                                 fmt(number(r, "selection_upper_mean"))])
            out.append(longtable(escaped(suite.replace("_", " ")) + ", " + TITLES[dataset]
                                 + ": all-outcome mean component times, calls, root bounds, and feasible original risk. These means include unsuccessful outcomes; an absent component is a dash.",
                                 ["$D$", "Method", "$n$", "Oracle (s)", "Master (s)", "Calls", "Root LB", "Risk UB"], rows, first_width="12mm"))
    out.append(r"\subsection{Diagnostic supports and probability loss}" + "\n")
    diagnostic_rows = []
    for dataset in MECHANISMS:
        for method in ("signal", "all"):
            r = e.lookup("statistical", dataset, method)
            if r:
                diagnostic_rows.append([rf"\shortstack[l]{{{escaped(TITLES[dataset])}\\{LABELS[method]}}}", r["n_total"],
                                        estimate(r, "power"), estimate(r, "null_rejection"),
                                        estimate(r, "robust_randomized_balanced_error")])
    out.append(longtable("Reference supports are diagnostic, not six-method budget-matched competitors. Known signal uses ground truth; all features exceed the acquisition budget. Values are means [95\\% training-seed intervals].",
                         ["Diagnostic reference", "$n$", "Power $\\uparrow$", "Null rejection", "Random BER $\\downarrow$"], diagnostic_rows, first_width="57mm"))
    for suite in ("statistical", "real"):
        datasets = MECHANISMS if suite == "statistical" else ("wdbc", "ionosphere")
        rows = []
        for dataset in datasets:
            for method in PRIMARY:
                r = e.lookup(suite, dataset, method)
                if r:
                    rows.append([rf"\shortstack[l]{{{escaped(dataset)}\\{LABELS[method]}}}",
                                 int(r.get("robust_predictive_log_loss_average_n", 0)),
                                 estimate(r, "robust_predictive_log_loss_average", real=suite == "real"),
                                 fmt(number(r, "stability_jaccard")),
                                 estimate(r, "selection_seconds", real=suite == "real", digits=2)])
        out.append(longtable(escaped(suite) + ": predictive cross-entropy of the half-hinge-trained common head, semantic support stability, and charged selection time. Synthetic intervals are seed-bootstrap intervals; real entries use split SD. Stability is pairwise Jaccard, not a confidence interval.",
                             ["Dataset / selector", "Valid $n$", "Log loss $\\downarrow$", "Stability $\\uparrow$", "Time (s) $\\downarrow$"], rows, first_width="46mm"))
    native = []
    for dataset in MECHANISMS:
        r = e.lookup("statistical", dataset, "wang")
        if r:
            native.append([escaped(TITLES[dataset]), int(r.get("native_additive_power_n", 0)),
                           estimate(r, "native_additive_power"), estimate(r, "native_additive_null_rejection")])
    out.append(longtable("Wang paper-objective weights with the independently frozen native additive statistic. This diagnostic evaluates the learned weighted statistic, rather than the common joint-RBF support test. Means [95\\% seed intervals].",
                         ["Mechanism", "Valid $n$", "Native power $\\uparrow$", "Native null rejection"], native, first_width="67mm"))
    out.append(outcome_inventory(e))
    violations = [number(r, "empirical_certificate_violation") for r in e.rows]
    violations = [v for v in violations if v is not None]
    out.append(f"There are {len(violations)} explicit shifted-source certificate checks. "
               f"The largest recorded nonnegative surrogate violation is {fmt(max(violations, default=None), scientific=True)}; "
               f"{sum(v > 1e-7 for v in violations)} exceed $10^{{-7}}$. "
               "This directly budget-feasible source-atom calculation is separate from population stress.\n")
    return "\n".join(out)


def outcome_inventory(e):
    groups = defaultdict(list)
    for row in e.rows:
        groups[(row["suite"], row["method"])].append(row)
    outcomes, denominators = [], []
    failures = {"master_failure", "oracle_failure", "numerical_failure", "numerical_inconsistency", "failure"}
    for (suite, method), group in sorted(groups.items()):
        counts = Counter(row["selection_status"] for row in group)
        optimal, limited = counts["optimal"], counts["time_limit"]
        failed = sum(counts[s] for s in failures)
        other = len(group) - optimal - limited - failed
        label = rf"\shortstack[l]{{{escaped(suite)}\\{LABELS[method]}}}"
        outcomes.append([label, len(group), sum(r["run_status"] == "complete" for r in group),
                         optimal, limited, failed, other])
        if suite in ("statistical", "real", "sensitivity"):
            denominators.append([label, len(group),
                                 sum(number(r, "power") is not None for r in group) if suite != "real" else "n/a",
                                 sum(number(r, "robust_balanced_error") is not None for r in group),
                                 sum(number(r, "empirical_certificate_violation") is not None for r in group)])
    return (r"\subsection{Outcome inventory and effective counts}" + "\n"
            + longtable("Final outcomes aggregated by suite and method. O denotes a numerical optimum for the original robust criterion; T a selection limit; F a numerical failure. Other includes baseline completion and any remaining status. Complete counts refer to evaluation.",
                        ["Suite / method", "Total", "Complete", "O", "T", "F", "Other"], outcomes,
                        first_width="39mm")
            + longtable("Effective counts aggregated for availability only. Test replicates are not extra trained selectors, and confidence intervals are computed within settings. Real-data testing power is undefined.",
                        ["Suite / method", "Total", "Test $n$", "Classifier $n$", "Source check $n$"], denominators,
                        first_width="39mm"))


def overview(e):
    text = ""
    if e.draft:
        text = (rf"\textbf{{Draft: {len(e.rows)}/{len(e.expected)} records available; "
                rf"{len(e.missing)} missing.}} " )
    text += ("We evaluate global feature selection for a two-class Wasserstein robust testing "
             "criterion using four synthetic alternatives and two public datasets. "
             "Independent selection, classifier-fitting and test samples separate optimization "
             "accuracy from statistical utility. ")
    n, difference, violations = computational_check(e)
    if n:
        delta = "zero" if difference == 0 else fmt(difference, scientific=True)
        text += (f"Across {n} solved small gate/enumeration pairs, the largest recorded "
                 f"objective difference is {delta}, with {violations} incompatible bound intervals. ")
    global_mean = e.lookup("statistical", "mean", "risk")
    screening_mean = e.lookup("statistical", "mean", "mmd")
    if global_mean and screening_mean:
        text += (f"On mean-shift data, {solved_count(global_mean)}/{global_mean['n_total']} "
                 f"global runs meet the optimization tolerance, but common-test power is "
                 f"{fmt(number(global_mean, 'power_mean'))} versus "
                 f"{fmt(number(screening_mean, 'power_mean'))} for marginal screening. ")
    text += ("The risk-scaled gate yields stronger lower models without consistently reducing "
             "runtime. Empirical classifier certificates do not establish population coverage, "
             "and an exploratory full-fraction master does not improve elapsed time on its "
             "paired instances. The records support the finite optimization construction while "
             "exposing limitations of computational efficiency and small-sample statistical selection.")
    return text


def computational_tables(e, suite="computational"):
    if suite == "ratio":
        return ("\n\n".join(computational_readout(e, suite, dataset) for dataset in ("mean", "mixture"))
                + "\n" + full_computational_tables(e, suite))
    rows = []
    for method in GATES:
        mean = e.lookup(suite, "mean", method, dimension=12)
        mixture = e.lookup(suite, "mixture", method, dimension=12)
        if mean is None or mixture is None:
            continue
        rows.append([LABELS[method], f"{solved_count(mean)}/{mean['n_total']}",
                     estimate(mean, "selection_seconds", digits=2),
                     f"{solved_count(mixture)}/{mixture['n_total']}",
                     estimate(mixture, "selection_seconds", digits=2)])
    text = longtable("Gate comparison at $D=12,d=3$. Charged time is mean seconds [95\\% seed interval]; solved counts retain all ten outcomes. Full dimensions, gaps and component times are in Appendix~\\ref{app:results}.",
                     ["Method", "Mean solved", "Mean time (s)", "Mixture solved", "Mixture time (s)"],
                     rows, first_width="28mm", label="tab:gate-main")
    n, difference, incompatible = computational_check(e)
    if n:
        delta = "zero" if difference == 0 else fmt(difference, scientific=True)
        text += (f"The {n} solved gate/reference pairs have maximum absolute objective "
                 f"difference {delta}; {incompatible} recorded intervals are incompatible with "
                 "enumeration at the combined numerical tolerances.\n\n")
    for dataset in ("mean", "mixture"):
        risk = e.lookup(suite, dataset, "risk", dimension=12)
        box = e.lookup(suite, dataset, "box", dimension=12)
        if risk and box:
            text += (f"At $D=12$ for {escaped(dataset)} data, the risk method visits "
                     f"{fmt(number(risk, 'selection_visited_mean'), 1)} supports on average versus "
                     f"{fmt(number(box, 'selection_visited_mean'), 1)} for box, but spends "
                     f"{fmt(number(risk, 'master_seconds_mean'), 2)} s in the master versus "
                     f"{fmt(number(box, 'master_seconds_mean'), 2)} s. ")
    scaling = [r for r in e.rows if r["suite"] == suite and int(r["dimension"]) >= 20]
    if scaling:
        limits = sum(r["selection_status"] == "time_limit" for r in scaling)
        text += (f"At $D\\ge20$, {limits}/{len(scaling)} gate runs end at the selection limit. "
                 "The achieved gaps remain informative even though no exact solution is recorded. ")
        for dataset in ("mean", "mixture"):
            risk = e.lookup(suite, dataset, "risk", dimension=20)
            box = e.lookup(suite, dataset, "box", dimension=20)
            if risk and box:
                text += (f"At $D=20$ for {escaped(dataset)} data, mean gap is "
                         f"{fmt(number(risk, 'selection_gap_mean'))} for risk and "
                         f"{fmt(number(box, 'selection_gap_mean'))} for box. ")
    return text + "\n"


def statistical_tables(e):
    nulls = [number(r, "null_rejection_mean") for r in e.select("statistical") if r["method"] in PRIMARY]
    nulls = [v for v in nulls if v is not None]
    text = ""
    if nulls:
        text += (f"Common-test null rejection means range from {min(nulls):.3f} to "
                 f"{max(nulls):.3f} at nominal level 0.05. "
                 "Figures~\\ref{fig:power} and~\\ref{fig:recovery} show the mechanism-specific "
                 "power and recovery results.\n\n")
    text += "".join(mechanism_readout(e, dataset) for dataset in MECHANISMS)
    mean_global = e.lookup("statistical", "mean", "risk")
    mean_screen = e.lookup("statistical", "mean", "mmd")
    if mean_global and mean_screen:
        text += ("Independent classification follows a similar pattern on mean data: "
                 f"robust-classifier balanced error is {fmt(number(mean_global, 'robust_balanced_error_mean'))} "
                 f"for global selection and {fmt(number(mean_screen, 'robust_balanced_error_mean'))} "
                 f"for screening; SVM error is {fmt(number(mean_global, 'svm_balanced_error_mean'))} "
                 f"and {fmt(number(mean_screen, 'svm_balanced_error_mean'))}, respectively.\n\n")
    return text + training_bound_readout(e)


def real_tables(e):
    rows = []
    for dataset in ("wdbc", "ionosphere"):
        for i, method in enumerate(PRIMARY):
            r = e.lookup("real", dataset, method)
            if r:
                rows.append([("WDBC" if dataset == "wdbc" else "Ionosphere") if i == 0 else "", LABELS[method],
                             int(r.get("robust_balanced_error_n", 0)),
                             estimate(r, "robust_balanced_error", real=True),
                             estimate(r, "svm_balanced_error", real=True),
                             estimate(r, "robust_auc", real=True)])
    text = longtable("Common classifiers on real datasets. Entries are mean $\\pm$ repeated-split SD. BER is thresholded balanced error; AUC uses the robust classifier's scores.",
                     ["Dataset", "Selector", "$n$", "Robust BER", "SVM BER", "Robust AUC"],
                     rows, first_width="20mm", label="tab:real-main")
    risk_rows = [r for r in e.rows if r["suite"] == "real" and r["method"] == "risk"]
    limits = sum(r["selection_status"] == "time_limit" for r in risk_rows)
    if risk_rows:
        text += (f"Global searches reach the selection limit in {limits}/{len(risk_rows)} splits. "
                 "Their classifier results use the retained incumbent supports. ")
    for dataset in ("wdbc", "ionosphere"):
        risk = e.lookup("real", dataset, "risk")
        screen = e.lookup("real", dataset, "mmd")
        if risk and screen:
            comparisons = []
            for metric, classifier in (("robust_balanced_error_mean", "robust classifier"),
                                       ("svm_balanced_error_mean", "SVM")):
                left, right = number(risk, metric), number(screen, metric)
                if left is None or right is None:
                    continue
                relation = "lower" if left < right else "higher" if left > right else "equal"
                comparisons.append(f"{relation} observed {classifier} error")
            text += (f"On {'WDBC' if dataset == 'wdbc' else 'Ionosphere'}, global selection has " + " and ".join(comparisons)
                     + " than marginal screening. ")
    return text + "\n"


def sensitivity_tables(e):
    return "\n\n".join(sensitivity_readout(e, dataset) for dataset in ("mean", "mixture"))


def appendix_tables(e):
    return "\n".join([
        r"\subsection{Original study metadata}", run_metadata(e),
        r"\subsection{Full optimization comparisons}", full_computational_tables(e),
        r"\subsection{Testing, recovery and independent prediction}", full_statistical_tables(e),
        r"\subsection{Real-data prediction and selection}", full_real_tables(e),
        r"\subsection{Radius sensitivity}", full_sensitivity_tables(e),
        supplementary_tables(e),
    ])


def render(evidence, template):
    blocks = {"OVERVIEW": overview(evidence), "COMPUTATIONAL_TABLE": computational_tables(evidence),
              "STATISTICAL_TABLE": statistical_tables(evidence), "REAL_TABLE": real_tables(evidence),
              "SENSITIVITY_TABLE": sensitivity_tables(evidence), "RATIO_TABLE": computational_tables(evidence, "ratio"),
              "REMEDIATION": remediation(evidence), "APPENDIX_TABLES": appendix_tables(evidence)}
    for marker in MARKERS:
        token = "%%" + marker + "%%"
        if template.count(token) != 1:
            raise ValueError(f"Template must have exactly one {token}")
        template = template.replace(token, blocks[marker])
    if re.search(r"%%[A-Z_]+%%", template):
        raise ValueError("Unreplaced report marker")
    return "\n".join(line.rstrip() for line in template.splitlines()) + "\n"


def main(argv=None):
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument("--root", type=Path, default=project / "results/research_20261003")
    parser.add_argument("--template", type=Path, default=project / "reports/report_template.tex")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--draft", action="store_true", help="Explicitly mark incomplete input as a draft")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = (Path(tempfile.mkdtemp(prefix="robusthw-report-draft-")) / "experiment_report.tex"
                       if args.draft else args.project_root / "reports/experiment_report.tex")
    if args.draft:
        temporary = Path(tempfile.gettempdir()).resolve()
        if not args.output.resolve().is_relative_to(temporary) or args.output.parent.resolve() == temporary:
            raise ValueError("Draft output must be inside a task-specific system TEMP directory")
    evidence = Evidence(args.project_root.resolve(), args.root.resolve(), args.draft)
    source = render(evidence, args.template.read_text(encoding="utf-8"))
    files = [args.root / "summary.json", args.root / "per_trial.csv", args.template]
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
    template_source = args.template.read_text(encoding="utf-8")
    figure_stems = set(re.findall(r"\\ResearchFigure\{([a-z_]+)\}", template_source))
    figure_stems.update(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([a-z_]+)\.pdf\}", template_source))
    for stem in sorted(figure_stems):
        asset = args.template.parent / "figures" / (stem + ".pdf")
        if asset.is_file():
            files.append(asset)
        elif not args.draft:
            raise ValueError(f"Required figure PDF is missing: {asset}")
    input_hashes = {}
    for path in files:
        resolved = path.resolve()
        label = (resolved.relative_to(args.project_root.resolve()).as_posix()
                 if resolved.is_relative_to(args.project_root.resolve()) else str(resolved))
        input_hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(state="draft" if args.draft else "complete", records=len(evidence.rows),
                    expected_records=len(evidence.expected), missing_records=len(evidence.missing),
                    inputs=input_hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"source": str(args.output), **{k: v for k, v in manifest.items() if k != "inputs"}}))


if __name__ == "__main__":
    main()
