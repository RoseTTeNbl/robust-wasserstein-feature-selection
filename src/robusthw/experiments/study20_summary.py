"""Tables and figures for the 20-dimensional checks (results/study20-*/summary.md)."""

import json
from pathlib import Path

import numpy as np

LEVELS = ["fixed", "honest_x0.25", "honest_x0.5", "honest_x1", "honest_x2"]
LABELS = {"fixed": "fixed 0.6", "honest_x0.25": "0.25 rho_h", "honest_x0.5": "0.5 rho_h",
          "honest_x1": "rho_h", "honest_x2": "2 rho_h"}


def _load(folder, name):
    return json.loads((Path(folder) / name).read_text())


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _slope(sizes, values):
    return float(np.polyfit(np.log(sizes), np.log(values), 1)[0])


def null_section(folder, development, figures):
    config, radii = development["config"], development["honest_radii"]
    records = _load(folder, "null.json")["records"]
    sizes, dims = config["train_sizes"], [*config["budgets"], config["dimension"]]
    theory = {1: "-1/2", 2: "-1/2 (log)", 3: "-1/3",
              config["dimension"]: f"-1/{config['dimension']}"}
    rows, curves = [], {}
    for d in dims:
        maxima, medians = [], []
        for n in sizes:
            chosen = [r["sizes"][str(d)] for r in records if r["n"] == n]
            needed = np.array([c["needed_max"] for c in chosen])
            honest = radii[str(d)][str(n)]
            maxima.append(needed.mean())
            medians.append(np.mean([c["needed_median"] for c in chosen]))
            below = sum(c["below_lower_bound"] for c in chosen) / sum(c["subsets"] for c in chosen)
            rows.append([
                d, n, f"{honest:.3f}", f"{needed.mean():.3f}", f"{medians[-1]:.3f}",
                f"{chosen[0]['lower_bound']:.3f}", f"{below:.3f}",
                f"{np.mean(needed > honest):.2f}",
                f"{np.mean(needed > config['fixed_radius']):.2f}",
                f"{np.mean([c['fixed_radius_best_value'] for c in chosen]):.2f}",
            ])
        curves[d] = (maxima, medians)
    slopes = _table(
        ["d", "slope of max_S radius", "slope of median_S radius", "theory (W1 rate)"],
        [[d, f"{_slope(sizes, curves[d][0]):.2f}", f"{_slope(sizes, curves[d][1]):.2f}", theory[d]]
         for d in dims],
    )
    _null_figure(sizes, curves, config, figures)
    return "\n\n".join([
        "## Experiment 1: identical classes (false separation)",
        f"Both classes N(0, I_{config['dimension']}), {len(records) // len(sizes)} confirmation "
        f"draws per n. 'Needed radius' of S is the smallest rho_0+rho_1 with V(S) >= "
        f"{2 * config['null_mass']:.2f}; above it the certificate does not separate identical "
        "classes. rho_h is the development 95% quantile of the largest needed radius over all "
        "subsets of size d (frozen before confirmation).",
        _table(
            ["d", "n", "rho_h (dev)", "max_S needed", "median_S needed", "Prop. 8 bound",
             "subsets below bound", "false sep. at rho_h", "false sep. at 0.6",
             "best V at 0.6"],
            rows,
        ),
        "Log-log slopes of the needed radius against n (mean over draws):",
        slopes,
        "![needed radius](figures/null_radius.png)",
    ])


def _null_figure(sizes, curves, config, figures):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from matplotlib.ticker import NullFormatter

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for d, (maxima, medians) in curves.items():
        line, = ax.plot(sizes, maxima, "o-", label=f"d={d}, worst subset")
        ax.plot(sizes, medians, "s--", color=line.get_color(), alpha=0.6,
                label=f"d={d}, median subset")
    ax.axhline(config["fixed_radius"], color="gray", lw=1, ls=":", label="fixed radius 0.6")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("samples per class n")
    ax.set_ylabel("radius needed (rho_0 + rho_1)")
    ax.set_title("Identical classes: radius that avoids false separation")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(figures / "null_radius.png", dpi=160)
    plt.close(fig)


def signal_section(folder, development, figures):
    config = development["config"]
    data = _load(folder, "signal.json")
    records, sizes = data["records"], development["config"]["train_sizes"]
    shift, true_size = development["chosen_shift"], len(config["signal"])
    rows, adaptive_rows, worst_gap, curves = [], [], 0.0, {}

    def is_valid(fit):
        return fit["test_hinge"] <= fit["certificate"] + 2 * fit["test_hinge_se"]

    def inside(record):
        return record["budgets"][str(record["adaptive"])]["signal_hits"] == true_size

    for level in LEVELS:
        for n in sizes:
            chosen = [r["levels"][level] for r in records if r["n"] == n]
            fits = [c["detector_true_size"] for c in chosen]
            recovered = np.mean([c["budgets"][str(true_size)]["signal_hits"] == true_size
                                 for c in chosen])
            valid = np.mean([is_valid(f) for f in fits])
            rows.append([
                LABELS[level], n,
                f"{np.mean([c['budgets'][str(true_size)]['radius'] for c in chosen]):.2f}",
                f"{recovered:.2f}", f"{np.mean([f['certificate'] for f in fits]):.3f}",
                f"{np.mean([f['test_hinge'] for f in fits]):.3f}", f"{valid:.2f}",
                f"{np.mean([f['test_errors'] for f in fits]):.3f}",
            ])
            adaptive = [c["detector_adaptive"] for c in chosen]
            picks = [c["adaptive"] for c in chosen]
            adaptive_rows.append([
                LABELS[level], n,
                " / ".join(f"{np.mean(np.array(picks) == d):.2f}" for d in config["budgets"]),
                f"{np.mean([inside(c) for c in chosen]):.2f}",
                f"{np.mean([f['certificate'] for f in adaptive]):.3f}",
                f"{np.mean([f['test_hinge'] for f in adaptive]):.3f}",
                f"{np.mean([is_valid(f) for f in adaptive]):.2f}",
            ])
            curves[(level, n)] = (recovered, valid, np.mean([f["certificate"] for f in fits]),
                                  np.mean([f["test_hinge"] for f in fits]))
            for f in fits + adaptive:
                worst_gap = max(worst_gap, abs(f["core_upper"] - f["certificate"]))
    bayes = 2 * (1 - (2 * _phi(shift * np.sqrt(true_size) / 2) - 1))
    _signal_figure(sizes, curves, bayes, figures)
    return "\n\n".join([
        "## Experiment 2: signal on two coordinates (validity, recovery, choice of d)",
        f"Class 1 is shifted by {shift} in coordinates {config['signal']} (chosen on development "
        f"seeds); Bayes hinge risk on the signal pair B = {bayes:.3f}, Bayes total error "
        f"{bayes / 2:.3f}. {len(records) // len(sizes)} confirmation draws per n; test sets of "
        f"{config['test_size']} per class. A certificate is counted valid when the test hinge "
        "risk is at most the certificate plus two Monte Carlo standard errors.",
        f"Budget d = {true_size} (the true size):",
        _table(["radius", "n", "radius used", "exact recovery", "certificate V(S_hat)",
                "test hinge", "valid", "test total error"], rows),
        f"Budget chosen by the certificate among d in {config['budgets']} (Corollary 11):",
        _table(["radius", "n", "share of d=1 / 2 / 3", "signal pair inside", "certificate",
                "test hinge", "valid"], adaptive_rows),
        f"Largest |core fit_fixed value - 2 m_S| over all fitted detectors: {worst_gap:.1e}.",
        "![signal](figures/signal_levels.png)",
    ])


def _phi(x):
    from statistics import NormalDist

    return NormalDist().cdf(x)


def _signal_figure(sizes, curves, bayes, figures):
    """Recovery, validity and certificate against the radius as a multiple of rho_h (d=2)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = [level for level in LEVELS if level.startswith("honest_x")]
    factors = [float(level.removeprefix("honest_x")) for level in levels]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for n in sizes:
        values = np.array([curves[(level, n)] for level in levels])
        line, = axes[0].plot(factors, values[:, 0], "o-", label=f"n={n}")
        axes[1].plot(factors, values[:, 1], "o-", color=line.get_color(), label=f"n={n}")
        axes[2].plot(factors, values[:, 2], "o-", color=line.get_color(),
                     label=f"n={n}, certificate")
        axes[2].plot(factors, values[:, 3], "s--", color=line.get_color(), alpha=0.7,
                     label=f"n={n}, test hinge")
    axes[2].axhline(bayes, color="gray", lw=1, ls=":", label="Bayes hinge risk")
    titles = ["exact recovery of the signal pair", "certificate valid on test data",
              "certificate vs test hinge risk"]
    for ax, title in zip(axes, titles):
        ax.set_xscale("log", base=2)
        ax.set_xticks(factors)
        ax.set_xticklabels([f"{f:g}" for f in factors])
        ax.set_xlabel("radius / rho_h")
        ax.set_title(f"{title} (d=2)", fontsize=9)
    for ax in axes[:2]:
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=7)
    axes[2].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "signal_levels.png", dpi=160)
    plt.close(fig)


def computation_section(folder):
    records = _load(folder, "computation.json")["records"]
    rows = []
    for r in records:
        e, c, o = r["enumeration"], r["compact_program"], r["outer_approximation"]
        rows.append([
            r["n"], r["budget"], f"{e['value']:.4f} ({e['seconds']:.1f}s)",
            "--" if c["value"] is None
            else f"{c['value']:.4f} ({c['seconds']:.1f}s, {c['nodes']} nodes)",
            "--" if r["relaxation"]["value"] is None else f"{r['relaxation']['value']:.4f}",
            f"{o['upper']:.4f} / {0 if o['lower'] is None else o['lower']:.4f}",
            f"{o['status']}, {o['evaluated_subsets']} subsets, {o['seconds']:.0f}s",
        ])
    return "\n\n".join([
        "## Experiment 3: computation (rho_0 + rho_1 = 0.6)",
        "Enumeration evaluates every subset through the matching form; program (2) is solved by "
        "HiGHS; outer approximation is the core Algorithm 1 (select method 'oa').",
        _table(["n", "d", "enumeration", "program (2)", "relaxation of (2)",
                "OA upper / lower", "OA status"], rows),
    ])


def randomized_section(folder):
    records = _load(folder, "randomized.json")["records"]
    rows = []
    for budget in sorted({r["budget"] for r in records}):
        chosen = [r for r in records if r["budget"] == budget]
        water = [r["water_filling"] for r in chosen if "water_filling" in r]
        rows.append([
            budget, len(chosen), f"{np.mean([r['fixed_value'] for r in chosen]):.3f}",
            f"{np.mean([r['randomized_value'] for r in chosen]):.3f}",
            f"{np.mean(water):.3f}" if water else "--",
            f"{np.mean([r['relaxation'] for r in chosen]):.3f}",
            f"{np.mean([r['fixed_test_hinge'] for r in chosen]):.3f}",
            f"{np.mean([r['mixture_test_hinge'] for r in chosen]):.3f}",
        ])
    return "\n\n".join([
        "## Experiment 4: randomized selection",
        "Worst-case values against the adversary (left) and test hinge risk when the data are "
        "simply drawn from the true distributions (right). W_T uses tangents at 23 radii; for "
        "d=1 the exact value is the water-filling level of Theorem 19.",
        _table(["d", "draws", "fixed min V", "randomized W_T", "water-filling", "relaxation",
                "test hinge, fixed", "test hinge, mixture"], rows),
    ])


def summarize(folder):
    folder = Path(folder)
    figures = folder / "figures"
    figures.mkdir(exist_ok=True)
    development = _load(folder, "development.json")
    environment = development["environment"]
    parts = [
        "# 20-dimensional checks of the research directions",
        "Development checks, not formal performance results. Generated by "
        "`scripts/run_study20.py summarize`.",
        f"Model version `{environment['model_version']}`, code commit "
        f"`{environment['git_commit']}`, Python {environment['python']}, "
        f"NumPy {environment['numpy']}, SciPy {environment['scipy']}."
        + (" The working tree had uncommitted changes." if environment.get("git_dirty") else ""),
        *([(folder / "provenance.md").read_text().strip()]
          if (folder / "provenance.md").exists() else []),
        "Proposition, theorem and equation numbers refer to the research-directions note "
        "`note/directions/research_directions.pdf`.",
        f"Development: honest radii from {len(development['config']['development_seeds'])} null "
        f"draws per n; signal shift chosen from {development['config']['shift_grid']} "
        f"(exact-recovery rate at n={max(development['config']['train_sizes'])}, rho_h: "
        f"{development['shift_recovery']}) -> {development['chosen_shift']}.",
        null_section(folder, development, figures),
        signal_section(folder, development, figures),
        computation_section(folder),
        randomized_section(folder),
    ]
    (folder / "summary.md").write_text("\n\n".join(parts) + "\n")
