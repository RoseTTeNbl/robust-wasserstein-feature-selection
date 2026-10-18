"""Explicit, deduplicated experiment grids; generation never executes a method."""

import json
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

MODELS = ("mean", "variance", "correlation", "mixture")
CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"


@dataclass(frozen=True)
class ExperimentSpec:
    independent_data_id: str
    kind: str
    model: str
    D: int
    n: int
    s: int
    seed: int
    budgets: tuple[int, ...]
    metadata: dict

    def as_dict(self):
        return {
            "independent_data_id": self.independent_data_id,
            "kind": self.kind,
            "model": self.model,
            "D": self.D,
            "n": self.n,
            "s": self.s,
            "seed": self.seed,
            "budgets": list(self.budgets),
            "metadata": self.metadata,
        }


def _seed(base, key):
    tag = sha256(f"{int(base)}:{key}".encode()).digest()
    return int.from_bytes(tag[:4], "little")


def _spec(base, suite, model, D, n, s, repeat, budgets, **metadata):
    identity = f"{suite}/{model}/D{D}_n{n}_s{s}/r{repeat:03d}"
    kind = suite if suite in ("small", "ablation", "development") else "synthetic"
    return ExperimentSpec(
        identity,
        kind,
        model,
        D,
        n,
        s,
        _seed(base, identity),
        tuple(budgets),
        {"suite": suite, "repeat": repeat, **metadata},
    )


def _read(suite):
    path = CONFIG_ROOT / (suite + ".json")
    if not path.is_file():
        path = Path(sys.prefix) / "share" / "robusthw" / "configs" / (suite + ".json")
    if not path.is_file():
        raise FileNotFoundError(f"Missing packaged protocol configuration: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_plan(suite, seed=20261018):
    """Return one spec per data/setting, with shared budgets stored on that spec.

    `shift` references the exact main data IDs and seeds. Public n sweeps share
    split seeds, hence nested training samples, but differ in fitted transforms.
    Timing repeats are metadata, never additional independent data sets.
    """
    suite = {
        "synthetic": "main",
        "e1": "main",
        "e2": "shift",
        "e3": "scaling",
        "small_exact": "small",
        "real": "public",
    }.get(suite, suite)
    config = _read(suite)
    result = []
    if suite == "main":
        for model in config["models"]:
            for setting in config["settings"]:
                for repeat in range(config["repeats"]):
                    result.append(
                        _spec(
                            seed,
                            suite,
                            model,
                            setting["D"],
                            setting["n"],
                            setting["s"],
                            repeat,
                            setting["budgets"],
                            target_error=config["target_error"],
                            validation_per_class=config["validation_per_class"],
                            test_per_class=config["test_per_class"],
                        )
                    )
    elif suite == "anchor":
        for repeat in range(config["repeats"]):
            item = _spec(
                seed,
                suite,
                config["models"][0],
                config["D"],
                config["n"],
                config["s"],
                repeat,
                config["budgets"],
                parameter=config["parameter"],
                target_error=config["target_error"],
                validation_per_class=config["validation_per_class"],
                test_per_class=config["test_per_class"],
            )
            result.append(
                ExperimentSpec(
                    item.independent_data_id,
                    "anchor",
                    item.model,
                    item.D,
                    item.n,
                    item.s,
                    item.seed,
                    item.budgets,
                    item.metadata,
                )
            )
    elif suite == "public":
        for name, D in config["tabular_datasets"].items():
            for repeat in range(config["tabular_repeats"]):
                split_seed = _seed(seed, f"public/{name}/split{repeat}")
                conditions = {}
                for n, budget in config["tabular_conditions"]:
                    conditions.setdefault(n, []).append(budget)
                for n, budgets in conditions.items():
                    identity = f"public/{name}/n{n}/r{repeat:03d}"
                    result.append(
                        ExperimentSpec(
                            identity,
                            "tabular",
                            name,
                            D,
                            n,
                            0,
                            split_seed,
                            tuple(sorted(budgets)),
                            {"suite": suite, "repeat": repeat},
                        )
                    )
        for repeat in range(config["har_repeats"]):
            identity = f"public/har/r{repeat:03d}"
            result.append(
                ExperimentSpec(
                    identity,
                    "har",
                    "har",
                    561,
                    config["har_training_per_class"],
                    0,
                    _seed(seed, identity),
                    tuple(config["har_budgets"]),
                    {"suite": suite, "repeat": repeat},
                )
            )
    elif suite == "shift":
        for item in build_plan("main", seed):
            if (item.D, item.n, item.s) == (config["D"], config["n"], config["s"]):
                result.append(
                    ExperimentSpec(
                        item.independent_data_id,
                        "shift",
                        item.model,
                        item.D,
                        item.n,
                        item.s,
                        item.seed,
                        tuple(config["budgets"]),
                        {
                            **item.metadata,
                            "suite": suite,
                            "reuse_suite": "main",
                            "source_data_id": item.independent_data_id,
                            "delta_factors": config["delta_factors"],
                            "shift_types": config["shift_types"],
                            "refit": False,
                        },
                    )
                )
    elif suite == "small":
        for model in config["models"]:
            for n in config["sample_sizes"]:
                for repeat in range(config["repeats"]):
                    result.append(
                        _spec(
                            seed,
                            suite,
                            model,
                            config["D"],
                            n,
                            config["s"],
                            repeat,
                            config["budgets"],
                            radius_factor=config["radius_factor"],
                            outer_tolerance=config["outer_tolerance"],
                            validation_tolerance=config["validation_tolerance"],
                            exhaustive_sizes=config["cut_check_sizes"],
                            require_complete=config["require_complete"],
                        )
                    )
    elif suite == "scaling":
        center = tuple(config["center"])
        conditions = {center}
        for column, values in enumerate(
            (config["dimensions"], config["sample_sizes"], config["budgets"])
        ):
            for value in values:
                point = list(center)
                point[column] = value
                conditions.add(tuple(point))
        for model in config["models"]:
            for D, n, budget in sorted(conditions):
                for repeat in range(config["repeats"]):
                    # Budget-specific cases are independently sampled here, unlike E1 budgets.
                    item = _spec(
                        seed,
                        suite + f"_d{budget}",
                        model,
                        D,
                        n,
                        config["s"],
                        repeat,
                        [budget],
                        radius_factor=config["radius_factor"],
                        time_limit=config["time_limit"],
                        timing_repeats=(
                            config["center_timing_repeats"]
                            if (D, n, budget) == center
                            else config["other_timing_repeats"]
                        ),
                    )
                    result.append(
                        ExperimentSpec(
                            item.independent_data_id,
                            "scaling",
                            item.model,
                            D,
                            n,
                            config["s"],
                            item.seed,
                            item.budgets,
                            {**item.metadata, "suite": suite},
                        )
                    )
    elif suite == "ablation":
        for model in config["models"]:
            for repeat in range(config["repeats"]):
                result.append(
                    _spec(
                        seed,
                        suite,
                        model,
                        config["D"],
                        config["n"],
                        config["s"],
                        repeat,
                        config["budgets"],
                        radius_factors=config["radius_factors"],
                        radius_methods=config["radius_methods"],
                        initialization_methods=config["initialization_methods"],
                        additional_random_initializations=config[
                            "additional_random_initializations"
                        ],
                    )
                )
    elif suite == "development":
        for repeat in range(config["repeats"]):
            result.append(
                _spec(
                    seed,
                    suite,
                    config["models"][repeat % len(config["models"])],
                    config["D"],
                    config["n"],
                    config["s"],
                    repeat,
                    config["budgets"],
                    excluded_from_final_statistics=config["excluded_from_final_statistics"],
                )
            )
    else:
        raise ValueError(f"Unknown experiment suite: {suite}")
    if len({spec.independent_data_id for spec in result}) != len(result):
        raise ValueError("Duplicate experiment identity in protocol")
    return result
