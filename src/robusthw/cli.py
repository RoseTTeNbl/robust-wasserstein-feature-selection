"""The small command-line surface for the mathematical core and frozen studies."""

import argparse
from pathlib import Path

from robusthw.core import Problem, SolveOptions
from robusthw.core.selection import select


def parser():
    main = argparse.ArgumentParser(description="Wasserstein robust coordinate selection")
    commands = main.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Solve a small generated problem")
    demo.add_argument("--budget", type=int, default=2)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument(
        "--method", choices=("global", "enumeration", "forward", "forward_swap"), default="global"
    )
    demo.add_argument("--output", type=Path, default=Path("outputs/demo.json"))
    fetch = commands.add_parser("fetch-data", help="Fetch and verify the four official UCI sources")
    fetch.add_argument("--data-root", type=Path, default=Path("data/uci"))
    fetch.add_argument("--cache-root", type=Path, action="append", default=[])
    freeze = commands.add_parser(
        "freeze", help="Freeze an explicit study grid and its source snapshot"
    )
    freeze.add_argument(
        "--suite",
        choices=(
            "main",
            "anchor",
            "public",
            "shift",
            "small",
            "scaling",
            "ablation",
            "development",
        ),
        required=True,
    )
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--seed", type=int, default=20261018)
    freeze.add_argument("--methods", nargs="+")
    run = commands.add_parser("run", help="Execute or resume a frozen protocol")
    run.add_argument("protocol", type=Path)
    run.add_argument("--data-root", type=Path, default=Path("data/uci"))
    run.add_argument("--reuse-root", type=Path, help="Matching E1 output directory for E2")
    run.add_argument("--limit", type=int, help="Explicit partial execution; never a complete study")
    run.add_argument(
        "--workers", type=int, default=1, help="Parallel data sets; scaling requires 1"
    )
    analyze = commands.add_parser("summarize", help="Summarize all outcomes, including failures")
    analyze.add_argument("output_root", type=Path)
    analyze.add_argument("--destination", type=Path)
    return main


def main(argv=None):
    args = parser().parse_args(argv)
    from robusthw.experiments.storage import canonical_bytes, write_json

    if args.command == "demo":
        import numpy as np

        from robusthw.experiments.storage import fit_payload

        rng = np.random.default_rng(args.seed)
        x0, x1 = rng.normal(size=(5, 6)), rng.normal(size=(5, 6))
        x0[:, :2] -= 1
        x1[:, :2] += 1
        result = select(
            Problem(x0, x1, (0.1, 0.1)),
            args.budget,
            args.method,
            SolveOptions(),
            initial_support=tuple(range(args.budget)) if args.method == "global" else None,
        )
        payload = {
            "method": args.method,
            "status": result.status,
            "support": result.support,
            "upper": result.upper,
            "lower": result.lower,
            "gap": result.gap,
            "runtime": result.runtime,
            "fit_count": result.fit_count,
            "trace": result.trace,
            "fit": fit_payload(result.fit),
        }
        write_json(args.output, payload)
        print(
            canonical_bytes(
                {key: payload[key] for key in ("status", "support", "upper", "lower", "gap")}
            ).decode()
        )
    elif args.command == "fetch-data":
        from robusthw.experiments.datasets import fetch_datasets

        print(canonical_bytes(fetch_datasets(args.data_root, args.cache_root)).decode())
    elif args.command == "freeze":
        from robusthw.experiments.runner import METHODS, freeze

        result = freeze(args.suite, args.output, args.seed, args.methods or METHODS)
        print(
            canonical_bytes(
                {
                    key: result[key]
                    for key in ("suite", "data_settings", "budget_tasks", "manifest_hash")
                }
            ).decode()
        )
    elif args.command == "run":
        from robusthw.experiments.runner import run

        result = run(
            args.protocol,
            args.data_root,
            reuse_root=args.reuse_root,
            limit=args.limit,
            workers=args.workers,
        )
        print(
            canonical_bytes(
                {key: value for key, value in result.items() if key != "records"}
            ).decode()
        )
        if result["failed"]:
            raise SystemExit(1)
    else:
        from robusthw.experiments.analysis import summarize

        print(canonical_bytes(summarize(args.output_root, args.destination)).decode())
