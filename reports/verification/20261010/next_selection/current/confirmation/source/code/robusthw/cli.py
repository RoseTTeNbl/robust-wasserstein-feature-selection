"""Small command-line entry point; all experiment orchestration is separate."""
import argparse
import json
from pathlib import Path

import numpy as np

from .problem import Problem
from .solver import enumerate_supports, solve


def main(argv=None):
    parser = argparse.ArgumentParser(description="Wasserstein robust feature selection")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--theta", type=float, default=.25)
    parser.add_argument("--loss", choices=["overlap", "logistic"], default="overlap")
    parser.add_argument("--gate", choices=["risk", "triangle", "box"], default="risk")
    parser.add_argument("--cut-mode", choices=["tangent", "ratio"], default="tangent")
    parser.add_argument("--search", choices=["adaptive", "master"], default="adaptive")
    parser.add_argument("--proposals", type=int, default=8)
    parser.add_argument("--linear-bounds", action="store_true",
                        help="use the earlier tangent-only adaptive comparison")
    parser.add_argument("--time-limit", type=float, default=30.)
    parser.add_argument("--root-rounds", type=int, default=5,
                        help="fractional preprocessing for the legacy master search")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--enumerate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output exists; choose a new path to preserve results")
    if args.data:
        with np.load(args.data, allow_pickle=False) as stored:
            arrays = {k: stored[k] for k in ["x0", "x1", "q0", "q1"] if k in stored}
    else:
        rng = np.random.default_rng(args.seed)
        arrays = dict(x0=rng.normal(size=(4, 6)), x1=rng.normal(size=(4, 6)))
        arrays["x1"][:, :2] += 1.5
    p = Problem(**arrays, budget=args.budget, theta0=args.theta, theta1=args.theta,
                loss=args.loss)
    result = solve(p, gate=args.gate, cut_mode=args.cut_mode,
                   time_limit=args.time_limit, root_rounds=args.root_rounds,
                   search=args.search, proposal_count=args.proposals,
                   use_fractions=not args.linear_bounds)
    record = dict(version="0.6.0", result=result.to_dict())
    if args.enumerate:
        record["enumeration"] = enumerate_supports(p, time_limit=args.time_limit).to_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(status=result.status, support=result.support, lower=result.lower,
                          upper=result.upper, seconds=result.seconds, output=str(args.output))))
    return 0 if result.status == "optimal" else 2


if __name__ == "__main__":
    raise SystemExit(main())
