"""Run the D = 20 experiments (src/robusthw/experiments/study20.py).

  OMP_NUM_THREADS=1 PYTHONPATH=src python3 scripts/run_study20.py accuracy    --workers 60
  OMP_NUM_THREADS=1 PYTHONPATH=src python3 scripts/run_study20.py computation --workers 60
  PYTHONPATH=src python3 scripts/run_study20.py summarize

computation reads the radius that cross-validation chose in accuracy.json.
"""

import argparse
import json
from pathlib import Path

from robusthw.experiments import study20, study20_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["accuracy", "computation", "summarize"])
    parser.add_argument("--config", default="configs/study20.json")
    parser.add_argument("--output", default="results/study20-v2")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.command == "accuracy":
        study20.accuracy(config, args.output, args.workers)
    elif args.command == "computation":
        study20.computation(config, args.output, args.workers)
    study20_summary.summarize(args.output)


if __name__ == "__main__":
    main()
