"""Run the 20-dimensional checks of the research-directions note.

  PYTHONPATH=src python3 scripts/run_study20.py develop   --output results/study20-v1
  PYTHONPATH=src python3 scripts/run_study20.py confirm   --output results/study20-v1
  PYTHONPATH=src python3 scripts/run_study20.py summarize --output results/study20-v1

develop fixes the honest radii and the signal strength on development seeds;
confirm reuses them unchanged on fresh seeds; summarize writes summary.md and figures.
"""

import argparse
import json
from pathlib import Path

from robusthw.experiments import study20, study20_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["develop", "confirm", "summarize"])
    parser.add_argument("--config", default="configs/study20.json")
    parser.add_argument("--output", default="results/study20-v1")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.command == "develop":
        radii, shift = study20.develop(config, args.output, args.workers)
        print(json.dumps({"honest_radii": radii, "shift": shift}, indent=1))
    elif args.command == "confirm":
        study20.confirm(config, args.output, args.workers)
    else:
        study20_summary.summarize(args.output)


if __name__ == "__main__":
    main()
