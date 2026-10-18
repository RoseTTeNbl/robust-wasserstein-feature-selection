"""Convenience entry; the same implementation is exposed by robusthw fetch-data."""

import argparse

from robusthw.experiments.datasets import fetch_datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/uci")
    parser.add_argument("--cache-root", action="append", default=[])
    args = parser.parse_args()
    fetch_datasets(args.data_root, args.cache_root)
