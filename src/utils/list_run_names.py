from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List RUN_NAME directories under data/graph_dataset.")
    parser.add_argument("pattern", nargs="?", default="*", help="Glob pattern for RUN_NAME. Default: *")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("data/graph_dataset"),
        help="Root directory containing run folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root
    pattern = args.pattern

    if not runs_root.is_dir():
        raise SystemExit(f"Runs root not found: {runs_root}")

    matched = sorted(
        path.name
        for path in runs_root.iterdir()
        if path.is_dir() and fnmatch.fnmatch(path.name, pattern)
    )

    if not matched:
        raise SystemExit(f"No RUN_NAME matched pattern '{pattern}' in {runs_root}")

    for run_name in matched:
        print(run_name)


if __name__ == "__main__":
    main()
