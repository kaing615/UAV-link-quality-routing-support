"""Build isolated QoS/survival datasets for a small multi-horizon pilot."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

from src.preprocessing.gnn.build_labels import HORIZONS, TARGETS
from src.preprocessing.run_preprocessing import run_pipeline


def _scenario(run_dir: Path) -> dict:
    scenario_path = run_dir / "scenario.json"
    if not scenario_path.exists():
        raise FileNotFoundError(f"Missing scenario metadata: {scenario_path}")
    return json.loads(scenario_path.read_text(encoding="utf-8"))


def select_pilot_runs(raw_root: Path, pattern: str = "ns3big_*", limit: int = 3) -> list[str]:
    """Select deterministic runs while alternating available mobility models."""
    if limit <= 0:
        raise ValueError("limit must be positive")

    by_mobility: dict[str, deque[str]] = defaultdict(deque)
    for run_dir in sorted(raw_root.glob(pattern)):
        if run_dir.is_dir() and (run_dir / "nodes.csv").exists() and (run_dir / "edges.csv").exists():
            mobility = str(_scenario(run_dir).get("mobility_model", "unknown"))
            by_mobility[mobility].append(run_dir.name)

    selected: list[str] = []
    while len(selected) < limit and any(by_mobility.values()):
        for mobility in sorted(by_mobility):
            if by_mobility[mobility] and len(selected) < limit:
                selected.append(by_mobility[mobility].popleft())

    if not selected:
        raise FileNotFoundError(f"No complete raw runs matching {pattern!r} under {raw_root}")
    return selected


def run_pilot(raw_root: Path, run_names: list[str], output_root: Path, summary_csv: Path) -> pd.DataFrame:
    rows: list[dict] = []
    support_horizon = max(HORIZONS)

    for run_name in run_names:
        run_dir = raw_root / run_name
        nodes_csv = run_dir / "nodes.csv"
        edges_csv = run_dir / "edges.csv"
        if not nodes_csv.exists() or not edges_csv.exists():
            raise FileNotFoundError(f"Incomplete raw run: {run_dir}")
        scenario = _scenario(run_dir)

        for target in TARGETS:
            for horizon in HORIZONS:
                destination = output_root / target / f"k{horizon}" / run_name
                outputs = {
                    "edges_labeled": destination / "features" / "edges_labeled.csv",
                    "splits": destination / "splits" / "time_splits.csv",
                    "train_pt": destination / "graph_dataset" / "train.pt",
                    "val_pt": destination / "graph_dataset" / "val.pt",
                    "test_pt": destination / "graph_dataset" / "test.pt",
                }
                complete = all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
                if complete:
                    print(f"[SKIP] {run_name}: target={target}, horizon={horizon}")
                else:
                    print(f"[PILOT] {run_name}: target={target}, horizon={horizon}")
                    outputs = run_pipeline(
                        nodes_csv=nodes_csv,
                        edges_csv=edges_csv,
                        output_root=destination,
                        target=target,
                        horizon=horizon,
                        common_max_horizon=support_horizon,
                    )

                labels = pd.read_csv(outputs["edges_labeled"])
                splits = pd.read_csv(outputs["splits"])
                labeled_splits = labels.merge(splits, on="time", validate="many_to_one")
                positive_count = int(labels["label"].sum())
                row = {
                    "run_name": run_name,
                    "mobility": scenario.get("mobility_model", "unknown"),
                    "target": target,
                    "horizon": horizon,
                    "support_horizon": support_horizon,
                    "num_samples": len(labels),
                    "positive_count": positive_count,
                    "negative_count": len(labels) - positive_count,
                    "positive_ratio": positive_count / max(len(labels), 1),
                    **{
                        f"num_{split}_times": int((splits["split"] == split).sum())
                        for split in ("train", "val", "test", "purged")
                    },
                    "output_root": str(destination),
                }
                for split in ("train", "val", "test"):
                    split_labels = labeled_splits[labeled_splits["split"] == split]["label"]
                    row[f"num_{split}_samples"] = len(split_labels)
                    row[f"{split}_positive_ratio"] = float(split_labels.mean())
                rows.append(row)

    summary = pd.DataFrame(rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    print(f"[OK] wrote {summary_csv} ({len(summary)} target/horizon/run rows)")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw_snapshots"))
    parser.add_argument("--output-root", type=Path, default=Path("data/multihorizon"))
    parser.add_argument("--summary", type=Path, default=Path("reports/multihorizon_pilot_summary.csv"))
    parser.add_argument("--pattern", default="ns3big_*")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--runs", nargs="*", default=None, help="Explicit run names; otherwise select a balanced pilot."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_names = args.runs or select_pilot_runs(args.raw_root, args.pattern, args.limit)
    print(f"[PILOT] selected runs: {', '.join(run_names)}")
    run_pilot(args.raw_root, run_names, args.output_root, args.summary)


if __name__ == "__main__":
    main()
