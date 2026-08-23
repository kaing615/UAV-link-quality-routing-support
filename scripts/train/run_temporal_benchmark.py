"""Run the bounded within-run temporal Edge-SAGE--GRU benchmark."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from scripts.train.run_multihorizon_benchmark import (
    aggregate_metrics,
    collect_metrics,
    discover_datasets,
    select_representative_runs,
)

MODEL_ID = "edge-sage-gru"


def _run(command: list[str]) -> None:
    print("[RUN]", " ".join(command))
    subprocess.run(command, check=True)


def run_temporal_benchmark(
    data_root: Path,
    output_root: Path,
    summary_csv: Path,
    targets: set[str] | None = None,
    horizons: set[int] | None = None,
    runs_per_scenario: int = 10,
    epochs: int = 100,
    patience: int = 10,
    window_size: int = 4,
    force: bool = False,
) -> pd.DataFrame:
    """Train the temporal model on a reproducible bounded coordinate set."""
    targets = {"survival"} if targets is None else targets
    horizons = {3, 5} if horizons is None else horizons
    datasets = [
        item
        for item in discover_datasets(data_root)
        if item[0] in targets and item[1] in horizons
    ]
    selected = {
        (target, horizon, run)
        for target, horizon in sorted({item[:2] for item in datasets})
        for run in select_representative_runs(
            [item[2].name for item in datasets if item[:2] == (target, horizon)],
            runs_per_scenario,
        )
    }
    datasets = [item for item in datasets if (*item[:2], item[2].name) in selected]
    if not datasets:
        raise FileNotFoundError(f"No selected temporal graph datasets found under {data_root}")

    for target, horizon, run_root in datasets:
        output_dir = output_root / MODEL_ID / target / f"k{horizon}" / run_root.name
        if (output_dir / "metrics.csv").exists() and not force:
            print(f"[SKIP] {MODEL_ID}/{target}/k{horizon}/{run_root.name}")
            continue
        _run(
            [
                sys.executable,
                "-m",
                "src.training.gnn.train_temporal_gnn",
                "--run-name",
                run_root.name,
                "--data-root",
                str(data_root / target / f"k{horizon}"),
                "--output-dir",
                str(output_dir),
                "--window-size",
                str(window_size),
                "--epochs",
                str(epochs),
                "--patience",
                str(patience),
            ]
        )

    detailed = collect_metrics(output_root)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(summary_csv, index=False)
    aggregate_path = summary_csv.with_name(f"{summary_csv.stem}_aggregate.csv")
    if detailed.empty:
        pd.DataFrame().to_csv(aggregate_path, index=False)
    else:
        aggregate_metrics(detailed).to_csv(aggregate_path, index=False)
    print(f"[OK] wrote {summary_csv} ({len(detailed)} metric rows)")
    print(f"[OK] wrote {aggregate_path}")
    return detailed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/multihorizon"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/temporal"))
    parser.add_argument("--summary", type=Path, default=Path("reports/temporal_benchmark_summary.csv"))
    parser.add_argument("--targets", nargs="+", choices=("survival",), default=["survival"])
    parser.add_argument("--horizons", nargs="+", type=int, choices=(3, 5), default=[3, 5])
    parser.add_argument("--runs-per-scenario", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_temporal_benchmark(
        args.data_root,
        args.output_root,
        args.summary,
        set(args.targets),
        set(args.horizons),
        args.runs_per_scenario,
        args.epochs,
        args.patience,
        args.window_size,
        args.force,
    )
