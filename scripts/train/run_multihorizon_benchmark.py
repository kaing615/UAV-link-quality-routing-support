"""Run and aggregate the within-run multi-horizon benchmark."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.evaluation.paired_statistics import bootstrap_mean_ci

MODELS = ("logreg", "xgb", "edge-sage")
BASELINE_MODULES = {
    "logreg": "src.training.baselines.Logistic_Regression_Baseline",
    "xgb": "src.training.baselines.xgb_baseline",
}


def select_representative_runs(runs: list[str], runs_per_scenario: int) -> list[str]:
    if runs_per_scenario <= 0:
        raise ValueError("runs_per_scenario must be positive")
    grouped: dict[str, list[str]] = {}
    for run in runs:
        match = re.match(r"^stress_(.+)_(?:rwp|gm)_s\d+$", run)
        grouped.setdefault(match.group(1) if match else "other", []).append(run)
    for group in grouped.values():
        group.sort(
            key=lambda run: (int(match.group(1)), run) if (match := re.search(r"_s(\d+)$", run)) else (sys.maxsize, run)
        )
    return [run for scenario in sorted(grouped) for run in grouped[scenario][:runs_per_scenario]]


def discover_datasets(data_root: Path) -> list[tuple[str, int, Path]]:
    datasets = []
    for graph_dir in sorted(data_root.glob("*/k*/*/graph_dataset")):
        if all((graph_dir / f"{split}.pt").exists() for split in ("train", "val", "test")):
            datasets.append(
                (graph_dir.parents[2].name, int(graph_dir.parents[1].name.removeprefix("k")), graph_dir.parent)
            )
    return datasets


def collect_metrics(output_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(output_root.glob("*/*/k*/*/metrics.csv")):
        frame = pd.read_csv(path)
        frame["model_id"] = path.parents[3].name
        frame["target"] = path.parents[2].name
        frame["horizon"] = int(path.parents[1].name.removeprefix("k"))
        frame["run_name"] = path.parent.name
        frame["source_metrics"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_metrics(detailed: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        name
        for name in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "macro_f1",
            "roc_auc",
            "pr_auc",
            "inference_ms_per_sample",
        )
        if name in detailed.columns
    ]
    group_columns = ["model_id", "target", "horizon", "split"]
    result = (
        detailed.groupby(group_columns)
        .agg(
            n_runs=("run_name", "nunique"),
            **{f"{metric}_{stat}": (metric, stat) for metric in metrics for stat in ("mean", "std")},
        )
        .reset_index()
    )
    for metric in metrics:
        intervals = (
            detailed.groupby(group_columns)[metric]
            .apply(lambda values: bootstrap_mean_ci(values.to_numpy()))
            .apply(pd.Series)
            .rename(columns={0: f"{metric}_ci95_low", 1: f"{metric}_ci95_high"})
            .reset_index()
        )
        result = result.merge(intervals, on=group_columns, validate="one_to_one")
    return result


def _run(command: list[str]) -> None:
    print("[RUN]", " ".join(command))
    subprocess.run(command, check=True)


def _ensure_tabular_data(run_root: Path) -> Path:
    baseline_root = run_root / "baseline_standardized"
    imbalance_root = baseline_root / "imbalance"
    required = [
        baseline_root / "val_scaled.csv",
        baseline_root / "test_scaled.csv",
        imbalance_root / "train_weighted.csv",
        imbalance_root / "train_oversampled.csv",
    ]
    if all(path.exists() for path in required):
        return baseline_root

    _run(
        [
            sys.executable,
            "src/preprocessing/non-gnn/standardize_baseline_data.py",
            "--edges-labeled",
            str(run_root / "features" / "edges_labeled.csv"),
            "--splits",
            str(run_root / "splits" / "time_splits.csv"),
            "--output-dir",
            str(baseline_root),
        ]
    )
    _run(
        [
            sys.executable,
            "src/preprocessing/non-gnn/handle_imbalance.py",
            "--train-csv",
            str(baseline_root / "train_scaled.csv"),
            "--output-dir",
            str(imbalance_root),
        ]
    )
    return baseline_root


def run_benchmark(
    data_root: Path,
    output_root: Path,
    summary_csv: Path,
    models: list[str],
    targets: set[str] | None = None,
    horizons: set[int] | None = None,
    limit: int | None = None,
    force: bool = False,
    gnn_epochs: int = 200,
    gnn_patience: int = 20,
    runs_per_scenario: int | None = None,
) -> pd.DataFrame:
    datasets = [
        item
        for item in discover_datasets(data_root)
        if (targets is None or item[0] in targets) and (horizons is None or item[1] in horizons)
    ]
    if limit is not None and runs_per_scenario is not None:
        raise ValueError("limit and runs_per_scenario are mutually exclusive")
    if runs_per_scenario is not None:
        selected = {
            (target, horizon, run)
            for target, horizon in sorted({(item[0], item[1]) for item in datasets})
            for run in select_representative_runs(
                [item[2].name for item in datasets if item[:2] == (target, horizon)],
                runs_per_scenario,
            )
        }
        datasets = [item for item in datasets if (item[0], item[1], item[2].name) in selected]
    elif limit is not None:
        datasets = datasets[:limit]
    if not datasets:
        raise FileNotFoundError(f"No multi-horizon graph datasets found under {data_root}")

    for target, horizon, run_root in datasets:
        combination_root = data_root / target / f"k{horizon}"
        for model in models:
            output_dir = output_root / model / target / f"k{horizon}" / run_root.name
            if (output_dir / "metrics.csv").exists() and not force:
                print(f"[SKIP] {model}/{target}/k{horizon}/{run_root.name}")
                continue
            if model in BASELINE_MODULES:
                baseline_root = _ensure_tabular_data(run_root)
                _run(
                    [
                        sys.executable,
                        "-m",
                        BASELINE_MODULES[model],
                        "--train-weighted",
                        str(baseline_root / "imbalance" / "train_weighted.csv"),
                        "--train-oversampled",
                        str(baseline_root / "imbalance" / "train_oversampled.csv"),
                        "--val",
                        str(baseline_root / "val_scaled.csv"),
                        "--test",
                        str(baseline_root / "test_scaled.csv"),
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            elif model == "edge-sage":
                _run(
                    [
                        sys.executable,
                        "-m",
                        "src.training.gnn.train_gnn",
                        "--run-name",
                        run_root.name,
                        "--data-root",
                        str(combination_root),
                        "--model",
                        "edge-sage",
                        "--hidden",
                        "128",
                        "--lr",
                        "5e-4",
                        "--lr-scheduler",
                        "--output-dir",
                        str(output_dir),
                        "--epochs",
                        str(gnn_epochs),
                        "--patience",
                        str(gnn_patience),
                    ]
                )
            else:
                raise ValueError(f"Unknown model: {model}")

    summary = collect_metrics(output_root)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    aggregate_path = summary_csv.with_name(f"{summary_csv.stem}_aggregate.csv")
    aggregate_metrics(summary).to_csv(aggregate_path, index=False)
    print(f"[OK] wrote {summary_csv} ({len(summary)} metric rows)")
    print(f"[OK] wrote {aggregate_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/multihorizon"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/multihorizon"))
    parser.add_argument("--summary", type=Path, default=Path("reports/multihorizon_benchmark_summary.csv"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--targets", nargs="+", choices=("qos", "survival"), default=None)
    parser.add_argument("--horizons", nargs="+", type=int, choices=(1, 2, 3, 5), default=None)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--limit", type=int, default=None)
    limits.add_argument("--runs-per-scenario", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--gnn-epochs", type=int, default=200)
    parser.add_argument("--gnn-patience", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        args.data_root,
        args.output_root,
        args.summary,
        args.models,
        set(args.targets) if args.targets else None,
        set(args.horizons) if args.horizons else None,
        args.limit,
        args.force,
        args.gnn_epochs,
        args.gnn_patience,
        args.runs_per_scenario,
    )
