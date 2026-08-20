"""Create run-level statistical and worst-group generalization reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.paired_statistics import paired_comparisons

DEFAULT_COMPARISONS = [
    ("edge-sage", "logreg"),
    ("edge-sage", "xgb"),
    ("logreg", "persistence"),
    ("xgb", "persistence"),
]
DEFAULT_METRICS = {"macro_f1": True, "pr_auc": True}


def load_run_metadata(raw_root: Path, run_names: list[str]) -> pd.DataFrame:
    rows = []
    for run_name in sorted(set(run_names)):
        scenario = json.loads((raw_root / run_name / "scenario.json").read_text(encoding="utf-8"))
        x_span = float(scenario["x_limit"][1]) - float(scenario["x_limit"][0])
        y_span = float(scenario["y_limit"][1]) - float(scenario["y_limit"][0])
        area_km2 = x_span * y_span / 1_000_000
        rows.append(
            {
                "run_name": run_name,
                "mobility": scenario.get("mobility_model", "unknown"),
                "num_uavs": int(scenario["num_uavs"]),
                "comm_range": float(scenario["comm_range"]),
                "spatial_density_per_km2": float(scenario["num_uavs"]) / area_km2,
            }
        )
    result = pd.DataFrame(rows)
    median_density = result["spatial_density_per_km2"].median()
    result["density_group"] = np.where(
        result["spatial_density_per_km2"] <= median_density,
        "sparse",
        "dense",
    )
    return result


def paired_model_comparisons(
    detailed: pd.DataFrame,
    comparisons: list[tuple[str, str]] = DEFAULT_COMPARISONS,
    metrics: dict[str, bool] = DEFAULT_METRICS,
    n_resamples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    group_columns = ["target", "horizon", "split"]
    for coordinates, group in detailed.groupby(group_columns, sort=True):
        result = paired_comparisons(
            group.rename(columns={"model_id": "strategy"}),
            comparisons=comparisons,
            metrics=metrics,
            n_resamples=n_resamples,
            seed=seed,
        )
        for column, value in zip(group_columns, coordinates, strict=True):
            result[column] = value
        result["primary_endpoint"] = result["metric"] == "macro_f1"
        rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def worst_group_metrics(
    detailed: pd.DataFrame,
    metadata: pd.DataFrame,
    metrics: tuple[str, ...] = ("macro_f1", "pr_auc"),
) -> pd.DataFrame:
    merged = detailed.merge(metadata, on="run_name", how="left", validate="many_to_one")
    coordinates = ["model_id", "target", "horizon", "split"]
    rows = []
    for group_type, column in (("mobility", "mobility"), ("density", "density_group")):
        for metric in metrics:
            grouped = (
                merged.groupby([*coordinates, column], dropna=False)[metric]
                .agg([("n_runs", "count"), ("metric_mean", "mean"), ("metric_std", "std")])
                .reset_index()
                .rename(columns={column: "group_value"})
            )
            grouped["group_type"] = group_type
            grouped["metric"] = metric
            grouped["is_worst_group"] = grouped["metric_mean"].eq(
                grouped.groupby(coordinates)["metric_mean"].transform("min")
            )
            rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("reports/multihorizon_benchmark_summary.csv"))
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw_snapshots"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/stage6"))
    args = parser.parse_args()

    detailed = pd.read_csv(args.summary)
    test = detailed[detailed["split"] == "test"].copy()
    metadata = load_run_metadata(args.raw_root, test["run_name"].unique().tolist())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.output_dir / "run_metadata.csv", index=False)
    paired_model_comparisons(test).to_csv(args.output_dir / "paired_model_comparisons.csv", index=False)
    worst_group_metrics(test, metadata).to_csv(args.output_dir / "worst_group_metrics.csv", index=False)
    print(f"[OK] wrote generalization and statistical reports to {args.output_dir}")


if __name__ == "__main__":
    main()
