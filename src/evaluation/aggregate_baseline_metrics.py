from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

import pandas as pd

METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "macro_f1",
    "n_samples",
    "positive_ratio",
    "tn",
    "fp",
    "fn",
    "tp",
]


def infer_scenario(run_name: str) -> str:
    lowered = run_name.lower()
    if "_gm_" in lowered:
        return "gauss-markov"
    if "_rwp_" in lowered:
        return "random-waypoint"
    if "dense" in lowered:
        return "dense"
    if "sparse" in lowered:
        return "sparse"
    if "fast" in lowered:
        return "fast"
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate baseline metrics across runs and models.")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs/baselines"),
        help="Root directory containing outputs/baselines/<MODEL_ID>/<RUN_NAME>/metrics.csv",
    )
    parser.add_argument(
        "--model-pattern",
        type=str,
        default="*",
        help="Glob pattern for model_id directories under outputs-root.",
    )
    parser.add_argument(
        "--run-pattern",
        type=str,
        default="*",
        help="Glob pattern for run_name directories under each model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/aggregates/baselines"),
        help="Directory to save aggregated CSV files.",
    )
    return parser.parse_args()


def collect_detail_rows(outputs_root: Path, model_pattern: str, run_pattern: str) -> pd.DataFrame:
    if not outputs_root.is_dir():
        raise FileNotFoundError(f"Outputs root not found: {outputs_root}")

    frames: list[pd.DataFrame] = []
    for model_dir in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
        if not fnmatch.fnmatch(model_dir.name, model_pattern):
            continue

        for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
            if not fnmatch.fnmatch(run_dir.name, run_pattern):
                continue

            metrics_path = run_dir / "metrics.csv"
            if not metrics_path.exists():
                continue

            df = pd.read_csv(metrics_path)
            df["model_dir"] = model_dir.name
            df["run_name"] = run_dir.name
            df["scenario"] = df["run_name"].map(infer_scenario)
            frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No metrics.csv found under {outputs_root} for model_pattern='{model_pattern}' "
            f"and run_pattern='{run_pattern}'"
        )

    detail_df = pd.concat(frames, ignore_index=True)
    if "model_id" not in detail_df.columns:
        detail_df["model_id"] = detail_df["model_dir"]
    return detail_df


def aggregate(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metric_columns = [col for col in METRIC_COLUMNS if col in df.columns]
    grouped = df.groupby(group_columns, dropna=False)[metric_columns]
    mean_df = grouped.mean().reset_index()
    std_df = grouped.std(ddof=0).reset_index()

    rename_mean = {col: f"{col}_mean" for col in metric_columns}
    rename_std = {col: f"{col}_std" for col in metric_columns}

    mean_df = mean_df.rename(columns=rename_mean)
    std_df = std_df.rename(columns=rename_std)
    return mean_df.merge(std_df, on=group_columns, how="left")


def main() -> None:
    args = parse_args()
    detail_df = collect_detail_rows(args.outputs_root, args.model_pattern, args.run_pattern)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = output_dir / "detailed_metrics.csv"
    summary_model_path = output_dir / "summary_by_model_split.csv"
    summary_scenario_path = output_dir / "summary_by_scenario_model_split.csv"

    detail_df.to_csv(detail_path, index=False)
    aggregate(detail_df, ["model_id", "split"]).to_csv(summary_model_path, index=False)
    aggregate(detail_df, ["scenario", "model_id", "split"]).to_csv(summary_scenario_path, index=False)

    print("[OK] Aggregated baseline metrics saved.")
    print(f"- detailed_metrics          : {detail_path}")
    print(f"- summary_by_model_split    : {summary_model_path}")
    print(f"- summary_by_scenario_split : {summary_scenario_path}")


if __name__ == "__main__":
    main()
