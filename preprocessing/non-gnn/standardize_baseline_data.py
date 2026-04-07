from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "distance",
    "rssi",
    "snr",
    "delay",
    "packet_loss",
    "relative_speed",
    "throughput",
]


def load_edges_with_splits(edges_labeled_csv: Path, splits_csv: Path) -> pd.DataFrame:
    edges = pd.read_csv(edges_labeled_csv)
    splits = pd.read_csv(splits_csv)

    required_edge_columns = set(FEATURE_COLUMNS + ["time", "label"])
    missing_edge_columns = sorted(required_edge_columns - set(edges.columns))
    if missing_edge_columns:
        raise ValueError(
            "Missing required columns in edges_labeled.csv: "
            + ", ".join(missing_edge_columns)
        )

    if "time" not in splits.columns or "split" not in splits.columns:
        raise ValueError("time_splits.csv must contain columns: time, split")

    df = edges.merge(splits, on="time", how="left")
    if df["split"].isna().any():
        missing_times = sorted(df.loc[df["split"].isna(), "time"].unique().tolist())
        raise ValueError(f"Missing split assignments for time values: {missing_times}")

    return df


def standardize_by_train_split(df: pd.DataFrame) -> tuple[StandardScaler, dict[str, pd.DataFrame]]:
    train_df = df[df["split"] == "train"].copy()
    if train_df.empty:
        raise ValueError("No train rows found. Check time_splits.csv.")

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])

    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name].copy()
        if split_df.empty:
            continue
        split_df.loc[:, FEATURE_COLUMNS] = scaler.transform(split_df[FEATURE_COLUMNS])
        split_frames[split_name] = split_df

    return scaler, split_frames


def scaler_stats_dict(scaler: StandardScaler) -> dict[str, dict[str, float]]:
    return {
        feature: {
            "mean": float(mean),
            "scale": float(scale),
        }
        for feature, mean, scale in zip(FEATURE_COLUMNS, scaler.mean_, scaler.scale_)
    }


def save_outputs(
    output_dir: Path,
    split_frames: dict[str, pd.DataFrame],
    scaler: StandardScaler,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}
    ordered_frames: list[pd.DataFrame] = []

    for split_name in ["train", "val", "test"]:
        if split_name not in split_frames:
            continue
        out_path = output_dir / f"{split_name}_scaled.csv"
        split_frames[split_name].to_csv(out_path, index=False)
        outputs[f"{split_name}_scaled"] = out_path
        ordered_frames.append(split_frames[split_name])

    all_scaled = pd.concat(ordered_frames, ignore_index=True)
    all_scaled_path = output_dir / "all_scaled.csv"
    all_scaled.to_csv(all_scaled_path, index=False)
    outputs["all_scaled"] = all_scaled_path

    scaler_stats_path = output_dir / "scaler_stats.json"
    scaler_stats_path.write_text(
        json.dumps(
            {
                "fit_on_split": "train",
                "feature_columns": FEATURE_COLUMNS,
                "stats": scaler_stats_dict(scaler),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    outputs["scaler_stats"] = scaler_stats_path
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize baseline tabular features using train split statistics."
    )
    parser.add_argument(
        "--edges-labeled",
        type=Path,
        default=Path("data/processed/edges_labeled.csv"),
        help="Path to labeled edge table.",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/splits/time_splits.csv"),
        help="Path to time-based train/val/test split file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/baseline_standardized"),
        help="Directory to save standardized CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_edges_with_splits(
        edges_labeled_csv=args.edges_labeled,
        splits_csv=args.splits,
    )
    scaler, split_frames = standardize_by_train_split(df)
    outputs = save_outputs(
        output_dir=args.output_dir,
        split_frames=split_frames,
        scaler=scaler,
    )

    print("[OK] Standardized baseline data saved.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
