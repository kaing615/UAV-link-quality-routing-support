from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

DEFAULT_TRAIN_CSV = Path("data/processed/baseline_standardized/train_scaled.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/baseline_standardized/imbalance")


def validate_label_column(df: pd.DataFrame, label_column: str) -> None:
    if label_column not in df.columns:
        raise ValueError(f"Missing label column: {label_column}")

    unique_labels = sorted(df[label_column].dropna().unique().tolist())
    if len(unique_labels) != 2:
        raise ValueError(
            f"Expected binary labels in column '{label_column}', found: {unique_labels}"
        )


def class_counts(df: pd.DataFrame, label_column: str) -> dict[str, int]:
    counts = df[label_column].value_counts().sort_index().to_dict()
    return {str(label): int(count) for label, count in counts.items()}


def infer_minority_majority_labels(df: pd.DataFrame, label_column: str) -> tuple[int, int]:
    counts = df[label_column].value_counts()
    minority_label = int(counts.idxmin())
    majority_label = int(counts.idxmax())
    return minority_label, majority_label


def add_sample_weights(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    weighted_df = df.copy()
    counts = weighted_df[label_column].value_counts()
    total = len(weighted_df)
    weight_map = {
        int(label): float(total / (2 * count))
        for label, count in counts.items()
    }
    weighted_df["sample_weight"] = weighted_df[label_column].map(weight_map).astype(float)
    return weighted_df


def oversample_minority_class(
    df: pd.DataFrame,
    label_column: str,
    target_ratio: float,
    random_state: int,
) -> pd.DataFrame:
    if target_ratio <= 0:
        raise ValueError("--target-ratio must be > 0")

    minority_label, majority_label = infer_minority_majority_labels(df, label_column)

    minority_df = df[df[label_column] == minority_label].copy()
    majority_df = df[df[label_column] == majority_label].copy()

    majority_count = len(majority_df)
    minority_count = len(minority_df)
    target_minority_count = max(minority_count, math.ceil(majority_count * target_ratio))

    extra_needed = target_minority_count - minority_count
    if extra_needed <= 0:
        return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    sampled_minority = minority_df.sample(
        n=extra_needed,
        replace=True,
        random_state=random_state,
    )

    balanced_df = pd.concat([majority_df, minority_df, sampled_minority], ignore_index=True)
    return balanced_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    train_csv = args.train_csv
    output_dir = args.output_dir

    if args.run_name:
        run_root = Path("data/graph_dataset") / args.run_name / "baseline_standardized"

        if train_csv == DEFAULT_TRAIN_CSV:
            train_csv = run_root / "train_scaled.csv"
        if output_dir == DEFAULT_OUTPUT_DIR:
            output_dir = run_root / "imbalance"

    return train_csv, output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Handle class imbalance for non-GNN baseline training data."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Preprocessed run name under data/graph_dataset/<RUN_NAME>/baseline_standardized. "
        "If provided, default train/output paths are resolved from that run.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=DEFAULT_TRAIN_CSV,
        help="Path to the standardized train CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save imbalance-handling outputs.",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Binary label column name.",
    )
    parser.add_argument(
        "--method",
        choices=["weights", "oversample", "both"],
        default="both",
        help="Imbalance handling strategy to export.",
    )
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=1.0,
        help="Minority/majority target ratio after oversampling. 1.0 means fully balanced.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for oversampling and shuffle.",
    )
    return parser.parse_args()


def validate_args(train_csv: Path, output_dir: Path) -> None:
    if not train_csv.exists():
        raise FileNotFoundError(f"train_scaled.csv not found: {train_csv}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    train_csv, output_dir = resolve_paths(args)
    validate_args(train_csv, output_dir)

    print(f"[RUN] run_name={args.run_name or 'default'}")
    print(f"- train_csv : {train_csv}")
    print(f"- output_dir: {output_dir}")

    train_df = pd.read_csv(train_csv)
    validate_label_column(train_df, args.label_column)

    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    summary = {
        "input_csv": str(train_csv),
        "label_column": args.label_column,
        "method": args.method,
        "target_ratio": args.target_ratio,
        "random_state": args.random_state,
        "original_class_counts": class_counts(train_df, args.label_column),
    }

    if args.method in {"weights", "both"}:
        weighted_df = add_sample_weights(train_df, args.label_column)
        weighted_path = output_dir / "train_weighted.csv"
        weighted_df.to_csv(weighted_path, index=False)
        outputs["train_weighted"] = str(weighted_path)
        summary["weight_class_counts"] = class_counts(weighted_df, args.label_column)
        summary["weight_map"] = {
            str(label): float(
                weighted_df.loc[
                    weighted_df[args.label_column] == label,
                    "sample_weight",
                ].iloc[0]
            )
            for label in sorted(weighted_df[args.label_column].unique().tolist())
        }

    if args.method in {"oversample", "both"}:
        oversampled_df = oversample_minority_class(
            df=train_df,
            label_column=args.label_column,
            target_ratio=args.target_ratio,
            random_state=args.random_state,
        )
        oversampled_path = output_dir / "train_oversampled.csv"
        oversampled_df.to_csv(oversampled_path, index=False)
        outputs["train_oversampled"] = str(oversampled_path)
        summary["oversampled_class_counts"] = class_counts(oversampled_df, args.label_column)

    summary["outputs"] = outputs

    summary_path = output_dir / "imbalance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[OK] Imbalance handling outputs saved.")
    for name, path in outputs.items():
        print(f"- {name}: {path}")
    print(f"- summary: {summary_path}")


if __name__ == "__main__":
    main()
