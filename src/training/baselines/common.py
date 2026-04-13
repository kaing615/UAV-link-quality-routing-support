from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

FEATURE_COLUMNS = [
    "distance",
    "rssi",
    "snr",
    "delay",
    "packet_loss",
    "relative_speed",
    "throughput",
]

DEFAULT_TRAIN_WEIGHTED = Path("data/processed/baseline_standardized/imbalance/train_weighted.csv")
DEFAULT_TRAIN_OVERSAMPLED = Path("data/processed/baseline_standardized/imbalance/train_oversampled.csv")
DEFAULT_VAL = Path("data/processed/baseline_standardized/val_scaled.csv")
DEFAULT_TEST = Path("data/processed/baseline_standardized/test_scaled.csv")


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def extract_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = sorted(set(FEATURE_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if "label" not in df.columns:
        raise ValueError("Missing label column: label")
    return df[FEATURE_COLUMNS], df["label"]


def resolve_paths(args, model_id: str, default_output_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    train_weighted = args.train_weighted
    train_oversampled = args.train_oversampled
    val_csv = args.val
    test_csv = args.test
    output_dir = args.output_dir

    if args.run_name:
        run_root = Path("data/graph_dataset") / args.run_name / "baseline_standardized"
        imbalance_root = run_root / "imbalance"

        if train_weighted == DEFAULT_TRAIN_WEIGHTED:
            train_weighted = imbalance_root / "train_weighted.csv"
        if train_oversampled == DEFAULT_TRAIN_OVERSAMPLED:
            train_oversampled = imbalance_root / "train_oversampled.csv"
        if val_csv == DEFAULT_VAL:
            val_csv = run_root / "val_scaled.csv"
        if test_csv == DEFAULT_TEST:
            test_csv = run_root / "test_scaled.csv"
        if output_dir == default_output_dir:
            output_dir = Path("outputs/baselines") / model_id / args.run_name

    return train_weighted, train_oversampled, val_csv, test_csv, output_dir


def evaluate_split(model, model_id: str, model_name: str, df: pd.DataFrame, split_name: str) -> tuple[dict[str, object], pd.DataFrame]:
    X, y_true = extract_xy(df)
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    unique_labels = sorted(pd.Series(y_true).dropna().unique().tolist())
    has_both_classes = set(unique_labels) == {0, 1}

    metrics = {
        "model_id": model_id,
        "model_name": model_name,
        "split": split_name,
        "n_samples": int(len(df)),
        "unique_labels": unique_labels,
        "has_both_classes": bool(has_both_classes),
        "positive_ratio": float(pd.Series(y_true).mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    prediction_df = df.copy()
    prediction_df["pred_label"] = y_pred
    if y_score is not None:
        prediction_df["pred_score"] = y_score

    return metrics, prediction_df


def save_outputs(
    output_dir: Path,
    model,
    metadata: dict[str, object],
    metrics_rows: list[dict[str, object]],
    prediction_frames: dict[str, pd.DataFrame],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for split_name, prediction_df in prediction_frames.items():
        prediction_df.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)

    with (output_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)
