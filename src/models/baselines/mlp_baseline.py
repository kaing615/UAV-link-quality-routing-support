from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier

MODEL_ID = "mlp"
MODEL_NAME = "Small MLP"
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
DEFAULT_OUTPUT_DIR = Path("outputs/baselines/mlp")


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)


def build_mlp(random_state: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_state,
    )


def extract_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = sorted(set(FEATURE_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if "label" not in df.columns:
        raise ValueError("Missing label column: label")
    return df[FEATURE_COLUMNS], df["label"]


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
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
        if output_dir == DEFAULT_OUTPUT_DIR:
            output_dir = Path("outputs/baselines") / MODEL_ID / args.run_name

    return train_weighted, train_oversampled, val_csv, test_csv, output_dir


def fit_mlp(
    weighted_train_df: pd.DataFrame,
    oversampled_train_df: pd.DataFrame,
    random_state: int,
) -> tuple[MLPClassifier, str]:
    X_weighted, y_weighted = extract_xy(weighted_train_df)
    sample_weight = weighted_train_df["sample_weight"] if "sample_weight" in weighted_train_df.columns else None

    model = build_mlp(random_state=random_state)
    if sample_weight is not None:
        try:
            model.fit(X_weighted, y_weighted, sample_weight=sample_weight)
            return model, "weighted"
        except TypeError:
            pass

    X_over, y_over = extract_xy(oversampled_train_df)
    model = build_mlp(random_state=random_state)
    model.fit(X_over, y_over)
    return model, "oversampled"


def evaluate_split(model: MLPClassifier, df: pd.DataFrame, split_name: str) -> tuple[dict[str, object], pd.DataFrame]:
    X, y_true = extract_xy(df)
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    unique_labels = sorted(pd.Series(y_true).dropna().unique().tolist())
    has_both_classes = set(unique_labels) == {0, 1}

    metrics = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
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
    model: MLPClassifier,
    train_strategy: str,
    metrics_rows: list[dict[str, object]],
    prediction_frames: dict[str, pd.DataFrame],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(metrics_rows).to_csv(output_dir / "metrics.csv", index=False)

    metadata = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "train_strategy": train_strategy,
        "feature_columns": FEATURE_COLUMNS,
        "n_iter": int(model.n_iter_),
        "loss_curve_length": len(model.loss_curve_),
        "best_validation_score": (
            float(model.best_validation_score_)
            if getattr(model, "best_validation_score_", None) is not None
            else None
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    for split_name, prediction_df in prediction_frames.items():
        prediction_df.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)

    with (output_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the non-GNN MLP baseline.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Preprocessed run name under data/graph_dataset/<RUN_NAME>/baseline_standardized. "
        "If provided, default input/output paths are resolved from that run.",
    )
    parser.add_argument("--train-weighted", type=Path, default=DEFAULT_TRAIN_WEIGHTED)
    parser.add_argument("--train-oversampled", type=Path, default=DEFAULT_TRAIN_OVERSAMPLED)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_weighted, train_oversampled, val_csv, test_csv, output_dir = resolve_paths(args)

    print(f"[RUN] run_name={args.run_name or 'default'}")
    print(f"- train_weighted   : {train_weighted}")
    print(f"- train_oversampled: {train_oversampled}")
    print(f"- val              : {val_csv}")
    print(f"- test             : {test_csv}")
    print(f"- output_dir       : {output_dir}")

    weighted_train_df = load_dataframe(train_weighted)
    oversampled_train_df = load_dataframe(train_oversampled)
    val_df = load_dataframe(val_csv)
    test_df = load_dataframe(test_csv)

    model, train_strategy = fit_mlp(
        weighted_train_df=weighted_train_df,
        oversampled_train_df=oversampled_train_df,
        random_state=args.random_state,
    )

    val_metrics, val_predictions = evaluate_split(model, val_df, "val")
    test_metrics, test_predictions = evaluate_split(model, test_df, "test")

    save_outputs(
        output_dir=output_dir,
        model=model,
        train_strategy=train_strategy,
        metrics_rows=[val_metrics, test_metrics],
        prediction_frames={"val": val_predictions, "test": test_predictions},
    )

    print("[OK] MLP baseline finished.")
    print(f"- model_id      : {MODEL_ID}")
    print(f"- model_name    : {MODEL_NAME}")
    print(f"- train_strategy: {train_strategy}")
    print(f"- metrics       : {output_dir / 'metrics.csv'}")
    print(f"- metadata      : {output_dir / 'metadata.json'}")
    print(f"- model         : {output_dir / 'model.pkl'}")


if __name__ == "__main__":
    main()
