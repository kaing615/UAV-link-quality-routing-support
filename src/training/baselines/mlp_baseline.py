from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.neural_network import MLPClassifier

from src.training.baselines.common import (
    DEFAULT_TEST,
    DEFAULT_TRAIN_OVERSAMPLED,
    DEFAULT_TRAIN_WEIGHTED,
    DEFAULT_VAL,
    FEATURE_COLUMNS,
    evaluate_split,
    extract_xy,
    load_dataframe,
    resolve_paths,
    save_outputs,
)

MODEL_ID = "mlp"
MODEL_NAME = "Small MLP"
DEFAULT_OUTPUT_DIR = Path("outputs/baselines/mlp")


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


def fit_mlp(weighted_train_df, oversampled_train_df, random_state: int) -> tuple[MLPClassifier, str]:
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
    train_weighted, train_oversampled, val_csv, test_csv, output_dir = resolve_paths(args, MODEL_ID, DEFAULT_OUTPUT_DIR)

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

    model, train_strategy = fit_mlp(weighted_train_df, oversampled_train_df, args.random_state)
    val_metrics, val_predictions = evaluate_split(model, MODEL_ID, MODEL_NAME, val_df, "val")
    test_metrics, test_predictions = evaluate_split(model, MODEL_ID, MODEL_NAME, test_df, "test")

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
    save_outputs(output_dir, model, metadata, [val_metrics, test_metrics], {"val": val_predictions, "test": test_predictions})

    print("[OK] MLP baseline finished.")
    print(f"- model_id      : {MODEL_ID}")
    print(f"- model_name    : {MODEL_NAME}")
    print(f"- train_strategy: {train_strategy}")
    print(f"- metrics       : {output_dir / 'metrics.csv'}")
    print(f"- metadata      : {output_dir / 'metadata.json'}")
    print(f"- model         : {output_dir / 'model.pkl'}")


if __name__ == "__main__":
    main()
