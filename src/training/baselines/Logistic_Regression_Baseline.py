from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.linear_model import LogisticRegression

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

MODEL_ID = "logreg"
MODEL_NAME = "Logistic Regression"
DEFAULT_OUTPUT_DIR = Path("outputs/baselines/logreg")


# =========================
# Build model
# =========================
def build_logreg(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight=None,  # có thể chỉnh sau
        random_state=random_state,
    )


# =========================
# Fit model
# =========================
def fit_logreg(weighted_train_df, oversampled_train_df, random_state: int):
    X_weighted, y_weighted = extract_xy(weighted_train_df)
    sample_weight = weighted_train_df["sample_weight"] if "sample_weight" in weighted_train_df.columns else None

    model = build_logreg(random_state=random_state)

    # ✅ Ưu tiên weighted
    if sample_weight is not None:
        model.fit(X_weighted, y_weighted, sample_weight=sample_weight)
        return model, "weighted"

    # ❌ fallback sang oversampled
    X_over, y_over = extract_xy(oversampled_train_df)
    model = build_logreg(random_state=random_state)
    model.fit(X_over, y_over)
    return model, "oversampled"


# =========================
# Args
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Logistic Regression baseline.")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--train-weighted", type=Path, default=DEFAULT_TRAIN_WEIGHTED)
    parser.add_argument("--train-oversampled", type=Path, default=DEFAULT_TRAIN_OVERSAMPLED)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


# =========================
# Main
# =========================
def main() -> None:
    args = parse_args()
    train_weighted, train_oversampled, val_csv, test_csv, output_dir = resolve_paths(
        args, MODEL_ID, DEFAULT_OUTPUT_DIR
    )

    print(f"[RUN] run_name={args.run_name or 'default'}")

    weighted_train_df = load_dataframe(train_weighted)
    oversampled_train_df = load_dataframe(train_oversampled)
    val_df = load_dataframe(val_csv)
    test_df = load_dataframe(test_csv)

    model, train_strategy = fit_logreg(
        weighted_train_df, oversampled_train_df, args.random_state
    )

    val_metrics, val_predictions = evaluate_split(model, MODEL_ID, MODEL_NAME, val_df, "val")
    test_metrics, test_predictions = evaluate_split(model, MODEL_ID, MODEL_NAME, test_df, "test")

    metadata = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "train_strategy": train_strategy,
        "feature_columns": FEATURE_COLUMNS,
        "penalty": model.penalty,
        "C": float(model.C),
        "solver": model.solver,
        "max_iter": int(model.max_iter),
    }

    save_outputs(
        output_dir,
        model,
        metadata,
        [val_metrics, test_metrics],
        {"val": val_predictions, "test": test_predictions},
    )

    print("[OK] Logistic Regression baseline finished.")
    print(f"- model_id      : {MODEL_ID}")
    print(f"- train_strategy: {train_strategy}")
    print(f"- metrics       : {output_dir / 'metrics.csv'}")
    print(f"- metadata      : {output_dir / 'metadata.json'}")
    print(f"- model         : {output_dir / 'model.pkl'}")


if __name__ == "__main__":
    main()