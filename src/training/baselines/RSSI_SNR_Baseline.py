from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

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

MODEL_ID = "threshold"
MODEL_NAME = "RSSI_SNR_Threshold"
DEFAULT_OUTPUT_DIR = Path("outputs/baselines/threshold")


# =========================
# Rule-based model
# =========================
class ThresholdModel:
    def __init__(self, rssi_thresh: float, snr_thresh: float):
        self.rssi_thresh = rssi_thresh
        self.snr_thresh = snr_thresh

    def predict(self, X):
        rssi = X["rssi"].values
        snr = X["snr"].values

        # Rule: nếu cả 2 đều tốt → class 1, ngược lại → 0
        return ((rssi >= self.rssi_thresh) & (snr >= self.snr_thresh)).astype(int)

    def predict_proba(self, X):
        preds = self.predict(X)
        # fake probability (0 hoặc 1)
        return np.vstack([1 - preds, preds]).T


# =========================
# Grid search threshold
# =========================
def find_best_threshold(train_df):
    X, y = extract_xy(train_df)

    rssi_values = np.linspace(X["rssi"].min(), X["rssi"].max(), 20)
    snr_values = np.linspace(X["snr"].min(), X["snr"].max(), 20)

    best_score = -1
    best_rssi = None
    best_snr = None

    for rssi_t in rssi_values:
        for snr_t in snr_values:
            preds = ((X["rssi"] >= rssi_t) & (X["snr"] >= snr_t)).astype(int)
            score = (preds == y).mean()  # accuracy đơn giản

            if score > best_score:
                best_score = score
                best_rssi = rssi_t
                best_snr = snr_t

    return best_rssi, best_snr, best_score


# =========================
# Fit model
# =========================
def fit_threshold(weighted_train_df, oversampled_train_df):
    # Ưu tiên weighted nếu có
    train_df = weighted_train_df if "sample_weight" in weighted_train_df.columns else oversampled_train_df

    rssi_t, snr_t, score = find_best_threshold(train_df)

    model = ThresholdModel(rssi_t, snr_t)
    return model, "threshold_search", rssi_t, snr_t, score


# =========================
# Args
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate RSSI/SNR threshold baseline.")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--train-weighted", type=Path, default=DEFAULT_TRAIN_WEIGHTED)
    parser.add_argument("--train-oversampled", type=Path, default=DEFAULT_TRAIN_OVERSAMPLED)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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

    model, train_strategy, rssi_t, snr_t, train_score = fit_threshold(
        weighted_train_df, oversampled_train_df
    )

    val_metrics, val_predictions = evaluate_split(model, MODEL_ID, MODEL_NAME, val_df, "val")
    test_metrics, test_predictions = evaluate_split(model, MODEL_ID, MODEL_NAME, test_df, "test")

    metadata = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "train_strategy": train_strategy,
        "feature_columns": FEATURE_COLUMNS,
        "rssi_threshold": float(rssi_t),
        "snr_threshold": float(snr_t),
        "train_accuracy": float(train_score),
    }

    save_outputs(
        output_dir,
        model,
        metadata,
        [val_metrics, test_metrics],
        {"val": val_predictions, "test": test_predictions},
    )

    print("[OK] Threshold baseline finished.")
    print(f"- rssi_threshold : {rssi_t}")
    print(f"- snr_threshold  : {snr_t}")


if __name__ == "__main__":
    main()