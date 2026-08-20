"""Evaluate a no-training baseline that carries the current link state forward."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.gnn.build_labels import HORIZONS, TARGETS
from src.training.baselines.common import evaluate_split, save_outputs

MODEL_ID = "persistence"
MODEL_NAME = "Last-state Persistence"


class PersistenceModel:
    def __init__(self, target: str, tau_snr: float, tau_loss: float, tau_delay: float):
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}")
        self.target = target
        self.tau_snr = tau_snr
        self.tau_loss = tau_loss
        self.tau_delay = tau_delay

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.target == "survival":
            return np.ones(len(features), dtype=int)
        return (
            (
                (features["snr"] >= self.tau_snr)
                & (features["packet_loss"] <= self.tau_loss)
                & (features["delay"] <= self.tau_delay)
            )
            .astype(int)
            .to_numpy()
        )

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        predictions = self.predict(features)
        return np.column_stack((1 - predictions, predictions))


def _single_value(frame: pd.DataFrame, column: str):
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"Expected one {column!r} value, found {values}")
    return values[0]


def evaluate_run(run_root: Path, output_dir: Path) -> list[dict]:
    labels = pd.read_csv(run_root / "features" / "edges_labeled.csv")
    splits = pd.read_csv(run_root / "splits" / "time_splits.csv")
    data = labels.merge(splits, on="time", validate="many_to_one")

    target = str(_single_value(labels, "target"))
    horizon = int(_single_value(labels, "horizon"))
    support_horizon = int(_single_value(labels, "support_horizon"))
    tau_snr = float(_single_value(labels, "tau_snr"))
    tau_loss = float(_single_value(labels, "tau_loss"))
    tau_delay = float(_single_value(labels, "tau_delay"))
    model = PersistenceModel(target, tau_snr, tau_loss, tau_delay)

    metrics_rows: list[dict] = []
    predictions: dict[str, pd.DataFrame] = {}
    for split in ("val", "test"):
        split_data = data[data["split"] == split].copy()
        if split_data.empty:
            raise ValueError(f"No {split} rows in {run_root}")
        metrics, split_predictions = evaluate_split(model, MODEL_ID, MODEL_NAME, split_data, split)
        metrics.update(
            {
                "run_name": run_root.name,
                "target": target,
                "horizon": horizon,
                "support_horizon": support_horizon,
            }
        )
        metrics_rows.append(metrics)
        predictions[split] = split_predictions

    save_outputs(
        output_dir,
        model,
        {
            "model_id": MODEL_ID,
            "model_name": MODEL_NAME,
            "definition": "current QoS state for target=qos; current connectivity for target=survival",
            "target": target,
            "horizon": horizon,
            "support_horizon": support_horizon,
            "tau_snr": tau_snr,
            "tau_loss": tau_loss,
            "tau_delay": tau_delay,
        },
        metrics_rows,
        predictions,
    )
    return metrics_rows


def run_batch(data_root: Path, output_root: Path, summary_csv: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for target in TARGETS:
        for horizon in HORIZONS:
            combination_root = data_root / target / f"k{horizon}"
            if not combination_root.exists():
                continue
            for run_root in sorted(path for path in combination_root.iterdir() if path.is_dir()):
                print(f"[PERSISTENCE] {run_root.name}: target={target}, horizon={horizon}")
                rows.extend(
                    evaluate_run(
                        run_root,
                        output_root / target / f"k{horizon}" / run_root.name,
                    )
                )

    if not rows:
        raise FileNotFoundError(f"No multi-horizon runs found under {data_root}")
    summary = pd.DataFrame(rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    print(f"[OK] wrote {summary_csv} ({len(summary)} metric rows)")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/multihorizon"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/multihorizon/persistence"))
    parser.add_argument("--summary", type=Path, default=Path("reports/persistence_pilot_summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_batch(args.data_root, args.output_root, args.summary)


if __name__ == "__main__":
    main()
