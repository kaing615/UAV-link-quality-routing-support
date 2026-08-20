"""Public artifact checks for the multi-horizon benchmark runner."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from scripts.train.run_multihorizon_benchmark import aggregate_metrics, collect_metrics, discover_datasets

from src.training.baselines.loro_baselines import load_run_rows


def test_discovery_and_collection_preserve_benchmark_coordinates(tmp_path: Path):
    data_root = tmp_path / "data"
    run_root = data_root / "qos" / "k2" / "run-a"
    (run_root / "graph_dataset").mkdir(parents=True)
    for split in ("train", "val", "test"):
        (run_root / "graph_dataset" / f"{split}.pt").touch()

    assert discover_datasets(data_root) == [("qos", 2, run_root)]

    metrics_path = tmp_path / "outputs" / "logreg" / "qos" / "k2" / "run-a" / "metrics.csv"
    metrics_path.parent.mkdir(parents=True)
    pd.DataFrame([{"split": "test", "macro_f1": 0.75}]).to_csv(metrics_path, index=False)

    result = collect_metrics(tmp_path / "outputs")
    row = result.iloc[0]
    assert (row["model_id"], row["target"], row["horizon"], row["run_name"]) == (
        "logreg",
        "qos",
        2,
        "run-a",
    )


def test_aggregate_metrics_keeps_run_as_the_independent_unit():
    detailed = pd.DataFrame(
        [
            {"model_id": "xgb", "target": "qos", "horizon": 1, "split": "test", "run_name": "a", "macro_f1": 0.6},
            {"model_id": "xgb", "target": "qos", "horizon": 1, "split": "test", "run_name": "b", "macro_f1": 0.8},
        ]
    )

    row = aggregate_metrics(detailed).iloc[0]
    assert row["n_runs"] == 2
    assert row["macro_f1_mean"] == 0.7


def test_aggregate_metrics_reports_run_level_bootstrap_interval():
    detailed = pd.DataFrame(
        [
            {"model_id": "xgb", "target": "qos", "horizon": 1, "split": "test", "run_name": run, "macro_f1": 0.7}
            for run in ("a", "b", "c")
        ]
    )

    row = aggregate_metrics(detailed).iloc[0]
    assert row["macro_f1_ci95_low"] == pytest.approx(0.7)
    assert row["macro_f1_ci95_high"] == pytest.approx(0.7)


def test_loro_baseline_loader_accepts_multihorizon_data_root(tmp_path: Path):
    run_root = tmp_path / "run-a"
    (run_root / "features").mkdir(parents=True)
    (run_root / "splits").mkdir()
    row = {
        "time": 0,
        "distance": 100.0,
        "rssi": -60.0,
        "snr": 30.0,
        "delay": 2.0,
        "packet_loss": 0.0,
        "relative_speed": 1.0,
        "throughput": 10.0,
        "label": 1,
    }
    pd.DataFrame([row]).to_csv(run_root / "features" / "edges_labeled.csv", index=False)
    pd.DataFrame([{"time": 0, "split": "train"}]).to_csv(run_root / "splits" / "time_splits.csv", index=False)

    result = load_run_rows(tmp_path, "run-a", ["train"])
    assert len(result) == 1
    assert result.iloc[0]["label"] == 1
