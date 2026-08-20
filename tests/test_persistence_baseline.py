"""Tests for the last-state persistence baseline."""

from __future__ import annotations

import pandas as pd

from src.preprocessing.run_preprocessing import run_pipeline
from src.training.baselines.persistence_baseline import PersistenceModel, run_batch


def test_persistence_model_reuses_current_target_state():
    features = pd.DataFrame(
        {
            "snr": [20.0, 17.0, 20.0, 20.0],
            "packet_loss": [0.01, 0.01, 0.20, 0.01],
            "delay": [2.0, 2.0, 2.0, 11.0],
        }
    )

    assert PersistenceModel("qos", 18.0, 0.10, 10.0).predict(features).tolist() == [1, 0, 0, 0]
    assert PersistenceModel("survival", 18.0, 0.10, 10.0).predict(features).tolist() == [1, 1, 1, 1]


def test_run_batch_evaluates_preprocessed_run(tmp_path):
    times = range(50)
    nodes = pd.DataFrame(
        [
            {
                "time": time,
                "node_id": node,
                "x": float(node),
                "y": 0.0,
                "z": 50.0,
                "vx": 1.0,
                "vy": 0.0,
                "vz": 0.0,
                "speed": 1.0,
                "degree": 1,
            }
            for time in times
            for node in (0, 1)
        ]
    )
    edges = pd.DataFrame(
        [
            {
                "time": time,
                "src": 0,
                "dst": 1,
                "connected": 1,
                "distance": 100.0,
                "rssi": -60.0,
                "snr": 25.0 if time % 2 else 10.0,
                "delay": 2.0,
                "packet_loss": 0.01,
                "relative_speed": 0.0,
                "throughput": 10.0,
                "p_stable": 0.9,
                "weight": 0.1,
            }
            for time in times
        ]
    )
    nodes_csv = tmp_path / "nodes.csv"
    edges_csv = tmp_path / "edges.csv"
    nodes.to_csv(nodes_csv, index=False)
    edges.to_csv(edges_csv, index=False)
    run_pipeline(
        nodes_csv,
        edges_csv,
        tmp_path / "data" / "qos" / "k1" / "run_001",
        target="qos",
        horizon=1,
        common_max_horizon=5,
    )

    summary_csv = tmp_path / "reports" / "persistence.csv"
    summary = run_batch(tmp_path / "data", tmp_path / "outputs", summary_csv)

    assert set(summary["split"]) == {"val", "test"}
    assert set(summary["model_id"]) == {"persistence"}
    assert set(summary["target"]) == {"qos"}
    assert summary_csv.exists()
    assert (tmp_path / "outputs" / "qos" / "k1" / "run_001" / "metrics.csv").exists()
