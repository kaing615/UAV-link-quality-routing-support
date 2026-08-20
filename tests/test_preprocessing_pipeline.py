"""Integration checks for leakage-safe multi-horizon preprocessing."""

from __future__ import annotations

import pandas as pd
import torch

from src.preprocessing.run_preprocessing import run_pipeline


def test_pipeline_builds_survival_target_on_common_support_without_split_leakage(tmp_path):
    times = range(20)
    nodes = pd.DataFrame(
        [
            {
                "time": time,
                "node_id": node_id,
                "x": float(node_id),
                "y": 0.0,
                "z": 50.0,
                "vx": 1.0,
                "vy": 0.0,
                "vz": 0.0,
                "speed": 1.0,
                "degree": 1,
            }
            for time in times
            for node_id in (0, 1)
        ]
    )
    edges = pd.DataFrame(
        [
            {
                "time": time,
                "src": 0,
                "dst": 1,
                "connected": 0 if time == 1 else 1,
                "distance": 100.0,
                "rssi": -60.0,
                "snr": 25.0,
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

    outputs = run_pipeline(
        nodes_csv=nodes_csv,
        edges_csv=edges_csv,
        output_root=tmp_path / "processed",
        target="survival",
        horizon=2,
        common_max_horizon=5,
        train_ratio=0.5,
        val_ratio=0.25,
    )

    labeled = pd.read_csv(outputs["edges_labeled"])
    assert labeled["time"].max() == 14
    assert labeled.loc[labeled["time"] == 0, "label"].item() == 0
    assert set(labeled["target"]) == {"survival"}
    assert set(labeled["horizon"]) == {2}

    splits = pd.read_csv(outputs["splits"])
    retained = {name: splits.loc[splits["split"] == name, "time"].tolist() for name in ("train", "val", "test")}
    assert max(retained["train"]) + 2 < min(retained["val"])
    assert max(retained["val"]) + 2 < min(retained["test"])

    for split, expected_times in retained.items():
        graphs = torch.load(outputs[f"{split}_pt"], weights_only=False)
        assert [graph["time"] for graph in graphs] == expected_times
        assert {(graph["target"], graph["horizon"], graph["support_horizon"]) for graph in graphs} == {
            ("survival", 2, 5)
        }
