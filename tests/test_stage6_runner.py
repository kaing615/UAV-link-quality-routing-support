"""Generalization benchmark runner contracts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from scripts.train.run_stage6_benchmark import (
    collect_protocol_metrics,
    discover_coordinates,
    with_full_ablation_reference,
)

from src.training.gnn.train_gnn_loro import load_run_split


def test_discover_coordinates_groups_runs_by_target_and_horizon(tmp_path: Path):
    for run in ("run-a", "run-b"):
        graph_dir = tmp_path / "qos" / "k5" / run / "graph_dataset"
        graph_dir.mkdir(parents=True)
        for split in ("train", "val", "test"):
            (graph_dir / f"{split}.pt").touch()

    assert discover_coordinates(tmp_path, {"qos"}, {5}) == [("qos", 5, ["run-a", "run-b"])]


def test_collect_protocol_metrics_preserves_test_run_from_file(tmp_path: Path):
    path = tmp_path / "loro" / "logreg" / "qos" / "k1" / "fold-a" / "metrics.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [{"split": "test", "run_name": "run-a", "model_id": "logreg", "macro_f1": 0.8}]
    ).to_csv(path, index=False)

    row = collect_protocol_metrics(tmp_path / "loro", "loro").iloc[0]
    assert (row["protocol"], row["target"], row["horizon"], row["run_name"]) == (
        "loro",
        "qos",
        1,
        "run-a",
    )


def test_collect_protocol_metrics_uses_output_directory_when_metric_has_no_run_name(tmp_path: Path):
    path = tmp_path / "ablation" / "edge-sage-noedge" / "qos" / "k1" / "run-a" / "metrics.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame([{"split": "test", "model_id": "edge-sage-noedge", "macro_f1": 0.4}]).to_csv(
        path, index=False
    )

    row = collect_protocol_metrics(tmp_path / "ablation", "ablation").iloc[0]
    assert row["run_name"] == "run-a"


def test_loro_gnn_loader_accepts_multihorizon_data_root(tmp_path: Path):
    graph_dir = tmp_path / "run-a" / "graph_dataset"
    graph_dir.mkdir(parents=True)
    graph = {
        "x": torch.zeros((2, 8)),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "edge_attr": torch.zeros((2, 7)),
        "edge_label_index": torch.tensor([[0], [1]]),
        "edge_label": torch.tensor([1]),
    }
    torch.save([graph], graph_dir / "train.pt")

    result = load_run_split(tmp_path, "run-a", "train")
    assert len(result) == 1
    assert result[0].labeled_edge_attr.shape == (1, 7)


def test_ablation_report_includes_existing_full_edge_sage_reference():
    ablation = pd.DataFrame(
        [{"model_id": "edge-sage-noedge", "target": "qos", "horizon": 1, "split": "test", "run_name": "r1"}]
    )
    benchmark = pd.DataFrame(
        [
            {"model_id": "edge-sage", "target": "qos", "horizon": 1, "split": "test", "run_name": "r1"},
            {"model_id": "logreg", "target": "qos", "horizon": 1, "split": "test", "run_name": "r1"},
        ]
    )

    result = with_full_ablation_reference(ablation, benchmark)
    assert set(result["model_id"]) == {"edge-sage", "edge-sage-noedge"}
