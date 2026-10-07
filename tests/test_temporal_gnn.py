from __future__ import annotations

import copy
import json

import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from src.models.gnn.edge_gnn import EdgeAwareSAGEGRUEdgeClassifier
from src.training.gnn.temporal_data import build_temporal_windows
from src.training.gnn.train_temporal_gnn import main as train_temporal_main


def _record(time: int | None, node_ids: list[int] | None = None) -> dict:
    record = {"node_ids": node_ids or [10, 20, 30], "x": torch.randn(3, 8)}
    if time is not None:
        record["time"] = time
    return record


def _write_graph_splits(tmp_path, records_by_split: dict[str, list[dict]]) -> None:
    graph_dir = tmp_path / "graph_dataset"
    graph_dir.mkdir()
    for split in ("train", "val", "test", "purged"):
        torch.save(records_by_split.get(split, []), graph_dir / f"{split}.pt")


def test_build_temporal_windows_orders_history_across_purged_records(tmp_path):
    _write_graph_splits(
        tmp_path,
        {
            "train": [_record(0), _record(1)],
            "purged": [_record(2)],
            "val": [_record(3), _record(4)],
        },
    )

    windows = build_temporal_windows(tmp_path, "val")

    assert [[record["time"] for record in window.history] for window in windows] == [[0, 1, 2], [1, 2, 3]]
    assert [window.endpoint["time"] for window in windows] == [3, 4]
    assert {window.split for window in windows} == {"val"}


def test_build_temporal_windows_rejects_a_record_without_time(tmp_path):
    _write_graph_splits(
        tmp_path,
        {"train": [_record(0), _record(None), _record(2)], "val": [_record(3)]},
    )

    with pytest.raises(ValueError, match="time"):
        build_temporal_windows(tmp_path, "val")


def test_build_temporal_windows_rejects_a_missing_snapshot_time(tmp_path):
    _write_graph_splits(
        tmp_path,
        {"train": [_record(0), _record(1), _record(2)], "val": [_record(4)]},
    )

    with pytest.raises(ValueError, match="consecutive"):
        build_temporal_windows(tmp_path, "val")


def test_build_temporal_windows_rejects_node_id_mismatch(tmp_path):
    _write_graph_splits(
        tmp_path,
        {
            "train": [_record(0), _record(1), _record(2, [10, 30, 20])],
            "val": [_record(3)],
        },
    )

    with pytest.raises(ValueError, match="node_ids"):
        build_temporal_windows(tmp_path, "val")


def _model_graph() -> Data:
    return Data(
        x=torch.randn(3, 8),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.randn(4, 7),
        edge_label_index=torch.tensor([[0, 1], [1, 2]]),
        labeled_edge_attr=torch.randn(2, 7),
        edge_label=torch.tensor([0, 1]),
    )


def test_edge_aware_sage_gru_returns_finite_edge_logits_and_gradients():
    model = EdgeAwareSAGEGRUEdgeClassifier(hidden_channels=16, dropout=0)
    graphs = [_model_graph() for _ in range(4)]

    logits = model.forward_window(graphs)
    logits.sum().backward()

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    assert model.gru.weight_ih_l0.grad is not None


def test_edge_aware_sage_gru_ignores_historical_labels():
    model = EdgeAwareSAGEGRUEdgeClassifier(hidden_channels=16, dropout=0).eval()
    graphs = [_model_graph() for _ in range(4)]
    expected = model.forward_window(graphs)
    changed_labels = copy.deepcopy(graphs)
    for graph in changed_labels[:-1]:
        graph.edge_label = 1 - graph.edge_label

    actual = model.forward_window(changed_labels)

    torch.testing.assert_close(actual, expected)


def _trainer_record(time: int) -> dict:
    label = torch.tensor([time % 2, 1 - time % 2])
    return {
        "time": time,
        "node_ids": [10, 20, 30],
        "x": torch.tensor(
            [[float(time), 0, 0, 0, 0, 0, 1, 1], [0, float(time), 0, 0, 0, 0, 1, 1], [0, 0, float(time), 0, 0, 0, 1, 1]]
        ),
        "edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        "edge_attr": torch.full((4, 7), float(time + 1)),
        "edge_label_index": torch.tensor([[0, 1], [1, 2]]),
        "edge_label": label,
    }


def test_temporal_trainer_writes_auditable_endpoint_artifacts(tmp_path, monkeypatch):
    run_root = tmp_path / "data" / "synthetic"
    run_root.mkdir(parents=True)
    _write_graph_splits(
        run_root,
        {
            "train": [_trainer_record(time) for time in range(5)],
            "val": [_trainer_record(time) for time in range(5, 7)],
            "test": [_trainer_record(time) for time in range(7, 9)],
        },
    )
    output_dir = tmp_path / "output"
    args = [
        "train_temporal_gnn.py",
        "--run-name",
        "synthetic",
        "--data-root",
        str(tmp_path / "data"),
        "--output-dir",
        str(output_dir),
        "--epochs",
        "2",
        "--patience",
        "2",
        "--hidden",
        "8",
        "--num-layers",
        "1",
        "--dropout",
        "0",
        "--seed",
        "7",
    ]
    monkeypatch.setattr(
        "sys.argv",
        args,
    )

    train_temporal_main()

    assert {path.name for path in output_dir.iterdir()} >= {
        "best_model.pt",
        "metadata.json",
        "metrics.csv",
        "val_predictions.csv",
        "test_predictions.csv",
    }
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_id"] == "edge-sage-gru"
    assert metadata["window_size"] == 4
    assert metadata["feature_flags"]["historical_labels_used"] is False
    assert metadata["resume_marker"] == {"seed": 7, "window_size": 4}

    for split, times in (("val", [5, 6]), ("test", [7, 8])):
        predictions = pd.read_csv(output_dir / f"{split}_predictions.csv")
        assert list(predictions.columns[:3]) == ["time", "src", "dst"]
        assert predictions["time"].tolist() == [time for time in times for _ in range(2)]
        assert predictions[["src", "dst"]].values.tolist() == [[10, 20], [20, 30]] * len(times)

    metrics = pd.read_csv(output_dir / "metrics.csv")
    assert metrics["split"].tolist() == ["val", "test"]

    resumed_output_dir = tmp_path / "resumed-output"
    args[args.index(str(output_dir))] = str(resumed_output_dir)
    monkeypatch.setattr("sys.argv", args)
    train_temporal_main()

    metadata_fields = (
        "model_id",
        "model_name",
        "run_name",
        "window_size",
        "hidden_channels",
        "num_layers",
        "dropout",
        "lr",
        "weight_decay",
        "batch_size",
        "seed",
        "threshold",
        "best_epoch",
        "best_val_macro_f1",
        "feature_flags",
        "resume_marker",
    )
    resumed_metadata = json.loads((resumed_output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert {field: metadata[field] for field in metadata_fields} == {
        field: resumed_metadata[field] for field in metadata_fields
    }
    for split in ("val", "test"):
        pd.testing.assert_frame_equal(
            pd.read_csv(output_dir / f"{split}_predictions.csv"),
            pd.read_csv(resumed_output_dir / f"{split}_predictions.csv"),
        )
    checkpoint = torch.load(output_dir / "best_model.pt", weights_only=True)
    resumed_checkpoint = torch.load(resumed_output_dir / "best_model.pt", weights_only=True)
    assert checkpoint.keys() == resumed_checkpoint.keys()
    for name, tensor in checkpoint.items():
        torch.testing.assert_close(tensor, resumed_checkpoint[name], rtol=0, atol=0)
