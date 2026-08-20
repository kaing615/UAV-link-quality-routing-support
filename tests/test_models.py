"""Test GNN model forward pass shapes and output validity."""

import pytest
import torch

import src.training.gnn.train_gnn as train_gnn
from src.models.gnn.edge_gnn import (
    EdgeAwareSAGEEdgeClassifier,
    GATEdgeClassifier,
    GraphSAGEEdgeClassifier,
)
from src.training.gnn.train_gnn import resolve_edge_mode


@pytest.mark.parametrize(
    "model_cls,kwargs",
    [
        (GraphSAGEEdgeClassifier, {}),
        (GATEdgeClassifier, {}),
        (EdgeAwareSAGEEdgeClassifier, {}),
    ],
)
def test_forward_output_shape(model_cls, kwargs, sample_graph, sample_labeled_edge_attr):
    model = model_cls(node_in_channels=8, edge_in_channels=7, hidden_channels=32, **kwargs)
    model.eval()
    g = sample_graph
    with torch.no_grad():
        logits = model(g["x"], g["edge_index"], g["edge_attr"], g["edge_label_index"], sample_labeled_edge_attr)
    n_query_edges = g["edge_label_index"].shape[1]
    assert logits.shape == (n_query_edges,)
    assert not torch.isnan(logits).any()


@pytest.mark.parametrize(
    "model_cls",
    [
        GraphSAGEEdgeClassifier,
        GATEdgeClassifier,
        EdgeAwareSAGEEdgeClassifier,
    ],
)
def test_noedge_ablation(model_cls, sample_graph, sample_labeled_edge_attr):
    model = model_cls(node_in_channels=8, edge_in_channels=7, hidden_channels=32, use_edge_features=False)
    model.eval()
    g = sample_graph
    with torch.no_grad():
        logits = model(g["x"], g["edge_index"], g["edge_attr"], g["edge_label_index"], sample_labeled_edge_attr)
    assert logits.shape == (g["edge_label_index"].shape[1],)


@pytest.mark.parametrize(
    "message_edges,decoder_edges,expected_message_dim,expected_decoder_width",
    [
        (True, True, 7, 71),
        (False, True, 0, 71),
        (True, False, 7, 64),
        (False, False, 0, 64),
    ],
)
def test_edge_sage_supports_separate_message_and_decoder_ablation(
    message_edges,
    decoder_edges,
    expected_message_dim,
    expected_decoder_width,
    sample_graph,
    sample_labeled_edge_attr,
):
    model = EdgeAwareSAGEEdgeClassifier(
        node_in_channels=8,
        edge_in_channels=7,
        hidden_channels=32,
        use_message_edge_features=message_edges,
        use_decoder_edge_features=decoder_edges,
    )
    model.eval()
    with torch.no_grad():
        logits = model(
            sample_graph["x"],
            sample_graph["edge_index"],
            sample_graph["edge_attr"],
            sample_graph["edge_label_index"],
            sample_labeled_edge_attr,
        )

    assert model.convs[0].edge_dim == expected_message_dim
    assert model.edge_mlp[0].in_features == expected_decoder_width
    assert logits.shape == (sample_graph["edge_label_index"].shape[1],)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("full", (True, True, "edge-sage")),
        ("decoder-only", (False, True, "edge-sage-decoder-only")),
        ("message-only", (True, False, "edge-sage-message-only")),
        ("noedge", (False, False, "edge-sage-noedge")),
    ],
)
def test_resolve_edge_mode_maps_ablation_to_model_identity(mode, expected):
    assert resolve_edge_mode(mode) == expected


def test_dvclive_tracking_does_not_modify_pipeline_file(tmp_path, monkeypatch):
    pipeline = tmp_path / "dvc.yaml"
    original = "stages:\n  smoke:\n    cmd: echo ok\n"
    pipeline.write_text(original, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    make_live = getattr(train_gnn, "make_live", None)
    assert make_live is not None, "training must expose the safe DVCLive factory"

    with make_live(tmp_path / "outputs" / "dvclive") as live:
        live.log_metric("loss", 1.0)
        live.next_step()

    assert pipeline.read_text(encoding="utf-8") == original
