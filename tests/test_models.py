"""Test GNN model forward pass shapes and output validity."""

import pytest
import torch

from src.models.gnn.edge_gnn import (
    EdgeAwareSAGEEdgeClassifier,
    GATEdgeClassifier,
    GraphSAGEEdgeClassifier,
)


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
