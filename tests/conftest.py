"""Shared pytest fixtures for UAV-GNN tests."""
import pytest
import torch


@pytest.fixture
def sample_graph():
    """Create a sample graph dataset format for testing."""
    n_nodes = 5
    n_edges = 8
    return {
        "x": torch.randn(n_nodes, 8),
        "edge_index": torch.randint(0, n_nodes, (2, n_edges * 2)),
        "edge_attr": torch.randn(n_edges * 2, 7),
        "edge_label_index": torch.randint(0, n_nodes, (2, n_edges)),
        "edge_label": torch.randint(0, 2, (n_edges,)),
    }

@pytest.fixture
def sample_labeled_edge_attr(sample_graph):
    return sample_graph["edge_attr"][::2]
