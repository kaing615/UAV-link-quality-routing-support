from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv


class GraphSAGEEdgeClassifier(nn.Module):
    """
    GraphSAGE encoder + MLP edge classifier for link stability prediction.

    Forward pass:
      1. BatchNorm normalizes raw node & edge features (coordinates, RSSI, etc. have very different scales)
      2. SAGEConv layers aggregate neighbor info → node embeddings h [N, hidden]
      3. For each labeled edge (u,v): concat(h[u], h[v], edge_features) → MLP → logit
    """

    def __init__(
        self,
        node_in_channels: int = 8,
        edge_in_channels: int = 5,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        self.node_bn = nn.BatchNorm1d(node_in_channels)
        self.edge_bn = nn.BatchNorm1d(edge_in_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        in_ch = node_in_channels
        for i in range(num_layers):
            self.convs.append(SAGEConv(in_ch, hidden_channels))
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            in_ch = hidden_channels

        mlp_in = hidden_channels * 2 + edge_in_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.node_bn(x)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = self.bns[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def decode(
        self,
        h: torch.Tensor,
        edge_label_index: torch.Tensor,
        labeled_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_label_index
        e = self.edge_bn(labeled_edge_attr)
        edge_repr = torch.cat([h[src], h[dst], e], dim=1)
        return self.edge_mlp(edge_repr).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        labeled_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode(x, edge_index)
        return self.decode(h, edge_label_index, labeled_edge_attr)


class GATEdgeClassifier(nn.Module):
    """
    GAT encoder + MLP edge classifier.

    Same interface as GraphSAGEEdgeClassifier — swap by changing the model class.
    """

    def __init__(
        self,
        node_in_channels: int = 8,
        edge_in_channels: int = 5,
        hidden_channels: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        self.node_bn = nn.BatchNorm1d(node_in_channels)
        self.edge_bn = nn.BatchNorm1d(edge_in_channels)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        in_ch = node_in_channels
        for i in range(num_layers):
            is_last = i == num_layers - 1
            out_ch = hidden_channels // heads if not is_last else hidden_channels
            n_heads = heads if not is_last else 1
            self.convs.append(GATConv(in_ch, out_ch, heads=n_heads, dropout=dropout, concat=not is_last))
            if not is_last:
                self.bns.append(nn.BatchNorm1d(out_ch * n_heads))
            in_ch = out_ch * n_heads if not is_last else hidden_channels

        mlp_in = hidden_channels * 2 + edge_in_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.node_bn(x)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = self.bns[i](h)
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def decode(
        self,
        h: torch.Tensor,
        edge_label_index: torch.Tensor,
        labeled_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_label_index
        e = self.edge_bn(labeled_edge_attr)
        edge_repr = torch.cat([h[src], h[dst], e], dim=1)
        return self.edge_mlp(edge_repr).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor,
        labeled_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode(x, edge_index)
        return self.decode(h, edge_label_index, labeled_edge_attr)
