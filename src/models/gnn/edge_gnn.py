from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing, SAGEConv


class GraphSAGEEdgeClassifier(nn.Module):

    def __init__(self, node_in_channels: int=8, edge_in_channels: int=5, hidden_channels: int=64, num_layers: int=2, dropout: float=0.3, use_edge_features: bool=True):
        super().__init__()
        self.dropout = dropout
        self.use_edge_features = use_edge_features
        self.node_bn = nn.BatchNorm1d(node_in_channels)
        self.edge_bn = nn.BatchNorm1d(edge_in_channels) if use_edge_features else None
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        in_ch = node_in_channels
        for i in range(num_layers):
            self.convs.append(SAGEConv(in_ch, hidden_channels))
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            in_ch = hidden_channels
        mlp_in = hidden_channels * 2 + (edge_in_channels if use_edge_features else 0)
        self.edge_mlp = nn.Sequential(nn.Linear(mlp_in, hidden_channels), nn.BatchNorm1d(hidden_channels), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_channels, 32), nn.ReLU(), nn.Linear(32, 1))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.node_bn(x)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = self.bns[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def decode(self, h: torch.Tensor, edge_label_index: torch.Tensor, labeled_edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_label_index
        if self.use_edge_features:
            e = self.edge_bn(labeled_edge_attr)
            edge_repr = torch.cat([h[src], h[dst], e], dim=1)
        else:
            edge_repr = torch.cat([h[src], h[dst]], dim=1)
        return self.edge_mlp(edge_repr).squeeze(-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, edge_label_index: torch.Tensor, labeled_edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, edge_index)
        return self.decode(h, edge_label_index, labeled_edge_attr)

class GATEdgeClassifier(nn.Module):

    def __init__(self, node_in_channels: int=8, edge_in_channels: int=5, hidden_channels: int=64, num_layers: int=2, heads: int=4, dropout: float=0.3, use_edge_features: bool=True):
        super().__init__()
        self.dropout = dropout
        self.use_edge_features = use_edge_features
        self.node_bn = nn.BatchNorm1d(node_in_channels)
        self.edge_bn = nn.BatchNorm1d(edge_in_channels) if use_edge_features else None
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
        mlp_in = hidden_channels * 2 + (edge_in_channels if use_edge_features else 0)
        self.edge_mlp = nn.Sequential(nn.Linear(mlp_in, hidden_channels), nn.BatchNorm1d(hidden_channels), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_channels, 32), nn.ReLU(), nn.Linear(32, 1))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.node_bn(x)
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = self.bns[i](h)
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def decode(self, h: torch.Tensor, edge_label_index: torch.Tensor, labeled_edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_label_index
        if self.use_edge_features:
            e = self.edge_bn(labeled_edge_attr)
            edge_repr = torch.cat([h[src], h[dst], e], dim=1)
        else:
            edge_repr = torch.cat([h[src], h[dst]], dim=1)
        return self.edge_mlp(edge_repr).squeeze(-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, edge_label_index: torch.Tensor, labeled_edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, edge_index)
        return self.decode(h, edge_label_index, labeled_edge_attr)

class EdgeAwareSAGEConv(MessagePassing):

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super().__init__(aggr='mean')
        self.edge_dim = edge_dim
        self.lin_msg = nn.Linear(in_channels + edge_dim, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None) -> torch.Tensor:
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.lin_self(x) + agg

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor | None) -> torch.Tensor:
        if self.edge_dim == 0:
            return self.lin_msg(x_j)
        return self.lin_msg(torch.cat([x_j, edge_attr], dim=-1))

class EdgeAwareSAGEEdgeClassifier(nn.Module):

    def __init__(self, node_in_channels: int=8, edge_in_channels: int=7, hidden_channels: int=128, num_layers: int=2, dropout: float=0.3, use_edge_features: bool=True):
        super().__init__()
        self.dropout = dropout
        self.use_edge_features = use_edge_features
        self.node_bn = nn.BatchNorm1d(node_in_channels)
        self.edge_mp_bn = nn.BatchNorm1d(edge_in_channels) if use_edge_features else None
        self.edge_clf_bn = nn.BatchNorm1d(edge_in_channels) if use_edge_features else None
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        in_ch = node_in_channels
        edge_dim = edge_in_channels if use_edge_features else 0
        for i in range(num_layers):
            self.convs.append(EdgeAwareSAGEConv(in_ch, hidden_channels, edge_dim=edge_dim))
            if i < num_layers - 1:
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            in_ch = hidden_channels
        mlp_in = hidden_channels * 2 + (edge_in_channels if use_edge_features else 0)
        self.edge_mlp = nn.Sequential(nn.Linear(mlp_in, hidden_channels), nn.BatchNorm1d(hidden_channels), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_channels, hidden_channels // 2), nn.ReLU(), nn.Linear(hidden_channels // 2, 1))

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.node_bn(x)
        ea = self.edge_mp_bn(edge_attr) if self.use_edge_features else None
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, ea)
            if i < len(self.convs) - 1:
                h = self.bns[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def decode(self, h: torch.Tensor, edge_label_index: torch.Tensor, labeled_edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_label_index
        if self.use_edge_features:
            e = self.edge_clf_bn(labeled_edge_attr)
            edge_repr = torch.cat([h[src], h[dst], e], dim=1)
        else:
            edge_repr = torch.cat([h[src], h[dst]], dim=1)
        return self.edge_mlp(edge_repr).squeeze(-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, edge_label_index: torch.Tensor, labeled_edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, edge_index, edge_attr)
        return self.decode(h, edge_label_index, labeled_edge_attr)
