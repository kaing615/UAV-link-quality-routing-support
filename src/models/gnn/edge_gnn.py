from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.nn import MessagePassing


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
        use_edge_features: bool = True,
    ):
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
        if self.use_edge_features:
            e = self.edge_bn(labeled_edge_attr)
            edge_repr = torch.cat([h[src], h[dst], e], dim=1)
        else:
            edge_repr = torch.cat([h[src], h[dst]], dim=1)
        return self.edge_mlp(edge_repr).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
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
        use_edge_features: bool = True,
    ):
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
        if self.use_edge_features:
            e = self.edge_bn(labeled_edge_attr)
            edge_repr = torch.cat([h[src], h[dst], e], dim=1)
        else:
            edge_repr = torch.cat([h[src], h[dst]], dim=1)
        return self.edge_mlp(edge_repr).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_label_index: torch.Tensor,
        labeled_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode(x, edge_index)
        return self.decode(h, edge_label_index, labeled_edge_attr)


# ---------------------------------------------------------------------------
# Edge-aware GraphSAGE
# ---------------------------------------------------------------------------

class EdgeAwareSAGEConv(MessagePassing):
    """
    GraphSAGE convolution that incorporates edge features into message passing.

    Standard SAGEConv aggregates only neighbor node embeddings:
        agg_i = mean { h_j | j ∈ N(i) }

    This variant concatenates the edge feature (RSSI, SNR, distance, …) with
    the neighbor embedding before projecting, so the aggregated signal reflects
    link quality, not just node identity:
        msg_{j→i} = W_msg · concat(h_j, e_{ij})
        agg_i     = mean { msg_{j→i} | j ∈ N(i) }
        h_i'      = W_self · h_i + agg_i + b
    """

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int):
        super().__init__(aggr="mean")
        # edge_dim=0 disables edge features in messages (ablation) → plain mean-SAGE
        self.edge_dim = edge_dim
        self.lin_msg = nn.Linear(in_channels + edge_dim, out_channels, bias=False)
        self.lin_self = nn.Linear(in_channels, out_channels, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.lin_self(x) + agg

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor | None) -> torch.Tensor:
        if self.edge_dim == 0:
            return self.lin_msg(x_j)
        return self.lin_msg(torch.cat([x_j, edge_attr], dim=-1))


class EdgeAwareSAGEEdgeClassifier(nn.Module):
    """
    Edge-aware GraphSAGE encoder + MLP edge classifier.

    Key difference from GraphSAGEEdgeClassifier: edge features (RSSI, SNR,
    distance, packet_loss, …) participate in the message-passing step, not
    only in the final decoder.  Each UAV node therefore learns an embedding
    that reflects the quality of its wireless links to neighbors — which is
    exactly the signal needed for routing decisions.

    Architecture:
      BN(node_feats) + BN(edge_feats_mp)
      → num_layers × EdgeAwareSAGEConv(hidden) with BN+ReLU+Dropout
      → decode: concat(h_u, h_v, BN(labeled_edge_feats)) → MLP → logit
    """

    def __init__(
        self,
        node_in_channels: int = 8,
        edge_in_channels: int = 7,
        hidden_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_edge_features: bool = True,
    ):
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
        self.edge_mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.node_bn(x)
        ea = self.edge_mp_bn(edge_attr) if self.use_edge_features else None
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index, ea)
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
        if self.use_edge_features:
            e = self.edge_clf_bn(labeled_edge_attr)
            edge_repr = torch.cat([h[src], h[dst], e], dim=1)
        else:
            edge_repr = torch.cat([h[src], h[dst]], dim=1)
        return self.edge_mlp(edge_repr).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_label_index: torch.Tensor,
        labeled_edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode(x, edge_index, edge_attr)
        return self.decode(h, edge_label_index, labeled_edge_attr)
