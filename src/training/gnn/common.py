from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def load_graphs(pt_path: Path) -> list[Data]:
    """
    Load a .pt snapshot file and return a list of PyG Data objects.

    Each stored graph dict has:
      - edge_attr [2E, 5]: message-passing edges (undirected, duplicated)
      - edge_label_index [2, E]: edges to classify (original forward direction)
    edge_attr[::2] extracts the E original edge features in the same order
    as edge_label_index, which we store separately as labeled_edge_attr.
    """
    raw = torch.load(pt_path, weights_only=False)
    graphs = []
    for g in raw:
        data = Data(
            x=g["x"],
            edge_index=g["edge_index"],
            edge_attr=g["edge_attr"],
            edge_label_index=g["edge_label_index"],
            edge_label=g["edge_label"],
            labeled_edge_attr=g["edge_attr"][::2],
        )
        graphs.append(data)
    return graphs


def make_loader(pt_path: Path, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(load_graphs(pt_path), batch_size=batch_size, shuffle=shuffle)


def compute_pos_weight(train_graphs: list[Data]) -> torch.Tensor:
    """pos_weight = n_negative / n_positive for BCEWithLogitsLoss."""
    n0 = sum(int((g.edge_label == 0).sum()) for g in train_graphs)
    n1 = sum(int((g.edge_label == 1).sum()) for g in train_graphs)
    return torch.tensor([n0 / max(n1, 1)], dtype=torch.float32)


def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_id: str,
    model_name: str,
    split_name: str,
) -> tuple[dict, pd.DataFrame]:
    model.eval()
    all_labels, all_preds, all_scores = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_label_index,
                batch.labeled_edge_attr,
            )
            scores = torch.sigmoid(logits).cpu()
            preds = (scores >= 0.5).long()
            all_labels.append(batch.edge_label.cpu())
            all_preds.append(preds)
            all_scores.append(scores)

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_score = torch.cat(all_scores).numpy()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model_id": model_id,
        "model_name": model_name,
        "split": split_name,
        "n_samples": int(len(y_true)),
        "has_both_classes": bool(set(y_true.tolist()) == {0, 1}),
        "positive_ratio": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "pred_score": y_score,
    })

    return metrics, pred_df
