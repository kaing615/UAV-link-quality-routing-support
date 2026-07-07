from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_graphs(pt_path: Path) -> list[Data]:
    raw = torch.load(pt_path, weights_only=False)
    graphs = []
    for g in raw:
        data = Data(x=g['x'], edge_index=g['edge_index'], edge_attr=g['edge_attr'], edge_label_index=g['edge_label_index'], edge_label=g['edge_label'], labeled_edge_attr=g['edge_attr'][::2])
        graphs.append(data)
    return graphs

def make_loader(pt_path: Path, batch_size: int, shuffle: bool=False) -> DataLoader:
    return DataLoader(load_graphs(pt_path), batch_size=batch_size, shuffle=shuffle)

def compute_pos_weight(train_graphs: list[Data]) -> torch.Tensor:
    n0 = sum((int((g.edge_label == 0).sum()) for g in train_graphs))
    n1 = sum((int((g.edge_label == 1).sum()) for g in train_graphs))
    return torch.tensor([n0 / max(n1, 1)], dtype=torch.float32)

def collect_scores(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_scores = ([], [])
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.edge_label_index, batch.labeled_edge_attr)
            all_labels.append(batch.edge_label.cpu())
            all_scores.append(torch.sigmoid(logits).cpu())
    y_true = torch.cat(all_labels).numpy()
    y_score = torch.cat(all_scores).numpy()
    return (y_true, y_score)

def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray, lo: float=0.3, hi: float=0.7, min_gain: float=0.02) -> tuple[float, float]:

    def macro_f1_at(t: float) -> float:
        preds = (y_score >= t).astype(int)
        return float(f1_score(y_true, preds, labels=[0, 1], average='macro', zero_division=0))
    f1_default = macro_f1_at(0.5)
    best_t, best_f1 = (0.5, f1_default)
    for t in np.arange(lo, hi + 1e-09, 0.01):
        f1 = macro_f1_at(t)
        if f1 > best_f1:
            best_t, best_f1 = (float(t), f1)
    if best_f1 - f1_default < min_gain:
        return (0.5, f1_default)
    return (best_t, best_f1)

def evaluate_split(model: torch.nn.Module, loader: DataLoader, device: torch.device, model_id: str, model_name: str, split_name: str, threshold: float=0.5) -> tuple[dict, pd.DataFrame]:
    t0 = time.perf_counter()
    y_true, y_score = collect_scores(model, loader, device)
    inference_s = time.perf_counter() - t0
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    has_both = set(y_true.tolist()) == {0, 1}
    roc_auc = float(roc_auc_score(y_true, y_score)) if has_both else None
    pr_auc = float(average_precision_score(y_true, y_score)) if has_both else None
    metrics = {'model_id': model_id, 'model_name': model_name, 'split': split_name, 'threshold': float(threshold), 'n_samples': int(len(y_true)), 'has_both_classes': bool(set(y_true.tolist()) == {0, 1}), 'positive_ratio': float(y_true.mean()), 'accuracy': float(accuracy_score(y_true, y_pred)), 'precision': float(precision_score(y_true, y_pred, zero_division=0)), 'recall': float(recall_score(y_true, y_pred, zero_division=0)), 'f1': float(f1_score(y_true, y_pred, zero_division=0)), 'macro_f1': float(f1_score(y_true, y_pred, labels=[0, 1], average='macro', zero_division=0)), 'roc_auc': roc_auc, 'pr_auc': pr_auc, 'inference_time_ms': float(inference_s * 1000), 'inference_ms_per_sample': float(inference_s * 1000 / max(len(y_true), 1)), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    pred_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'pred_score': y_score})
    return (metrics, pred_df)