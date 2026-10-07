from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric.data import Data

from src.models.gnn.edge_gnn import EdgeAwareSAGEGRUEdgeClassifier
from src.training.gnn.common import compute_pos_weight, find_best_threshold
from src.training.gnn.temporal_data import TemporalWindow, build_temporal_windows
from src.training.gnn.train_gnn import EDGE_IN, NODE_IN, make_live

MODEL_ID = "edge-sage-gru"
MODEL_NAME = "Edge-Aware GraphSAGE-GRU Edge Classifier"


def _graph(record: dict, endpoint: bool, device: torch.device) -> Data:
    graph = Data(x=record["x"], edge_index=record["edge_index"], edge_attr=record["edge_attr"])
    if endpoint:
        graph.edge_label_index = record["edge_label_index"]
        graph.labeled_edge_attr = record["edge_attr"][::2]
        graph.edge_label = record["edge_label"]
    return graph.to(device)


def _window_graphs(window: TemporalWindow, device: torch.device) -> list[Data]:
    return [_graph(record, endpoint=False, device=device) for record in window.history] + [
        _graph(window.endpoint, endpoint=True, device=device)
    ]


def _endpoint_graphs(windows: list[TemporalWindow]) -> list[Data]:
    return [_graph(window.endpoint, endpoint=True, device=torch.device("cpu")) for window in windows]


def train_one_epoch(
    model: EdgeAwareSAGEGRUEdgeClassifier,
    windows: list[TemporalWindow],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    model.train()
    total_loss = 0.0
    for index in torch.randperm(len(windows), generator=generator).tolist():
        graphs = _window_graphs(windows[index], device)
        endpoint = graphs[-1]
        optimizer.zero_grad()
        logits = model.forward_window(graphs)
        loss = criterion(logits, endpoint.edge_label.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(windows)


def collect_scores(
    model: EdgeAwareSAGEGRUEdgeClassifier, windows: list[TemporalWindow], device: torch.device
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    model.eval()
    labels, scores, rows = [], [], []
    with torch.no_grad():
        for window in windows:
            graphs = _window_graphs(window, device)
            endpoint = graphs[-1]
            logits = model.forward_window(graphs)
            labels.extend(endpoint.edge_label.cpu().tolist())
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            edge_label_index = window.endpoint["edge_label_index"]
            node_ids = window.endpoint["node_ids"]
            rows.extend(
                {
                    "time": int(window.endpoint["time"]),
                    "src": int(node_ids[edge_label_index[0, index]]),
                    "dst": int(node_ids[edge_label_index[1, index]]),
                }
                for index in range(edge_label_index.shape[1])
            )
    return np.asarray(labels), np.asarray(scores), pd.DataFrame(rows)


def evaluate_windows(
    model: EdgeAwareSAGEGRUEdgeClassifier,
    windows: list[TemporalWindow],
    device: torch.device,
    split_name: str,
    threshold: float = 0.5,
) -> tuple[dict, pd.DataFrame]:
    t0 = time.perf_counter()
    y_true, y_score, identities = collect_scores(model, windows, device)
    inference_s = time.perf_counter() - t0
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both = set(y_true.tolist()) == {0, 1}
    metrics = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "split": split_name,
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "has_both_classes": has_both,
        "positive_ratio": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both else None,
        "pr_auc": float(average_precision_score(y_true, y_score)) if has_both else None,
        "inference_time_ms": float(inference_s * 1000),
        "inference_ms_per_sample": float(inference_s * 1000 / max(len(y_true), 1)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    predictions = identities.assign(y_true=y_true, y_pred=y_pred, pred_score=y_score)
    return metrics, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an endpoint-only temporal Edge-SAGE-GRU classifier.")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/graph_dataset"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-tune-threshold", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("Temporal windows require --batch-size 1")
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    run_root = args.data_root / args.run_name
    output_dir = args.output_dir or (Path("outputs/gnn") / MODEL_ID / args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_windows = build_temporal_windows(run_root, "train", args.window_size)
    val_windows = build_temporal_windows(run_root, "val", args.window_size)
    test_windows = build_temporal_windows(run_root, "test", args.window_size)
    if not train_windows or not val_windows or not test_windows:
        raise ValueError("Each train, val, and test split needs at least one complete temporal window")

    model = EdgeAwareSAGEGRUEdgeClassifier(
        node_in_channels=NODE_IN,
        edge_in_channels=EDGE_IN,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(_endpoint_graphs(train_windows)).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    with make_live(output_dir / "dvclive") as live:
        for name, value in {
            "model_id": MODEL_ID,
            "run_name": args.run_name,
            "window_size": args.window_size,
            "hidden": args.hidden,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "protocol": "within-run-endpoint-only",
        }.items():
            live.log_param(name, value)
        best_macro_f1, best_epoch, patience_counter = -1.0, 0, 0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_windows, optimizer, criterion, device, generator)
            val_metrics, _ = evaluate_windows(model, val_windows, device, "val")
            val_f1 = val_metrics["macro_f1"]
            live.log_metric("train/loss", train_loss)
            live.log_metric("val/macro_f1", val_f1)
            live.log_metric("val/f1", val_metrics["f1"])
            live.log_metric("val/recall", val_metrics["recall"])
            live.log_metric("val/precision", val_metrics["precision"])
            live.next_step()
            if val_f1 > best_macro_f1:
                best_macro_f1, best_epoch, patience_counter = val_f1, epoch, 0
                torch.save(model.state_dict(), output_dir / "best_model.pt")
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                break

        model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
        threshold = 0.5
        if not args.no_tune_threshold:
            val_true, val_score, _ = collect_scores(model, val_windows, device)
            threshold, _ = find_best_threshold(val_true, val_score)
        val_metrics, val_predictions = evaluate_windows(model, val_windows, device, "val", threshold)
        test_metrics, test_predictions = evaluate_windows(model, test_windows, device, "test", threshold)
        live.log_metric("test/macro_f1", test_metrics["macro_f1"])
        live.log_metric("test/f1", test_metrics["f1"])
        live.log_metric("test/recall", test_metrics["recall"])
        live.log_metric("test/precision", test_metrics["precision"])
        live.summary["best_epoch"] = best_epoch
        live.summary["threshold"] = threshold
        live.summary["best_val_macro_f1"] = best_macro_f1

    pd.DataFrame([val_metrics, test_metrics]).to_csv(output_dir / "metrics.csv", index=False)
    val_predictions.to_csv(output_dir / "val_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    metadata = {
        "model_id": MODEL_ID,
        "model_name": MODEL_NAME,
        "run_name": args.run_name,
        "window_size": args.window_size,
        "hidden_channels": args.hidden,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "threshold": threshold,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "feature_flags": {
            "use_edge_features": True,
            "use_message_edge_features": True,
            "use_decoder_edge_features": True,
            "historical_labels_used": False,
        },
        "resume_marker": {"seed": args.seed, "window_size": args.window_size},
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
