from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from src.models.gnn.edge_gnn import GATEdgeClassifier, GraphSAGEEdgeClassifier
from src.training.gnn.common import compute_pos_weight, evaluate_split, load_graphs, make_loader

_MODELS = {
    "graphsage": (GraphSAGEEdgeClassifier, "GraphSAGE Edge Classifier"),
    "gat": (GATEdgeClassifier, "GAT Edge Classifier"),
}

NODE_IN = 8
EDGE_IN = 7


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_label_index, batch.labeled_edge_attr)
        loss = criterion(logits, batch.edge_label.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a GNN edge classifier on UAV link stability data.")
    p.add_argument("--run-name", type=str, required=True, help="Batch run name under data/graph_dataset/")
    p.add_argument("--model", type=str, default="graphsage", choices=list(_MODELS.keys()))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    model_cls, model_name = _MODELS[args.model]
    model_id = args.model

    run_root = Path("data/graph_dataset") / args.run_name / "graph_dataset"
    output_dir = args.output_dir or (Path("outputs/gnn") / model_id / args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[RUN] model={model_id} | run={args.run_name} | device={device}")
    print(f"      hidden={args.hidden} | layers={args.num_layers} | dropout={args.dropout}")
    print(f"      lr={args.lr} | epochs={args.epochs} | patience={args.patience}")

    train_graphs = load_graphs(run_root / "train.pt")
    train_loader = make_loader(run_root / "train.pt", batch_size=args.batch_size, shuffle=True)
    val_loader   = make_loader(run_root / "val.pt",   batch_size=args.batch_size, shuffle=False)
    test_loader  = make_loader(run_root / "test.pt",  batch_size=args.batch_size, shuffle=False)

    pos_weight = compute_pos_weight(train_graphs).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = model_cls(
        node_in_channels=NODE_IN,
        edge_in_channels=EDGE_IN,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _ = evaluate_split(model, val_loader, device, model_id, model_name, "val")
        val_f1 = val_metrics["macro_f1"]

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  ep {epoch:3d} | loss={train_loss:.4f} | val_macro_f1={val_f1:.4f}"
                f" | best={best_macro_f1:.4f} @ ep{best_epoch}"
            )

        if patience_counter >= args.patience:
            print(f"[STOP] early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print(f"\n[BEST] epoch={best_epoch} | val_macro_f1={best_macro_f1:.4f}")

    model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
    val_metrics,  val_preds  = evaluate_split(model, val_loader,  device, model_id, model_name, "val")
    test_metrics, test_preds = evaluate_split(model, test_loader, device, model_id, model_name, "test")

    pd.DataFrame([val_metrics, test_metrics]).to_csv(output_dir / "metrics.csv", index=False)
    val_preds.to_csv(output_dir / "val_predictions.csv",   index=False)
    test_preds.to_csv(output_dir / "test_predictions.csv", index=False)

    metadata = {
        "model_id": model_id,
        "model_name": model_name,
        "run_name": args.run_name,
        "hidden_channels": args.hidden,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[OK]  val : macro_f1={val_metrics['macro_f1']:.4f}  f1={val_metrics['f1']:.4f}"
          f"  recall={val_metrics['recall']:.4f}")
    print(f"[OK]  test: macro_f1={test_metrics['macro_f1']:.4f}  f1={test_metrics['f1']:.4f}"
          f"  recall={test_metrics['recall']:.4f}")
    print(f"      outputs → {output_dir}")


if __name__ == "__main__":
    main()
