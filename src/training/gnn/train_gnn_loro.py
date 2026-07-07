from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from dvclive import Live
from torch_geometric.loader import DataLoader

from src.training.gnn.common import collect_scores, compute_pos_weight, evaluate_split, find_best_threshold, load_graphs
from src.training.gnn.train_gnn import _MODELS, EDGE_IN, NODE_IN, train_one_epoch


def load_run_split(run_name: str, split: str) -> list:
    pt_path = Path("data/graph_dataset") / run_name / "graph_dataset" / f"{split}.pt"
    return load_graphs(pt_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leave-one-run-out GNN training.")
    p.add_argument("--test-run", type=str, required=True, help="Held-out run (entire run used as test set)")
    p.add_argument("--train-runs", type=str, required=True, help="Comma-separated training run names")
    p.add_argument("--model", type=str, default="graphsage", choices=list(_MODELS.keys()))
    p.add_argument("--lr-scheduler", action="store_true", default=False)
    p.add_argument("--no-tune-threshold", action="store_true", default=False)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0001)
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
    train_runs = [r.strip() for r in args.train_runs.split(",") if r.strip()]
    output_dir = args.output_dir or Path("outputs/loro") / model_id / args.test_run
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LORO] model={model_id} | test_run={args.test_run} | device={device}")
    print(f"       train_runs ({len(train_runs)}): {train_runs}")
    train_graphs, val_graphs = ([], [])
    for run in train_runs:
        train_graphs.extend(load_run_split(run, "train"))
        val_graphs.extend(load_run_split(run, "val"))
    test_graphs = [g for split in ("train", "val", "test") for g in load_run_split(args.test_run, split)]
    print(f"       graphs: train={len(train_graphs)} val={len(val_graphs)} test={len(test_graphs)}")
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
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
    scheduler = None
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-05
        )
    with Live(dir=str(output_dir / "dvclive"), report="auto") as live:
        live.log_param("model_id", model_id)
        live.log_param("model_name", model_name)
        live.log_param("protocol", "leave-one-run-out")
        live.log_param("test_run", args.test_run)
        live.log_param("n_train_runs", len(train_runs))
        live.log_param("hidden", args.hidden)
        live.log_param("num_layers", args.num_layers)
        live.log_param("dropout", args.dropout)
        live.log_param("lr", args.lr)
        live.log_param("weight_decay", args.weight_decay)
        live.log_param("lr_scheduler", args.lr_scheduler)
        live.log_param("batch_size", args.batch_size)
        live.log_param("seed", args.seed)
        best_macro_f1 = -1.0
        best_epoch = 0
        patience_counter = 0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics, _ = evaluate_split(model, val_loader, device, model_id, model_name, "val")
            val_f1 = val_metrics["macro_f1"]
            if scheduler is not None:
                scheduler.step(val_f1)
            live.log_metric("train/loss", train_loss)
            live.log_metric("val/macro_f1", val_f1)
            live.log_metric("val/f1", val_metrics["f1"])
            live.log_metric("val/recall", val_metrics["recall"])
            live.log_metric("learning_rate", optimizer.param_groups[0]["lr"])
            live.next_step()
            if val_f1 > best_macro_f1:
                best_macro_f1 = val_f1
                best_epoch = epoch
                torch.save(model.state_dict(), output_dir / "best_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1
            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"  ep {epoch:3d} | loss={train_loss:.4f} | val_macro_f1={val_f1:.4f} | best={best_macro_f1:.4f} @ ep{best_epoch}"
                )
            if patience_counter >= args.patience:
                print(f"[STOP] early stopping at epoch {epoch}")
                break
        model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))
        threshold = 0.5
        if not args.no_tune_threshold:
            val_true, val_score = collect_scores(model, val_loader, device)
            threshold, tuned_val_f1 = find_best_threshold(val_true, val_score)
            print(f"[THR] tuned threshold={threshold:.2f} (val macro_f1 {best_macro_f1:.4f} → {tuned_val_f1:.4f})")
        val_metrics, val_preds = evaluate_split(
            model, val_loader, device, model_id, model_name, "val", threshold=threshold
        )
        test_metrics, test_preds = evaluate_split(
            model, test_loader, device, model_id, model_name, "test", threshold=threshold
        )
        live.log_metric("test/macro_f1", test_metrics["macro_f1"])
        live.log_metric("test/f1", test_metrics["f1"])
        live.log_metric("test/recall", test_metrics["recall"])
        if test_metrics.get("roc_auc") is not None:
            live.log_metric("test/roc_auc", test_metrics["roc_auc"])
        live.summary["best_epoch"] = best_epoch
        live.summary["threshold"] = threshold
        live.summary["best_val_macro_f1"] = best_macro_f1
    pd.DataFrame([val_metrics, test_metrics]).to_csv(output_dir / "metrics.csv", index=False)
    val_preds.to_csv(output_dir / "val_predictions.csv", index=False)
    test_preds.to_csv(output_dir / "test_predictions.csv", index=False)
    metadata = {
        "model_id": model_id,
        "model_name": model_name,
        "protocol": "leave-one-run-out",
        "test_run": args.test_run,
        "train_runs": train_runs,
        "hidden_channels": args.hidden,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lr_scheduler": args.lr_scheduler,
        "threshold": threshold,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"[OK]  test ({args.test_run}): macro_f1={test_metrics['macro_f1']:.4f}  f1={test_metrics['f1']:.4f}  recall={test_metrics['recall']:.4f}"
    )
    print(f"      outputs → {output_dir}")


if __name__ == "__main__":
    main()
