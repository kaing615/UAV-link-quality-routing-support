---
name: project-gnn-status
description: GNN implementation status, architecture decisions, and current results vs baselines
metadata:
  type: project
---

GNN edge classifier implemented and running for UAV link stability prediction.

**Why:** Main thesis contribution — use GNN to predict stable/unstable links in UAV mesh network.

**How to apply:** Use these results when discussing model comparison and next improvement steps.

## Files created
- `src/models/gnn/edge_gnn.py` — GraphSAGEEdgeClassifier + GATEdgeClassifier
- `src/training/gnn/common.py` — load_graphs, make_loader, evaluate_split
- `src/training/gnn/train_gnn.py` — training loop, --model graphsage|gat

## Architecture
- GraphSAGE: BatchNorm(node_feats) → 2× SAGEConv(64) → edge decoder: concat(h_u, h_v, BatchNorm(edge_feats)) → MLP(133→64→32→1)
- BatchNorm is CRITICAL — node coords (0-1000m) and RSSI (-90dB) are very different scales
- Edge features for classification taken as edge_attr[::2] (forward direction, same order as edge_label_index)

## Key numbers (test macro_f1, valid runs only)
- GraphSAGE: **0.7389** mean (8 valid runs)
- XGBoost baseline: **0.8622** mean (8 valid runs)
- Gap: ~0.12 macro_f1

On well-balanced runs (05,06,07,08,09): GNN gets ~0.83, XGBoost ~0.91.

## Known issues / next improvements
1. GNN uses only 5 edge features (distance, rssi, delay, packet_loss, relative_speed).
   Baselines use 7 (also snr, throughput). `snr` is in edges_features.csv but missing from build_graph_dataset.py.
   → Fix: add `snr` to EDGE_FEATURES in build_graph_dataset.py → regenerate .pt files.
2. Degenerate runs: 03 (val has 0% label=1), 04 (val has 100% label=1), 02 (99% label=1 everywhere).
   These hurt GNN more than sklearn models due to BatchNorm behavior with 1 class.
3. GAT not yet tested (code ready, just run --model gat).

## Run command
```bash
python3 -m src.training.gnn.train_gnn \
  --run-name <batch_name> --model graphsage \
  --epochs 200 --patience 25 --hidden 64 --num-layers 2
```
