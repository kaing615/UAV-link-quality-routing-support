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

## Key numbers (test macro_f1)

Sau khi tích hợp đầy đủ 7 đặc trưng (bao gồm cả `snr` và `throughput`):

### 1. Tất cả 10 run (bao gồm cả các run dị biệt)
- GAT: **0.6473** F1-Macro mean
- GraphSAGE: **0.6194** F1-Macro mean
- XGBoost: **0.7504** F1-Macro mean
- MLP: **0.6487** F1-Macro mean

### 2. Chỉ tính trên các run cân bằng tốt (well-balanced: 01, 04, 05, 07)
- GAT: **0.8584** F1-Macro mean (Tiệm cận MLP là 0.8816 và XGBoost là 0.9095)
- GraphSAGE: **0.8045** F1-Macro mean
- XGBoost: **0.9095** F1-Macro mean
- MLP: **0.8816** F1-Macro mean

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
