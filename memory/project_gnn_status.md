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

## Key numbers — CURRENT (test macro_f1, 4 balanced runs: 01,04,05,07)
Dataset: olsr_dataset_20260611_* with 7 edge features (distance, rssi, snr, delay, packet_loss, relative_speed, throughput)
- XGBoost:    **0.9096**
- MLP:        **0.8816**
- GAT:        **0.8584**  ← best GNN
- GraphSAGE:  **0.8046**

All 10 runs (including degenerate): XGBoost 0.7504, MLP 0.6487, GAT 0.6473, GraphSAGE 0.6194
Degenerate runs (positive_ratio > 0.95): 02,03,06,09,10 — filtered by aggregate_all_metrics.py --filter-balanced

## What's DONE
- 7 edge features in build_graph_dataset.py; .pt files regenerated with new olsr_dataset naming
- GraphSAGE + GAT both trained on all 10 runs
- Batch scripts: scripts/train/gnn/run_all_gnn_for_runs.sh
- Aggregate: src/evaluation/aggregate_all_metrics.py (supports --filter-balanced)
- Visualization: src/evaluation/plot_comparison.py → outputs/aggregates/all_models/model_comparison.png

## Edge-Aware GraphSAGE (added 2026-06-11)
- `EdgeAwareSAGEConv` in `edge_gnn.py` — custom MessagePassing layer that concatenates edge_attr with neighbor embedding before aggregation
- `EdgeAwareSAGEEdgeClassifier` — uses EdgeAwareSAGEConv, hidden=128 default, separate BN for MP edges vs classifier edges
- All models now have consistent 5-arg forward: `(x, edge_index, edge_attr, edge_label_index, labeled_edge_attr)`
- Training loop: grad clipping (max_norm=1.0), optional `--lr-scheduler` (ReduceLROnPlateau)
- Trained on all 10 runs. Balanced-runs (01,04,05,07) test macro_f1:
  XGB 0.9096 > MLP 0.8816 > GAT 0.8584 > **edge-sage 0.8468** > GraphSAGE 0.8046
- edge-sage beats base GraphSAGE by +0.042 but has high variance (std 0.079) and LOW recall (0.76)
  — run 01 hits 0.9821 but other runs drag the average down
- plot_comparison.py updated: dynamic bar positions, edge-sage green, legend above plot
- Chart: outputs/aggregates/all_models/model_comparison.png

## 4 improvements (2026-06-11, afternoon)

### 1. Threshold tuning (constrained)
- `find_best_threshold` in both `src/training/gnn/common.py` and `src/training/baselines/common.py`
- Unconstrained sweep [0.05, 0.95] OVERFITS the tiny val splits (val picks 0.05/0.14, test collapses):
  edge-sage dropped 0.8468 → 0.7015 in first attempt
- Final version: sweep restricted to [0.3, 0.7] + only accept if val gain ≥ 0.02; threshold stored
  in metrics.csv + metadata.json. `--no-tune-threshold` to disable for GNN.
- Net effect: MLP +0.004, XGB ≈0, GNNs ≈−0.01 (val/test temporal shift limits gains) — honest
  negative-ish result worth a thesis paragraph

### 2. Variance reduction — REFUTED
- hidden=64/dropout=0.4 made edge-sage much WORSE (0.70 ± 0.17 vs 0.84 ± 0.06): underfits small runs
- Kept hidden=128/dropout=0.3 (documented in run_edge_sage_for_runs.sh, env-overridable)

### 3. LORO cross-run generalization (NEW main thesis finding)
- `src/training/gnn/train_gnn_loro.py` + `src/training/baselines/loro_baselines.py` + `scripts/train/gnn/run_loro.sh`
- Protocol: train/val on 3 balanced runs, test = entire held-out 4th run; baselines use RAW
  features (per-run scalers would leak run identity); aggregate via `aggregate_all_metrics.py --loro`
- Test macro_f1: **XGB 0.934 ± 0.016** >> MLP 0.842 > GAT 0.783 ≈ GraphSAGE 0.776 > edge-sage 0.741
- Fold 01 (gauss-markov, trained on 3 rwp runs) is decisive: XGB 0.96 vs all neural ~0.45–0.61
  → XGB generalizes across mobility models, neural models do NOT. GNN does not beat XGB cross-run.

### 4. Ablation no-edge-features
- `use_edge_features=False` in all 3 GNN classes + `--no-edge-features` flag (model_id gets -noedge suffix)
- Within-run balanced: graphsage-noedge 0.628, edge-sage-noedge 0.612, gat-noedge 0.490
  vs 0.79–0.84 with edge features → edge features carry most of the signal; topology alone is weak.
  Explains why XGB (pure features) wins.

## Key numbers — FINAL (test macro_f1, 4 balanced runs, threshold-tuned)
Within-run: XGB 0.907 > MLP 0.885 > GAT 0.8375 ≈ edge-sage 0.8375 > GraphSAGE 0.793
LORO:       XGB 0.934 > MLP 0.842 > GAT 0.783 > GraphSAGE 0.776 > edge-sage 0.741
Charts: outputs/aggregates/all_models/model_comparison.png, outputs/aggregates/loro/loro_comparison.png

## Thesis narrative
Edge features dominate (ablation); within-run GNN ≈ MLP < XGB; cross-run XGB clearly wins
(robust to unseen mobility model). Honest framing: GNN adds structure but the link-quality
features already saturate the task; degenerate runs + tiny val splits limit deep models.

## ns-3 simulator (added 2026-06-11, evening) — user chose FULL ns-3 direction
- `simulation/ns3/uav-olsr-dataset.cc` + CMakeLists — real 802.11g ad-hoc + ns3::olsr,
  3D GaussMarkov/RandomWaypoint mobility, broadcast UDP probes (20/s) for measured
  delay/loss, PHY MonitorSnifferRx for measured RSSI, Nakagami fading (tuned m=3/1.5/1
  at 0.4/0.75×commRange — ns-3 defaults m1=m2=0.75 gave 79% unstable; no fading gave
  98% stable because RxSensitivity cutoff keeps all received SNR above tau_snr)
- Same CSV schema as Python sim → preprocessing/training unchanged
- Install: brew install ns-3 (3.48); broken .pc files → CMake links libs directly via glob
- Scripts: run_one_dataset_ns3.sh / run_many_random_datasets_ns3.sh (same interface)
- Validation run ns3_test_04_gm: 33.8% stable (balanced!), edge-sage 0.8145 vs XGB 0.8454
- ~145-step run takes ~2-3 min wall

## Remaining steps
- Generate full ns-3 batch (10+ runs) and rerun all experiments (within-run, LORO, ablation)
- Routing support module (src/routing is EMPTY — main thesis deliverable):
  map pred_score → edge weight, compare OLSR vs prediction-weighted OLSR on PDR/delay/stability
- More gauss-markov runs so LORO has >1 gm fold
- Try temporal GNN (snapshot sequence) — current models are static per-snapshot

## Run commands
```bash
# Base models
python3 -m src.training.gnn.train_gnn \
  --run-name <olsr_dataset_name> --model graphsage|gat \
  --epochs 200 --patience 25 --hidden 64 --num-layers 2

# Edge-aware GraphSAGE
python3 -m src.training.gnn.train_gnn \
  --run-name <olsr_dataset_name> --model edge-sage \
  --hidden 128 --num-layers 2 --dropout 0.3 \
  --lr 5e-4 --epochs 300 --patience 30 --lr-scheduler

# Ablation (no edge features)
python3 -m src.training.gnn.train_gnn --run-name <run> --model <m> --no-edge-features

# Batch all runs
bash scripts/train/gnn/run_edge_sage_for_runs.sh

# LORO (4 folds × 5 models) + aggregate
bash scripts/train/gnn/run_loro.sh
python3 -m src.evaluation.aggregate_all_metrics --loro

# Plots
python3 -m src.evaluation.plot_comparison
python3 -m src.evaluation.plot_comparison \
  --summary-csv outputs/aggregates/loro/summary_by_model_split.csv \
  --output-dir outputs/aggregates/loro --filename loro_comparison.png \
  --title "Cross-Run Generalization (LORO)"
```
