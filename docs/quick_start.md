# Quick Start

Tài liệu này gom các lệnh dùng hằng ngày cho pipeline hiện tại.

## 1. Kích hoạt môi trường

Nếu chưa activate virtualenv:

```bash
source simulation/.venv/bin/activate
```

## 2. Sinh một dataset

```bash
./scripts/dataset/run_one_dataset.sh <RUN_NAME> [SEED] [MOBILITY_MODEL]
```

Ví dụ:

```bash
./scripts/dataset/run_one_dataset.sh seed_42_rwp 42 random-waypoint
./scripts/dataset/run_one_dataset.sh seed_42_gm 42 gauss-markov
```

## 3. Sinh nhiều dataset ngẫu nhiên

```bash
./scripts/dataset/run_many_random_datasets.sh
./scripts/dataset/run_many_random_datasets.sh 10 exp01
```

## 3b. Sinh dataset bằng ns-3 (stack 802.11 + OLSR thật)

Yêu cầu `brew install ns-3` (binary tự build ở lần chạy đầu). Output cùng schema
với simulator Python nên mọi bước phía sau giữ nguyên. Chi tiết:
[simulation/ns3/README.md](../simulation/ns3/README.md).

```bash
./scripts/dataset/run_one_dataset_ns3.sh <RUN_NAME> [SEED] [MOBILITY]
./scripts/dataset/run_many_random_datasets_ns3.sh 10 ns3exp01
```

Vùng bay điều chỉnh qua `SIM_X_MAX`/`SIM_Y_MAX` (mặc định 500×500m). Script
batch random 10–30 UAV và tự scale vùng bay theo `sqrt(num_uavs/8)` để giữ
mật độ kết nối ổn định (tránh dataset degenerate khi tăng số node).

## 3c. Xóa toàn bộ data và output cũ

```bash
./scripts/utils/clean_data_outputs.sh             # hỏi xác nhận trước khi xóa
./scripts/utils/clean_data_outputs.sh 'ns3exp_*'  # chỉ xóa run khớp pattern
./scripts/utils/clean_data_outputs.sh --force     # không hỏi
```

## 4. Liệt kê các run đã có

```bash
python3 -m src.utils.list_run_names
python3 -m src.utils.list_run_names 'batch_*'
./scripts/utils/list_run_names.sh 'batch_*'
```

## 5. Train một model trên một run

### MLP (Baseline)

```bash
python3 -m src.training.baselines.mlp_baseline --run-name <RUN_NAME>
```

### XGBoost (Baseline)

```bash
python3 -m src.training.baselines.xgb_baseline --run-name <RUN_NAME>
```

### Baseline bổ sung: RSSI/SNR Threshold, Logistic Regression, Random Forest

```bash
python3 -m src.training.baselines.RSSI_SNR_Baseline --run-name <RUN_NAME>
python3 -m src.training.baselines.Logistic_Regression_Baseline --run-name <RUN_NAME>
python3 -m src.training.baselines.Random_Forest_Baseline --run-name <RUN_NAME>
```

Chạy cho tất cả run bằng vòng lặp:

```bash
for d in data/graph_dataset/<PATTERN>; do
  r=$(basename "$d")
  python3 -m src.training.baselines.RSSI_SNR_Baseline --run-name "$r"
  python3 -m src.training.baselines.Logistic_Regression_Baseline --run-name "$r"
  python3 -m src.training.baselines.Random_Forest_Baseline --run-name "$r"
done
```

### GNN (GraphSAGE / GAT)

```bash
python3 -m src.training.gnn.train_gnn --run-name <RUN_NAME> --model graphsage
python3 -m src.training.gnn.train_gnn --run-name <RUN_NAME> --model gat
```

### Edge-Aware GraphSAGE (mô hình đề xuất)

```bash
python3 -m src.training.gnn.train_gnn --run-name <RUN_NAME> --model edge-sage \
  --hidden 128 --num-layers 2 --dropout 0.3 \
  --lr 5e-4 --epochs 300 --patience 30 --lr-scheduler
```

> Lưu ý: hidden=64/dropout=0.4 đã được thử để giảm variance nhưng cho kết quả
> tệ hơn rõ rệt (underfit các run nhỏ) — giữ nguyên 128/0.3.

**Threshold tuning:** mặc định mọi model (GNN lẫn baseline) tự chọn ngưỡng quyết định
trên tập val (sweep giới hạn [0.3, 0.7], chỉ nhận nếu val macro-F1 cải thiện ≥ 0.02 so
với 0.5). Ngưỡng được lưu vào cột `threshold` trong `metrics.csv` và `metadata.json`.
Tắt bằng `--no-tune-threshold` (chỉ GNN).

### Ablation: GNN không dùng edge features

```bash
python3 -m src.training.gnn.train_gnn --run-name <RUN_NAME> --model <MODEL> --no-edge-features
```

Kết quả lưu vào `outputs/gnn/<MODEL>-noedge/<RUN_NAME>/` để so sánh trực tiếp với bản đầy đủ.

## 6. Train batch trên nhiều run

### MLP

```bash
./scripts/train/mlp/run_all_mlp_for_runs.sh
./scripts/train/mlp/run_all_mlp_for_runs.sh 'batch_*'
```

### XGBoost

```bash
./scripts/train/xgb/run_all_xgb_for_runs.sh
./scripts/train/xgb/run_all_xgb_for_runs.sh 'batch_*'
```

### GNN (GraphSAGE & GAT)

```bash
./scripts/train/gnn/run_all_gnn_for_runs.sh 'batch_*' graphsage
./scripts/train/gnn/run_all_gnn_for_runs.sh 'batch_*' gat
```

### Edge-Aware GraphSAGE

```bash
./scripts/train/gnn/run_edge_sage_for_runs.sh            # tất cả run, hidden=128 dropout=0.3
HIDDEN=64 DROPOUT=0.4 ./scripts/train/gnn/run_edge_sage_for_runs.sh   # override nếu cần
```

## 6b. Đánh giá cross-run (Leave-One-Run-Out)

Đánh giá khả năng tổng quát hóa sang run chưa từng thấy: với mỗi balanced run,
train trên các run còn lại và test trên **toàn bộ** run bị giữ lại. Chạy cả
5 model (graphsage, gat, edge-sage, xgb, mlp) trên 4 fold:

```bash
./scripts/train/gnn/run_loro.sh
BALANCED_IDS="01 04 05 07" ./scripts/train/gnn/run_loro.sh   # đổi tập fold nếu cần
```

Chạy lẻ một fold:

```bash
python3 -m src.training.gnn.train_gnn_loro --test-run <RUN_A> --train-runs <RUN_B>,<RUN_C> --model edge-sage
python3 -m src.training.baselines.loro_baselines --test-run <RUN_A> --train-runs <RUN_B>,<RUN_C> --model xgb
```

Kết quả lưu tại `outputs/loro/<MODEL_ID>/<TEST_RUN>/`. Baseline LORO dùng feature
thô (`features/edges_labeled.csv`) thay vì bản chuẩn hóa per-run để tránh leak
thông tin run qua scaler.

## 7. Tổng hợp metrics và vẽ biểu đồ so sánh

### Tổng hợp toàn bộ mô hình (Baselines + GNN)

Chạy script gộp để gom mọi dữ liệu về một bảng so sánh:

```bash
./scripts/train/aggregate_all.sh
```

Hoặc gọi trực tiếp module (nhiều tùy chọn hơn):

```bash
# Loại các run degenerate (positive_ratio > 0.95 hoặc < 0.05) — dùng cho bảng chính
python3 -m src.evaluation.aggregate_all_metrics --filter-balanced

# Tổng hợp kết quả LORO (đọc outputs/loro thay vì outputs/baselines + outputs/gnn)
python3 -m src.evaluation.aggregate_all_metrics --loro
```

Dữ liệu tổng hợp sẽ lưu tại:
*   `outputs/aggregates/all_models/detailed_metrics.csv`
*   `outputs/aggregates/all_models/summary_by_model_split.csv`
*   `outputs/aggregates/all_models/summary_by_scenario_model_split.csv`
*   `outputs/aggregates/loro/…` (cùng cấu trúc, cho kết quả cross-run)

### Vẽ biểu đồ so sánh hiệu năng

Sau khi đã chạy lệnh tổng hợp bên trên, bạn có thể sinh biểu đồ so sánh trực quan (Accuracy, F1, Recall) để đưa vào slide hoặc báo cáo:

```bash
# Biểu đồ within-run (mặc định)
python3 -m src.evaluation.plot_comparison

# Biểu đồ cross-run LORO
python3 -m src.evaluation.plot_comparison \
  --summary-csv outputs/aggregates/loro/summary_by_model_split.csv \
  --output-dir outputs/aggregates/loro \
  --filename loro_comparison.png \
  --title "Cross-Run Generalization (Leave-One-Run-Out)"
```

Biểu đồ được lưu tại:
*   [outputs/aggregates/all_models/model_comparison.png](file:///Users/dtam.21/Code/DACN/outputs/aggregates/all_models/model_comparison.png)
*   [outputs/aggregates/loro/loro_comparison.png](file:///Users/dtam.21/Code/DACN/outputs/aggregates/loro/loro_comparison.png)

### Làm mới metrics GNN cũ (ROC-AUC, PR-AUC, inference time)

`metrics.csv` giờ có thêm `roc_auc`, `pr_auc`, `inference_time_ms`,
`inference_ms_per_sample`. Với các GNN đã train trước khi bổ sung các cột này,
tính lại từ model đã lưu mà **không cần train lại**:

```bash
python3 -m src.evaluation.refresh_gnn_metrics --run-pattern '<PATTERN>'
```

(Baseline tabular train nhanh nên chạy lại script train là đủ.)

## 7b. Routing support và đánh giá hiệu năng mạng (replay)

Tích hợp xác suất ổn định dự đoán vào chọn tuyến (`w = 1 − ŷ` + Dijkstra) và
so sánh 4 chiến lược (shortest-hop ≈ OLSR lý tưởng, delay-weighted,
XGBoost-assisted, GNN-assisted) bằng replay trên test snapshot. Chi tiết:
[src/routing/README.md](../src/routing/README.md).

```bash
# Toàn bộ: inference + replay từng run + aggregate + biểu đồ
./scripts/routing/run_routing_for_runs.sh '<PATTERN>' edge-sage

# Khảo sát trade-off p_th (an toàn tuyến ↔ duy trì liên thông)
for d in data/graph_dataset/<PATTERN>; do
  python3 -m src.routing.replay_eval --run-name "$(basename "$d")" \
    --p-th 0.3,0.5,0.7 --strict
done
python3 -m src.routing.plot_pth_sweep
```

Kết quả: `outputs/aggregates/routing/{routing_comparison.png, pth_tradeoff.png}`.

### Tổng hợp riêng cho Baselines (đồ cũ)

```bash
./scripts/train/aggregate_baselines.sh
./scripts/train/aggregate_baselines.sh '*' 'batch_*'
```

## 8. Vị trí output chính

```text
data/raw_snapshots/<RUN_NAME>/
data/graph_dataset/<RUN_NAME>/
outputs/baselines/<MODEL_ID>/<RUN_NAME>/    # mlp | xgb | threshold | logreg | rf
outputs/gnn/<MODEL_ID>/<RUN_NAME>/          # graphsage | gat | edge-sage | *-noedge
outputs/loro/<MODEL_ID>/<TEST_RUN>/         # kết quả leave-one-run-out
outputs/routing/<RUN_NAME>/                 # predictions + replay (summary, details, *_pth*)
outputs/aggregates/all_models/
outputs/aggregates/loro/
outputs/aggregates/routing/
outputs/aggregates/baselines/
```

## 9. Luồng chuẩn hằng ngày

```bash
./scripts/dataset/run_many_random_datasets_ns3.sh 100 exp01
./scripts/train/mlp/run_all_mlp_for_runs.sh 'exp01_*'
./scripts/train/xgb/run_all_xgb_for_runs.sh 'exp01_*'
# + 3 baseline bổ sung (threshold/logreg/rf) — xem mục 5
./scripts/train/gnn/run_all_gnn_for_runs.sh 'exp01_*' graphsage
./scripts/train/gnn/run_all_gnn_for_runs.sh 'exp01_*' gat
./scripts/train/gnn/run_edge_sage_for_runs.sh 'exp01_*'
BALANCED_IDS="..." ./scripts/train/gnn/run_loro.sh
python3 -m src.evaluation.aggregate_all_metrics --filter-balanced
python3 -m src.evaluation.aggregate_all_metrics --loro
python3 -m src.evaluation.plot_comparison
./scripts/routing/run_routing_for_runs.sh 'exp01_*' edge-sage
```

## 10. Scenario mẫu cho báo cáo

### Random Waypoint baseline

```bash
./scripts/dataset/run_one_dataset.sh seed_42_rwp 42 random-waypoint
```

### Gauss-Markov baseline

```bash
./scripts/dataset/run_one_dataset.sh seed_42_gm 42 gauss-markov
```

### Cùng mobility, đổi seed

```bash
./scripts/dataset/run_one_dataset.sh seed_43_rwp 43 random-waypoint
./scripts/dataset/run_one_dataset.sh seed_44_rwp 44 random-waypoint
./scripts/dataset/run_one_dataset.sh seed_45_rwp 45 random-waypoint
```

### Scenario mạng thưa hơn

```bash
SIM_NUM_UAVS=6 \
SIM_COMM_RANGE=180 \
SIM_TIME_STEPS=100 \
SIM_RWP_SPEED_MIN=3 \
SIM_RWP_SPEED_MAX=7 \
./scripts/dataset/run_one_dataset.sh sparse_rwp_seed42 42 random-waypoint
```

### Scenario mạng dày hơn

```bash
SIM_NUM_UAVS=8 \
SIM_COMM_RANGE=280 \
SIM_TIME_STEPS=100 \
SIM_RWP_SPEED_MIN=3 \
SIM_RWP_SPEED_MAX=7 \
./scripts/dataset/run_one_dataset.sh dense_rwp_seed42 42 random-waypoint
```

### Scenario UAV di chuyển nhanh hơn

```bash
SIM_NUM_UAVS=6 \
SIM_COMM_RANGE=230 \
SIM_TIME_STEPS=120 \
SIM_RWP_SPEED_MIN=6 \
SIM_RWP_SPEED_MAX=10 \
./scripts/dataset/run_one_dataset.sh fast_rwp_seed42 42 random-waypoint
```

### Scenario thời gian mô phỏng dài hơn

```bash
SIM_NUM_UAVS=6 \
SIM_COMM_RANGE=230 \
SIM_TIME_STEPS=150 \
SIM_RWP_SPEED_MIN=3 \
SIM_RWP_SPEED_MAX=8 \
./scripts/dataset/run_one_dataset.sh long_rwp_seed42 42 random-waypoint
```

### Bộ scenario gợi ý

```bash
./scripts/dataset/run_one_dataset.sh seed_42_rwp 42 random-waypoint
./scripts/dataset/run_one_dataset.sh seed_42_gm 42 gauss-markov
./scripts/dataset/run_one_dataset.sh seed_43_rwp 43 random-waypoint
SIM_NUM_UAVS=8 SIM_COMM_RANGE=280 ./scripts/dataset/run_one_dataset.sh dense_rwp_seed42 42 random-waypoint
SIM_NUM_UAVS=6 SIM_COMM_RANGE=180 ./scripts/dataset/run_one_dataset.sh sparse_rwp_seed42 42 random-waypoint
SIM_RWP_SPEED_MIN=6 SIM_RWP_SPEED_MAX=10 ./scripts/dataset/run_one_dataset.sh fast_rwp_seed42 42 random-waypoint
```
