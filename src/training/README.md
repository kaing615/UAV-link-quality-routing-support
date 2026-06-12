# Training

Thư mục này dành cho script huấn luyện, validation, testing, và orchestration cho các mô hình.

## Cấu trúc

```text
training/
├── baselines/
│   ├── common.py                 # load/eval/save dùng chung + threshold tuning trên val
│   ├── mlp_baseline.py           # MLP (within-run)
│   ├── xgb_baseline.py           # XGBoost (within-run)
│   ├── Logistic_Regression_Baseline.py
│   ├── Random_Forest_Baseline.py
│   ├── RSSI_SNR_Baseline.py      # heuristic threshold RSSI/SNR
│   └── loro_baselines.py         # Cả 5 baseline (xgb/mlp/logreg/rf/threshold) theo giao thức Leave-One-Run-Out
└── gnn/
    ├── common.py                 # load_graphs, evaluate_split, threshold tuning
    ├── train_gnn.py              # train within-run: graphsage | gat | edge-sage
    └── train_gnn_loro.py         # train cross-run (Leave-One-Run-Out)
```

## Hai giao thức đánh giá

- **Within-run** (`train_gnn.py`, `*_baseline.py`): train/val/test là các cửa sổ thời gian
  khác nhau của **cùng một** run mô phỏng.
- **Leave-One-Run-Out** (`train_gnn_loro.py`, `loro_baselines.py`): train/val trên N−1 run,
  test trên **toàn bộ** run còn lại (chưa từng thấy) — đo khả năng tổng quát hóa sang
  topology/mobility khác. Baseline LORO dùng feature thô (`features/edges_labeled.csv`)
  để tránh leak thông tin run qua scaler per-run.

## Threshold tuning

Mặc định mọi model chọn ngưỡng quyết định trên tập val: sweep giới hạn `[0.3, 0.7]`,
chỉ chấp nhận nếu val macro-F1 cải thiện ≥ 0.02 so với ngưỡng 0.5 (sweep tự do
[0.05, 0.95] đã được thử và overfit nặng vì val split rất nhỏ). Ngưỡng dùng được ghi
vào cột `threshold` trong `metrics.csv` và `metadata.json`. Tắt bằng `--no-tune-threshold`
(GNN).

## Ablation

`train_gnn.py --no-edge-features` huấn luyện GNN không dùng edge features (cả trong
message passing lẫn decoder); `model_id` nhận hậu tố `-noedge` và kết quả ghi vào
`outputs/gnn/<model>-noedge/<run>/`.

Lệnh cụ thể: xem [docs/quick_start.md](../../docs/quick_start.md).
