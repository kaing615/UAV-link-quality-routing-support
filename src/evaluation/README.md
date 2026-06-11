# Evaluation

Thư mục này dành cho code đánh giá metric, tổng hợp kết quả, và phân tích đầu ra của mô hình.

## Các module chính

### `aggregate_all_metrics.py`

Gom `metrics.csv` của mọi model (baselines + GNN) theo run, tính mean/std theo
`model_id`/`scenario`/`split`.

```bash
# Tất cả run
python3 -m src.evaluation.aggregate_all_metrics

# Loại các run degenerate (positive_ratio > 0.95 hoặc < 0.05) — dùng cho bảng chính
python3 -m src.evaluation.aggregate_all_metrics --filter-balanced

# Tổng hợp kết quả Leave-One-Run-Out (đọc outputs/loro, ghi outputs/aggregates/loro)
python3 -m src.evaluation.aggregate_all_metrics --loro
```

### `plot_comparison.py`

Vẽ biểu đồ cột so sánh các model (Accuracy / F1 / Macro-F1 / Recall, kèm error bar std).
Nhận diện cả các model ablation (`*-noedge`).

```bash
# Within-run
python3 -m src.evaluation.plot_comparison

# Cross-run LORO
python3 -m src.evaluation.plot_comparison \
  --summary-csv outputs/aggregates/loro/summary_by_model_split.csv \
  --output-dir outputs/aggregates/loro \
  --filename loro_comparison.png \
  --title "Cross-Run Generalization (Leave-One-Run-Out)"
```

### `aggregate_baseline_metrics.py`

Bản cũ, chỉ gom baselines (`outputs/baselines`). Giữ để tương thích với
`scripts/train/aggregate_baselines.sh`.
