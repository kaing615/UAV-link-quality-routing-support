# Scripts

Tổng quan nhanh các script trong thư mục `scripts/`.

`docs/quick_start.md` là tài liệu chuẩn duy nhất cho các lệnh vận hành hằng ngày.

## Cấu trúc

```text
scripts/
├── dataset/
│   ├── run_one_dataset.sh
│   └── run_many_random_datasets.sh
├── train/
│   ├── aggregate_baselines.sh
│   ├── mlp/run_all_mlp_for_runs.sh
│   └── xgb/run_all_xgb_for_runs.sh
├── utils/
│   └── list_run_names.sh
```

## Dataset

### `scripts/dataset/run_one_dataset.sh`

- Sinh một dataset từ simulator.
- Tự chạy preprocessing, standardize, và imbalance handling cho baseline non-GNN.

Ví dụ:

```bash
./scripts/dataset/run_one_dataset.sh seed_42_rwp 42 random-waypoint
```

### `scripts/dataset/run_many_random_datasets.sh`

- Sinh nhiều dataset ngẫu nhiên.
- Mỗi run được lưu riêng trong `data/raw_snapshots/` và `data/graph_dataset/`.

Ví dụ:

```bash
./scripts/dataset/run_many_random_datasets.sh 10 exp01
```

## Train

### `scripts/train/mlp/run_all_mlp_for_runs.sh`

- Chạy baseline MLP cho nhiều run.

Ví dụ:

```bash
./scripts/train/mlp/run_all_mlp_for_runs.sh 'batch_*'
```

### `scripts/train/xgb/run_all_xgb_for_runs.sh`

- Chạy baseline XGBoost cho nhiều run.

Ví dụ:

```bash
./scripts/train/xgb/run_all_xgb_for_runs.sh 'batch_*'
```

### `scripts/train/aggregate_baselines.sh`

- Tổng hợp `metrics.csv` từ nhiều model và nhiều run.
- Wrapper cho `python -m src.evaluation.aggregate_baseline_metrics`.

Ví dụ:

```bash
./scripts/train/aggregate_baselines.sh
./scripts/train/aggregate_baselines.sh '*' 'batch_*'
./scripts/train/aggregate_baselines.sh 'xgb' 'exp01_*'
```

## Utils

### `scripts/utils/list_run_names.sh`

- Liệt kê các `RUN_NAME` trong `data/graph_dataset/`.
- Wrapper cho `python -m src.utils.list_run_names`.

Ví dụ:

```bash
./scripts/utils/list_run_names.sh
./scripts/utils/list_run_names.sh 'batch_*'
```

## Tham khảo thêm

- [docs/quick_start.md](../docs/quick_start.md): lệnh chuẩn dùng hằng ngày và scenario mẫu
