# Hướng dẫn chạy pipeline sinh dữ liệu

## 1. Chạy một dataset

Script:

```bash
./scripts/dataset/run_one_dataset.sh RUN_NAME [SEED] [MOBILITY_MODEL]
```

Ví dụ:

```bash
./scripts/dataset/run_one_dataset.sh seed_42_rwp 42 random-waypoint
./scripts/dataset/run_one_dataset.sh seed_42_gm 42 gauss-markov
```

### Có thể truyền thêm biến môi trường

```bash
SIM_NUM_UAVS=8 \
SIM_COMM_RANGE=245 \
SIM_TIME_STEPS=120 \
SIM_RWP_SPEED_MIN=3 \
SIM_RWP_SPEED_MAX=7 \
./scripts/dataset/run_one_dataset.sh seed_custom 42 random-waypoint
```

### Script sẽ tự chạy 4 bước

1. `simulation/main.py`
2. `src/preprocessing/run_preprocessing.py --run-name ...`
3. `src/preprocessing/non-gnn/standardize_baseline_data.py --run-name ...`
4. `src/preprocessing/non-gnn/handle_imbalance.py --run-name ...`

## 2. Chạy nhiều dataset ngẫu nhiên

Script:

```bash
./scripts/dataset/run_many_random_datasets.sh [COUNT] [PREFIX]
```

Ví dụ:

```bash
./scripts/dataset/run_many_random_datasets.sh
./scripts/dataset/run_many_random_datasets.sh 10 exp01
./scripts/dataset/run_many_random_datasets.sh 5 testbatch
```

### Script này sẽ random cho từng dataset

- `SEED`
- `MOBILITY_MODEL`
- `NUM_UAVS`
- `COMM_RANGE`
- `TIME_STEPS`
- `RWP_SPEED_RANGE`

## 3. Output nằm ở đâu

Với mỗi `RUN_NAME`, dữ liệu sẽ được lưu ở:

### Raw từ simulator

```text
data/raw_snapshots/<RUN_NAME>/
```

Gồm:

- `nodes.csv`
- `edges.csv`
- `traffic_log.csv`
- `scenario.json`

### Dữ liệu đã preprocessing
# Scripts Overview

Tóm tắt nhanh các script hiện có trong thư mục `scripts/`.

## Cấu trúc hiện tại

```text
scripts/
├── dataset/
│   ├── run_one_dataset.sh
│   └── run_many_random_datasets.sh
├── train/
│   ├── aggregate_all.sh
│   ├── aggregate_baselines.sh
│   ├── gnn/run_all_gnn_for_runs.sh
│   ├── mlp/run_all_mlp_for_runs.sh
│   └── xgb/run_all_xgb_for_runs.sh
├── utils/
│   └── list_run_names.sh
└── docs/
    ├── README.md
    └── RUN_EXAMPLES.md
```

## Dataset scripts

### `scripts/dataset/run_one_dataset.sh`

- Sinh một dataset từ simulator.
- Tự chạy luôn các bước preprocessing và chuẩn bị dữ liệu baseline non-GNN.

Ví dụ:

```bash
./scripts/dataset/run_one_dataset.sh seed_42_rwp 42 random-waypoint
```

### `scripts/dataset/run_many_random_datasets.sh`

- Sinh nhiều dataset ngẫu nhiên liên tiếp.
- Mỗi dataset có `RUN_NAME` riêng dưới `data/raw_snapshots/` và `data/graph_dataset/`.

Ví dụ:

```bash
./scripts/dataset/run_many_random_datasets.sh 10 exp01
```

## Training scripts

### `scripts/train/mlp/run_all_mlp_for_runs.sh`

- Chạy baseline MLP cho nhiều run đã preprocessing.

Ví dụ:

```bash
./scripts/train/mlp/run_all_mlp_for_runs.sh 'olsr_dataset_*'
```

### `scripts/train/xgb/run_all_xgb_for_runs.sh`

- Chạy baseline XGBoost cho nhiều run đã preprocessing.

Ví dụ:

```bash
./scripts/train/xgb/run_all_xgb_for_runs.sh 'olsr_dataset_*'
```

### `scripts/train/gnn/run_all_gnn_for_runs.sh`

- Chạy huấn luyện mô hình GNN (GraphSAGE hoặc GAT) cho nhiều run.

Ví dụ:

```bash
./scripts/train/gnn/run_all_gnn_for_runs.sh 'olsr_dataset_*' graphsage
./scripts/train/gnn/run_all_gnn_for_runs.sh 'olsr_dataset_*' gat
```

### `scripts/train/aggregate_all.sh`

- Tổng hợp `metrics.csv` của cả mô hình Baseline (MLP, XGBoost) và GNN (GraphSAGE, GAT) từ nhiều run.
- Wrapper cho `python -m src.evaluation.aggregate_all_metrics`.

Ví dụ:

```bash
./scripts/train/aggregate_all.sh
./scripts/train/aggregate_all.sh '*' 'olsr_dataset_*'
```

### `scripts/train/aggregate_baselines.sh`

- Tổng hợp `metrics.csv` từ nhiều model và nhiều run.
- Wrapper cho `python -m src.evaluation.aggregate_baseline_metrics`.

Ví dụ:

```bash
./scripts/train/aggregate_baselines.sh
./scripts/train/aggregate_baselines.sh '*' 'olsr_dataset_*'
./scripts/train/aggregate_baselines.sh 'xgb' 'olsr_dataset_*'
```

## Utility scripts

### `scripts/utils/list_run_names.sh`

- Liệt kê các `RUN_NAME` dưới `data/graph_dataset/`.
- Wrapper cho `python -m src.utils.list_run_names`.

Ví dụ:

```bash
./scripts/utils/list_run_names.sh
./scripts/utils/list_run_names.sh 'olsr_dataset_*'
```


## Script docs

### `scripts/docs/README.md`

- Giải thích pipeline sinh dữ liệu.

### `scripts/docs/RUN_EXAMPLES.md`

- Chứa các ví dụ scenario/report.

## Gợi ý dùng hằng ngày

Nếu bạn chỉ cần lệnh chuẩn, xem thêm:

- [docs/quick_start.md](../docs/quick_start.md)

```text
data/graph_dataset/<RUN_NAME>/
```

Gồm:

- `processed/`
- `splits/`
- `graph/`
- `baseline_standardized/`
- `baseline_standardized/imbalance/`

### Plot của simulator

```text
outputs/plots/<RUN_NAME>/
```

## 4. Naming hiện tại

- `run_one_dataset.sh`: chạy một dataset với tham số cụ thể
- `run_many_random_datasets.sh`: chạy nhiều dataset ngẫu nhiên

## 5. Lưu ý

- Nếu đang activate virtualenv thì script sẽ dùng `python3` hiện tại.
- Nếu không có virtualenv active, script sẽ fallback sang `simulation/.venv/bin/python`.
- `run_many_random_datasets.sh` gọi lại `run_one_dataset.sh` cho từng dataset.
