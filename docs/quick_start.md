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

## 4. Liệt kê các run đã có

```bash
python3 -m src.utils.list_run_names
python3 -m src.utils.list_run_names 'batch_*'
./scripts/utils/list_run_names.sh 'batch_*'
```

## 5. Train một baseline trên một run

### MLP

```bash
python3 -m src.training.baselines.mlp_baseline --run-name <RUN_NAME>
```

### XGBoost

```bash
python3 -m src.training.baselines.xgb_baseline --run-name <RUN_NAME>
```

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

## 7. Tổng hợp metrics cho báo cáo

### Dùng Python module trực tiếp

```bash
python3 -m src.evaluation.aggregate_baseline_metrics
python3 -m src.evaluation.aggregate_baseline_metrics --model-pattern 'xgb' --run-pattern 'batch_*'
```

### Dùng script batch

```bash
./scripts/train/aggregate_baselines.sh
./scripts/train/aggregate_baselines.sh '*' 'batch_*'
./scripts/train/aggregate_baselines.sh 'xgb' 'batch_*'
```

Output:

```text
outputs/aggregates/baselines/detailed_metrics.csv
outputs/aggregates/baselines/summary_by_model_split.csv
outputs/aggregates/baselines/summary_by_scenario_model_split.csv
```

## 8. Vị trí output chính

```text
data/raw_snapshots/<RUN_NAME>/
data/graph_dataset/<RUN_NAME>/
outputs/baselines/mlp/<RUN_NAME>/
outputs/baselines/xgb/<RUN_NAME>/
outputs/aggregates/baselines/
```

## 9. Luồng chuẩn hằng ngày

```bash
./scripts/dataset/run_many_random_datasets.sh 10 exp01
./scripts/train/mlp/run_all_mlp_for_runs.sh 'exp01_*'
./scripts/train/xgb/run_all_xgb_for_runs.sh 'exp01_*'
./scripts/train/aggregate_baselines.sh '*' 'exp01_*'
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
