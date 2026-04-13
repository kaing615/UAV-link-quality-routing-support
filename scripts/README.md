# Hướng dẫn chạy pipeline sinh dữ liệu

## 1. Chạy một dataset

Script:

```bash
./scripts/run_one_dataset.sh RUN_NAME [SEED] [MOBILITY_MODEL]
```

Ví dụ:

```bash
./scripts/run_one_dataset.sh seed_42_rwp 42 random-waypoint
./scripts/run_one_dataset.sh seed_42_gm 42 gauss-markov
```

### Có thể truyền thêm biến môi trường

```bash
SIM_NUM_UAVS=8 \
SIM_COMM_RANGE=245 \
SIM_TIME_STEPS=120 \
SIM_RWP_SPEED_MIN=3 \
SIM_RWP_SPEED_MAX=7 \
./scripts/run_one_dataset.sh seed_custom 42 random-waypoint
```

### Script sẽ tự chạy 4 bước

1. `simulation/main.py`
2. `preprocessing/run_preprocessing.py --run-name ...`
3. `preprocessing/non-gnn/standardize_baseline_data.py --run-name ...`
4. `preprocessing/non-gnn/handle_imbalance.py --run-name ...`

## 2. Chạy nhiều dataset ngẫu nhiên

Script:

```bash
./scripts/run_many_random_datasets.sh [COUNT] [PREFIX]
```

Ví dụ:

```bash
./scripts/run_many_random_datasets.sh
./scripts/run_many_random_datasets.sh 10 exp01
./scripts/run_many_random_datasets.sh 5 testbatch
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
