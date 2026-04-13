# Data Generation

## 1. Mục tiêu

Sinh nhiều dataset UAV riêng biệt từ simulator để:

- tránh phụ thuộc vào một seed duy nhất
- so sánh nhiều mobility / topology / density scenario
- đánh giá baseline trên nhiều run độc lập

## 2. Cấu trúc dữ liệu

### Raw data

Mỗi lần chạy simulator với một `RUN_NAME` sẽ tạo một thư mục riêng:

```text
data/raw_snapshots/<RUN_NAME>/
```

Gồm:

- `nodes.csv`
- `edges.csv`
- `traffic_log.csv`
- `nodes.parquet`
- `edges.parquet`
- `traffic_log.parquet`
- `scenario.json`

### Preprocessed data

Mỗi `RUN_NAME` sau khi chạy preprocessing sẽ có:

```text
data/graph_dataset/<RUN_NAME>/
```

Gồm:

- `processed/`
- `splits/`
- `graph/`
- `baseline_standardized/`
- `baseline_standardized/imbalance/`

## 3. Ý nghĩa của `RUN_NAME`

`RUN_NAME` là định danh duy nhất của một lần mô phỏng.

Ví dụ:

```text
batch_20260407_183340_01_gm_s14994_n8_c231_t119
```

Ý nghĩa:

- `batch_20260407_183340`: prefix + timestamp
- `01`: thứ tự dataset trong batch
- `gm`: mobility `gauss-markov`
- `s14994`: seed `14994`
- `n8`: `NUM_UAVS = 8`
- `c231`: `COMM_RANGE = 231`
- `t119`: `TIME_STEPS = 119`

## 4. Tham số có thể thay đổi

Simulator hiện hỗ trợ thay đổi qua biến môi trường:

- `SIM_SEED`
- `SIM_RUN_NAME`
- `SIM_MOBILITY_MODEL`
- `SIM_NUM_UAVS`
- `SIM_COMM_RANGE`
- `SIM_TIME_STEPS`
- `SIM_RWP_SPEED_MIN`
- `SIM_RWP_SPEED_MAX`

## 5. Script sinh dữ liệu

### Chạy một dataset

```bash
./scripts/run_one_dataset.sh seed_42_rwp 42 random-waypoint
```

Hoặc:

```bash
SIM_NUM_UAVS=8 \
SIM_COMM_RANGE=245 \
SIM_TIME_STEPS=120 \
SIM_RWP_SPEED_MIN=3 \
SIM_RWP_SPEED_MAX=7 \
./scripts/run_one_dataset.sh seed_custom 42 random-waypoint
```

### Chạy nhiều dataset ngẫu nhiên

```bash
./scripts/run_many_random_datasets.sh
./scripts/run_many_random_datasets.sh 10 exp01
```

Script batch sẽ random:

- `SEED`
- `MOBILITY_MODEL`
- `NUM_UAVS`
- `COMM_RANGE`
- `TIME_STEPS`
- `RWP_SPEED_RANGE`

## 6. Khi nào giữ seed, khi nào đổi seed

### Giữ cùng seed

Dùng khi muốn so sánh công bằng giữa các cấu hình.

Ví dụ:

- cùng `seed=42`
- đổi `random-waypoint` sang `gauss-markov`

### Đổi seed

Dùng khi muốn tạo nhiều dataset độc lập và đánh giá độ ổn định của mô hình.

Ví dụ:

- `seed=42`
- `seed=43`
- `seed=44`
- `seed=45`
- `seed=46`

## 7. Output plot

Mỗi run sẽ có thư mục plot riêng:

```text
outputs/plots/<RUN_NAME>/
```

Điều này tránh ghi đè ảnh topology giữa các run khác nhau.
