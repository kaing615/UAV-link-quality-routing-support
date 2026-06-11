# Baseline Pipeline

## 1. Mục tiêu

Pipeline baseline non-GNN dùng để:

- chuẩn hóa dữ liệu tabular
- xử lý mất cân bằng lớp
- huấn luyện và đánh giá các baseline thường

Các baseline mục tiêu:

- RSSI/SNR threshold
- Logistic Regression
- Random Forest
- Small MLP
- XGBoost

## 2. Thứ tự pipeline

### Bước 1. Sinh raw data từ simulator

Script:

```bash
./scripts/dataset/run_one_dataset.sh <RUN_NAME> <SEED> <MOBILITY_MODEL>
```

Hoặc chạy nhiều dataset:

```bash
./scripts/dataset/run_many_random_datasets.sh
```

### Bước 2. Graph preprocessing

Script:

```bash
python3 src/preprocessing/run_preprocessing.py --run-name <RUN_NAME>
```

Output:

- `features/nodes_features.csv`
- `features/edges_features.csv`
- `features/edges_labeled.csv`
- `splits/time_splits.csv`
- `graph_dataset/train.pt`
- `graph_dataset/val.pt`
- `graph_dataset/test.pt`

### Bước 3. Standardize cho baseline non-GNN

Script:

```bash
python3 src/preprocessing/non-gnn/standardize_baseline_data.py --run-name <RUN_NAME>
```

Output:

- `baseline_standardized/train_scaled.csv`
- `baseline_standardized/val_scaled.csv`
- `baseline_standardized/test_scaled.csv`
- `baseline_standardized/all_scaled.csv`
- `baseline_standardized/scaler_stats.json`

### Bước 4. Xử lý mất cân bằng lớp

Script:

```bash
python3 src/preprocessing/non-gnn/handle_imbalance.py --run-name <RUN_NAME>
```

Output:

- `baseline_standardized/imbalance/train_weighted.csv`
- `baseline_standardized/imbalance/train_oversampled.csv`
- `baseline_standardized/imbalance/imbalance_summary.json`

## 3. Ý nghĩa các file non-GNN

### `train_scaled.csv`

- dữ liệu train đã chuẩn hóa
- chưa xử lý mất cân bằng lớp

### `train_weighted.csv`

- giữ nguyên train set
- thêm cột `sample_weight`
- dùng cho model hỗ trợ `sample_weight`

### `train_oversampled.csv`

- train set đã được oversample lớp thiểu số
- dùng khi muốn fit trực tiếp mà không cần truyền `sample_weight`

### `val_scaled.csv`, `test_scaled.csv`

- chỉ chuẩn hóa
- không xử lý mất cân bằng
- dùng để đánh giá đúng phân phối dữ liệu

## 4. Baseline hiện tại

### MLP

Script:

```bash
python3 -m src.training.baselines.mlp_baseline --run-name <RUN_NAME>
```

Output:

```text
outputs/baselines/mlp/<RUN_NAME>/
```

Gồm:

- `metrics.csv`
- `metadata.json`
- `model.pkl`
- `val_predictions.csv`
- `test_predictions.csv`

### Chạy MLP cho tất cả run

```bash
./scripts/train/mlp/run_all_mlp_for_runs.sh
./scripts/train/mlp/run_all_mlp_for_runs.sh 'batch_*'
```

### Tổng hợp metric nhiều baseline

```bash
./scripts/train/aggregate_baselines.sh
./scripts/train/aggregate_baselines.sh '*' 'batch_*'
```

## 5. Quy tắc dùng weighted và oversampled

- `Logistic Regression`: ưu tiên `weighted`
- `Random Forest`: ưu tiên `class_weight`
- `Small MLP`: ưu tiên `weighted`, fallback `oversampled`
- `XGBoost`: ưu tiên `weighted`
- `Threshold baseline`: thường chạy trên dữ liệu raw

## 6. Lưu ý quan trọng

- Không fit scaler trên toàn bộ dataset.
- Chỉ fit scaler trên split `train`.
- Không xử lý mất cân bằng trên `val/test`.
- Với các split lệch lớp mạnh, nên xem thêm `macro_f1` và confusion matrix thay vì chỉ nhìn `accuracy`.
