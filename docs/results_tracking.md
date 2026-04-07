# Results Tracking

## 1. Mục tiêu

Quy định cách lưu, đọc, và tổng hợp metric từ nhiều `RUN_NAME` để:

- tránh ghi đè kết quả giữa các run
- dễ gom kết quả theo model hoặc theo scenario
- thuận tiện cho phần bảng biểu trong báo cáo

## 2. Quy ước output hiện tại

### Raw data

```text
data/raw_runs/<RUN_NAME>/
```

### Preprocessed data

```text
data/preprocessed_runs/<RUN_NAME>/
```

### Baseline outputs

Theo từng model:

```text
outputs/baselines/<MODEL_ID>/<RUN_NAME>/
```

Ví dụ với MLP:

```text
outputs/baselines/mlp/<RUN_NAME>/
```

## 3. File metric nên có

Mỗi baseline nên lưu tối thiểu:

- `metrics.csv`
- `metadata.json`
- `model.pkl` hoặc artifact tương đương

Có thể lưu thêm:

- `val_predictions.csv`
- `test_predictions.csv`

## 4. Quy ước cột trong `metrics.csv`

Tối thiểu:

- `model_id`
- `model_name`
- `split`
- `n_samples`
- `positive_ratio`
- `accuracy`
- `precision`
- `recall`
- `f1`

Nên thêm:

- `macro_f1`
- `tn`
- `fp`
- `fn`
- `tp`
- `unique_labels`
- `has_both_classes`

## 5. Quy ước `model_id`

```text
rssi_snr_thresh
logreg
rf
mlp
xgb
```

## 6. Quy ước `model_name`

```text
RSSI/SNR Threshold
Logistic Regression
Random Forest
Small MLP
XGBoost
```

## 7. Cách đọc kết quả theo từng run

Ví dụ với MLP:

```text
outputs/baselines/mlp/<RUN_NAME>/metrics.csv
outputs/baselines/mlp/<RUN_NAME>/metadata.json
```

Một `RUN_NAME` tương ứng với một dataset mô phỏng riêng.

## 8. Cách gom kết quả nhiều run

### Gom theo model

Ví dụ:

- đọc tất cả file:

```text
outputs/baselines/mlp/*/metrics.csv
```

- thêm cột `run_name` từ tên thư mục cha
- concat tất cả lại thành một bảng tổng hợp

### Gom theo scenario

Có thể suy ra scenario từ `RUN_NAME`, ví dụ:

- `_rwp_` -> `random-waypoint`
- `_gm_` -> `gauss-markov`
- `dense_` -> dense
- `sparse_` -> sparse
- `fast_` -> fast

## 9. Kiểu bảng tổng hợp nên làm

### Bảng chi tiết

Mỗi dòng là một `run_name` + `model_id` + `split`

Ví dụ cột:

- `run_name`
- `model_id`
- `split`
- `accuracy`
- `precision`
- `recall`
- `f1`
- `macro_f1`

### Bảng trung bình theo model

Nhóm theo:

- `model_id`
- `split`

Rồi tính:

- mean
- std

### Bảng trung bình theo scenario

Nhóm theo:

- `scenario`
- `model_id`
- `split`

## 10. Cách diễn giải kết quả

- Không chỉ nhìn `accuracy`.
- Với bài toán lệch lớp, cần xem thêm:
  - `recall`
  - `f1`
  - `macro_f1`
  - confusion matrix
- Nếu `has_both_classes = false`, cần ghi chú rằng split đó không đủ mạnh để kết luận đầy đủ.

## 11. Khuyến nghị thực tế

- Mỗi baseline nên có một thư mục riêng dưới `outputs/baselines/<MODEL_ID>/`
- Mỗi `RUN_NAME` là một thư mục con
- Khi viết báo cáo, ưu tiên dùng bảng tổng hợp mean/std trên nhiều run thay vì chỉ một run đơn lẻ
