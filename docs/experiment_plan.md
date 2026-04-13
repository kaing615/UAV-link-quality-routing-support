# Experiment Plan

## 1. Mục tiêu

Đánh giá baseline non-GNN trên nhiều dataset UAV được sinh từ simulator để:

- có kết quả ban đầu đáng tin hơn so với một split duy nhất
- so sánh các scenario mobility / density / connectivity khác nhau
- làm mốc đối chiếu cho GNN sau này

## 2. Baseline cần chạy

- `rssi_snr_thresh`
- `logreg`
- `rf`
- `mlp`
- `xgb`

Tên hiển thị:

- `RSSI/SNR Threshold`
- `Logistic Regression`
- `Random Forest`
- `Small MLP`
- `XGBoost`

## 3. Metric cần lưu

Tối thiểu:

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
- `n_samples`
- `positive_ratio`

## 4. Scenario nên có

### Scenario A. Baseline Random Waypoint

- mobility: `random-waypoint`
- cấu hình mặc định

### Scenario B. Baseline Gauss-Markov

- mobility: `gauss-markov`
- giữ các tham số chính tương tự Scenario A

### Scenario C. Sparse network

- `COMM_RANGE` nhỏ hơn
- ít liên kết hơn

### Scenario D. Dense network

- `NUM_UAVS` lớn hơn hoặc `COMM_RANGE` lớn hơn
- nhiều liên kết hơn

### Scenario E. Fast mobility

- tăng `RWP_SPEED_RANGE`
- topology biến động mạnh hơn

## 5. Thiết kế thí nghiệm đề xuất

### Mức tối thiểu

- 5 đến 10 dataset random
- chạy đầy đủ baseline non-GNN

### Mức tốt hơn

- 3 đến 5 scenario chính
- mỗi scenario chạy 3 đến 5 seed

Ví dụ:

- `random-waypoint`: 5 seed
- `gauss-markov`: 5 seed
- `sparse`: 3 seed
- `dense`: 3 seed
- `fast`: 3 seed

## 6. Cách dùng seed

### Giữ cùng seed

Dùng khi so sánh tác động của một tham số cụ thể.

Ví dụ:

- cùng `seed=42`
- so `random-waypoint` với `gauss-markov`

### Đổi seed

Dùng khi đánh giá độ ổn định của mô hình trên nhiều dataset độc lập.

Ví dụ:

- `seed=42, 43, 44, 45, 46`

## 7. Quy trình chạy thực tế

### Bước 1. Sinh dữ liệu

```bash
./scripts/dataset/run_many_random_datasets.sh 10 exp01
```

Hoặc chạy scenario cố định bằng tay:

```bash
./scripts/dataset/run_one_dataset.sh seed_42_rwp 42 random-waypoint
./scripts/dataset/run_one_dataset.sh seed_42_gm 42 gauss-markov
```

### Bước 2. Chạy baseline

Ví dụ với MLP:

```bash
./scripts/train/mlp/run_all_mlp_for_runs.sh 'exp01_*'
```

### Bước 3. Gom kết quả

Sau khi có đầy đủ baseline:

- gom `metrics.csv` của tất cả run
- tính trung bình / độ lệch chuẩn theo từng model
- lập bảng so sánh cuối

## 8. Kỳ vọng đầu ra cho báo cáo

Nên có ít nhất:

1. Bảng metric theo từng baseline trên từng run
2. Bảng trung bình metric theo từng baseline
3. So sánh giữa các scenario chính
4. Nhận xét baseline nào ổn định nhất trước khi sang GNN

## 9. Ghi chú

- Nếu `val/test` của một run quá lệch lớp, cần ghi rõ hạn chế khi phân tích.
- Không nên kết luận mô hình tốt chỉ dựa vào `accuracy`.
- Với bài toán này, nên chú ý đặc biệt tới `recall`, `f1`, `macro_f1`, và confusion matrix.
