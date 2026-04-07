# Ví dụ lệnh chạy cho các scenario báo cáo

## 1. Baseline Random Waypoint

```bash
./scripts/run_one_dataset.sh seed_42_rwp 42 random-waypoint
```

Mục đích:

- Tạo một dataset chuẩn làm baseline ban đầu với `random-waypoint`

## 2. Baseline Gauss-Markov

```bash
./scripts/run_one_dataset.sh seed_42_gm 42 gauss-markov
```

Mục đích:

- So sánh ảnh hưởng của mô hình di chuyển khác nhau trên cùng `seed`

## 3. Cùng mobility, đổi seed

```bash
./scripts/run_one_dataset.sh seed_43_rwp 43 random-waypoint
./scripts/run_one_dataset.sh seed_44_rwp 44 random-waypoint
./scripts/run_one_dataset.sh seed_45_rwp 45 random-waypoint
```

Mục đích:

- Đo độ ổn định của pipeline khi thay đổi ngẫu nhiên khởi tạo

## 4. Scenario mạng thưa hơn

```bash
SIM_NUM_UAVS=6 \
SIM_COMM_RANGE=180 \
SIM_TIME_STEPS=100 \
SIM_RWP_SPEED_MIN=3 \
SIM_RWP_SPEED_MAX=7 \
./scripts/run_one_dataset.sh sparse_rwp_seed42 42 random-waypoint
```

Mục đích:

- Giảm `COMM_RANGE` để tạo topology đứt gãy nhiều hơn

## 5. Scenario mạng dày hơn

```bash
SIM_NUM_UAVS=8 \
SIM_COMM_RANGE=280 \
SIM_TIME_STEPS=100 \
SIM_RWP_SPEED_MIN=3 \
SIM_RWP_SPEED_MAX=7 \
./scripts/run_one_dataset.sh dense_rwp_seed42 42 random-waypoint
```

Mục đích:

- Tăng số UAV và `COMM_RANGE` để tạo nhiều liên kết hơn

## 6. Scenario UAV di chuyển nhanh hơn

```bash
SIM_NUM_UAVS=6 \
SIM_COMM_RANGE=230 \
SIM_TIME_STEPS=120 \
SIM_RWP_SPEED_MIN=6 \
SIM_RWP_SPEED_MAX=10 \
./scripts/run_one_dataset.sh fast_rwp_seed42 42 random-waypoint
```

Mục đích:

- Tăng tốc độ di chuyển để làm topology biến động mạnh hơn

## 7. Scenario thời gian mô phỏng dài hơn

```bash
SIM_NUM_UAVS=6 \
SIM_COMM_RANGE=230 \
SIM_TIME_STEPS=150 \
SIM_RWP_SPEED_MIN=3 \
SIM_RWP_SPEED_MAX=8 \
./scripts/run_one_dataset.sh long_rwp_seed42 42 random-waypoint
```

Mục đích:

- Tăng số `TIME_STEPS` để lấy nhiều snapshot hơn

## 8. Sinh nhiều dataset ngẫu nhiên cho thống kê ban đầu

```bash
./scripts/run_many_random_datasets.sh 10 exp01
```

Mục đích:

- Tạo nhanh khoảng 10 dataset khác nhau
- Phù hợp để lấy metric trung bình ban đầu qua nhiều scenario

## 9. Bộ scenario gợi ý cho báo cáo

Có thể chọn một bộ nhỏ như sau:

```bash
./scripts/run_one_dataset.sh seed_42_rwp 42 random-waypoint
./scripts/run_one_dataset.sh seed_42_gm 42 gauss-markov
./scripts/run_one_dataset.sh seed_43_rwp 43 random-waypoint
SIM_NUM_UAVS=8 SIM_COMM_RANGE=280 ./scripts/run_one_dataset.sh dense_rwp_seed42 42 random-waypoint
SIM_NUM_UAVS=6 SIM_COMM_RANGE=180 ./scripts/run_one_dataset.sh sparse_rwp_seed42 42 random-waypoint
SIM_RWP_SPEED_MIN=6 SIM_RWP_SPEED_MAX=10 ./scripts/run_one_dataset.sh fast_rwp_seed42 42 random-waypoint
```

Bộ này cho phép so sánh:

- đổi seed
- đổi mobility model
- đổi mật độ topology
- đổi cường độ biến động chuyển động
