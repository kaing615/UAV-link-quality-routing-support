# Scenario Catalog

## 1. Mục tiêu

Liệt kê các scenario chuẩn dùng trong báo cáo để:

- chạy thí nghiệm nhất quán
- dễ đặt tên `RUN_NAME`
- dễ nhóm kết quả theo scenario

## 2. Scenario chuẩn

### 2.1 `rwp`

Tên:

- `random-waypoint baseline`

Mô tả:

- mobility dùng `random-waypoint`
- các tham số khác giữ mức mặc định hoặc gần mặc định

Mục đích:

- baseline chuẩn cho hầu hết các thí nghiệm ban đầu

Ví dụ:

```bash
./scripts/run_one_dataset.sh seed_42_rwp 42 random-waypoint
```

### 2.2 `gm`

Tên:

- `gauss-markov baseline`

Mô tả:

- mobility dùng `gauss-markov`
- giữ các tham số chính tương tự `rwp`

Mục đích:

- so sánh tác động của mobility model

Ví dụ:

```bash
./scripts/run_one_dataset.sh seed_42_gm 42 gauss-markov
```

### 2.3 `dense`

Mô tả:

- tăng `NUM_UAVS` hoặc `COMM_RANGE`
- topology dày hơn

Dấu hiệu:

- số cạnh trung bình cao hơn
- route dễ tồn tại hơn

Ví dụ:

```bash
SIM_NUM_UAVS=8 \
SIM_COMM_RANGE=280 \
./scripts/run_one_dataset.sh dense_rwp_seed42 42 random-waypoint
```

### 2.4 `sparse`

Mô tả:

- giảm `COMM_RANGE`
- có thể giữ hoặc giảm `NUM_UAVS`

Dấu hiệu:

- topology rời rạc hơn
- nhiều link yếu hoặc mất kết nối hơn

Ví dụ:

```bash
SIM_NUM_UAVS=6 \
SIM_COMM_RANGE=180 \
./scripts/run_one_dataset.sh sparse_rwp_seed42 42 random-waypoint
```

### 2.5 `fast`

Mô tả:

- tăng `RWP_SPEED_RANGE`
- topology thay đổi nhanh hơn theo thời gian

Dấu hiệu:

- liên kết biến động mạnh
- bài toán dự đoán khó hơn

Ví dụ:

```bash
SIM_RWP_SPEED_MIN=6 \
SIM_RWP_SPEED_MAX=10 \
./scripts/run_one_dataset.sh fast_rwp_seed42 42 random-waypoint
```

## 3. Scenario phụ có thể thêm

### `long`

- tăng `TIME_STEPS`
- dùng để lấy nhiều snapshot hơn

Ví dụ:

```bash
SIM_TIME_STEPS=150 \
./scripts/run_one_dataset.sh long_rwp_seed42 42 random-waypoint
```

### `multi_seed`

- cùng một scenario nhưng đổi nhiều `seed`
- dùng để đánh giá độ ổn định của baseline

Ví dụ:

```bash
./scripts/run_one_dataset.sh seed_42_rwp 42 random-waypoint
./scripts/run_one_dataset.sh seed_43_rwp 43 random-waypoint
./scripts/run_one_dataset.sh seed_44_rwp 44 random-waypoint
```

## 4. Gợi ý bộ scenario cho báo cáo

Một bộ nhỏ nhưng đủ dùng:

- `rwp`
- `gm`
- `dense`
- `sparse`
- `fast`

Nếu cần chắc hơn:

- với mỗi scenario, chạy 3 đến 5 seed

## 5. Quy tắc đặt tên

Nên để tên run phản ánh scenario, ví dụ:

- `seed_42_rwp`
- `seed_42_gm`
- `dense_rwp_seed42`
- `sparse_rwp_seed42`
- `fast_rwp_seed42`

Với batch random:

- scenario có thể suy ra từ `RUN_NAME`, ví dụ `_rwp_`, `_gm_`

## 6. Cách dùng trong báo cáo

Nên nhóm kết quả theo:

1. Mobility:
   - `rwp`
   - `gm`

2. Density:
   - `dense`
   - `sparse`

3. Dynamics:
   - `fast`

4. Stability:
   - nhiều seed trong cùng một scenario

## 7. Ghi chú

- `rwp` và `gm` là hai scenario nền để so baseline mobility.
- `dense` và `sparse` giúp đánh giá ảnh hưởng của connectivity.
- `fast` giúp đánh giá ảnh hưởng của topology dynamics.
- Khi số run còn ít, ưu tiên `rwp`, `gm`, và `multi_seed`.
