# Hướng dẫn dùng dữ liệu xử lý mất cân bằng lớp

## 1. `train_weighted.csv`

- Dùng khi model hỗ trợ `sample_weight` hoặc `class_weight`.
- Phù hợp nhất cho:
  - Logistic Regression
  - XGBoost
  - MLP nếu version `sklearn` hiện tại hỗ trợ `sample_weight`

### Ưu điểm

- Không tạo dữ liệu giả hoặc lặp.
- Train set vẫn giữ phân phối gốc.
- Đây là lựa chọn nên thử trước.

## 2. `train_oversampled.csv`

- Dùng khi model không xử lý weight tốt hoặc khi muốn ép train cân bằng hẳn bằng dữ liệu.
- Phù hợp khi:
  - MLP học kém với `sample_weight`
  - muốn thử thêm một baseline so sánh
  - luồng train không tiện truyền `sample_weight`

### Ưu điểm

- Dễ dùng, chỉ cần `fit(X, y)`.

### Nhược điểm

- Lặp lại mẫu lớp thiểu số nên dễ overfit hơn.

## 3. Khuyến nghị cho repo hiện tại

- Logistic Regression: ưu tiên `train_weighted.csv`
- Random Forest: không cần hai file trên nếu train trực tiếp từ `train_scaled.csv` với `class_weight="balanced_subsample"`
- MLP nhỏ: thử `train_weighted.csv` trước, nếu chưa ổn thì thử `train_oversampled.csv`
- XGBoost: ưu tiên `train_weighted.csv`
- Threshold baseline: không dùng hai file này, thường nên chạy trên dữ liệu raw

## 4. Không nên làm

- Không dùng `train_oversampled.csv` rồi lại gắn thêm `sample_weight`, vì như vậy là bù mất cân bằng hai lần.
- Không xử lý `val/test`. Giữ nguyên:
  - `data/processed/baseline_standardized/val_scaled.csv`
  - `data/processed/baseline_standardized/test_scaled.csv`

## 5. Quy tắc thực dụng

- Thử `weighted` trước.
- Chỉ chuyển sang `oversampled` nếu model đó không hỗ trợ weight tốt hoặc `recall` / `F1` của lớp `0` quá kém.

## 6. Quy tắc mặc định đề xuất

- `LR` / `XGB` -> `weighted`
- `RF` -> `class_weight`
- `MLP` -> `weighted`, fallback sang `oversampled`
