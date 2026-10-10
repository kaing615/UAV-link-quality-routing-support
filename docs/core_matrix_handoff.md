# Chuyển huấn luyện core matrix sang máy khác

Máy hiện tại chạy riêng `--stages prepare`, tự thoát khi tiền xử lý xong.
Không bắt đầu huấn luyện trên máy hiện tại.

## 1. Chờ tiền xử lý hoàn tất

Kiểm tra `reports/core_matrix_bursty_fixed/prepare-20261010.log` có dòng:

```text
[OK] requested core-matrix stages completed
```

`reports/core_matrix_bursty_fixed/dataset_summary.csv` phải có 1.680 dòng dữ liệu:
210 lượt mô phỏng × 2 mục tiêu (qos, survival) × 4 horizon (1, 2, 3, 5).
Chỉ sao chép dữ liệu sau khi tiến trình đã hoàn tất.

## 2. Sao chép sang máy huấn luyện

Giữ nguyên cấu trúc bên trong thư mục dự án:

- Code từ phiên bản Git đã chứa `scripts/dataset/run_core_matrix.py`,
  `scripts/analysis/analyze_core_matrix.py` và các sửa đổi huấn luyện đi kèm.
- `data/core_matrix_bursty_fixed/` gồm cả `raw/` và `multihorizon/`.
- `reports/core_matrix_bursty_fixed/`.
- Tài liệu này.

Có thể clone/pull phiên bản code đã cập nhật từ GitHub; dữ liệu phải sao chép riêng vì được Git bỏ qua.
Máy đích tạo môi trường Python riêng. Binary ns-3 và Docker không cần cho hai bước train/report.

## 3. Chuẩn bị môi trường máy đích

Dùng Python 3.12. Tạo và kích hoạt môi trường ảo trên máy đích.
Cài PyTorch phù hợp với phần cứng theo [hướng dẫn chính thức](https://pytorch.org/get-started/locally/),
sau đó cài thư viện dự án:

```shell
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Máy nguồn dùng PyTorch `2.11.0+cpu`. Ưu tiên cùng phiên bản PyTorch trên máy đích;
chọn bản có CUDA nếu sử dụng GPU NVIDIA. Để GNN chạy GPU, lệnh kiểm tra phải trả về `True`.
GNN tự chọn CUDA khi khả dụng; logreg và XGBoost trong cấu hình hiện tại chạy CPU.

## 4. Chạy huấn luyện và báo cáo

Chạy từ thư mục gốc dự án, trong môi trường Python đã chuẩn bị:

```shell
python -u -m scripts.dataset.run_core_matrix --stages train report --raw-root data/core_matrix_bursty_fixed/raw --data-root data/core_matrix_bursty_fixed/multihorizon --output-root outputs/core_matrix_bursty_fixed --report-root reports/core_matrix_bursty_fixed
```

Lệnh giữ nguyên 10 seed, hai mục tiêu, bốn horizon và các mô hình của kế hoạch hiện tại.
Kết quả huấn luyện nằm trong `outputs/core_matrix_bursty_fixed/`;
báo cáo cuối nằm trong `reports/core_matrix_bursty_fixed/REPORT.md`.

Nếu chuyển kết quả về máy nguồn để tổng hợp, sao chép toàn bộ `outputs/core_matrix_bursty_fixed/`
và chạy lại lệnh trên với `--stages report`.
Đường dẫn tuyệt đối được lưu trong fingerprint cache; chuyển máy có thể làm cache cũ không khớp.
Không chạy lại bước prepare trên máy đích nếu chỉ cần dùng dữ liệu đã chuyển.

Máy đích chưa được kiểm tra thực tế. Giữ bản dữ liệu gốc cho tới khi xác nhận máy đích đọc được dữ liệu.
