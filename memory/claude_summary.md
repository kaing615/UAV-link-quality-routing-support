# Tài liệu Tóm tắt Dự án cho AI Assistant (Claude Summary)

Tài liệu này tóm tắt toàn bộ trạng thái dự án **UAV Link Quality Routing Support** tính đến thời điểm hiện tại (sau khi tích hợp nhánh `feature/gnn` vào `main`). Tài liệu giúp các AI assistant (Claude) sau này nắm bắt nhanh tiến độ, kiến trúc và các bước cần thực hiện tiếp theo.

---

## 📌 1. Tổng Quan Dự Án

*   **Mục tiêu:** Dự đoán chất lượng và độ ổn định của liên kết (link stability) trong mạng mesh UAV động nhằm hỗ trợ các giao thức định tuyến (như OLSR).
*   **Phương pháp:** 
    1. Sử dụng simulator mạng UAV động chạy trên các mô hình di chuyển (Mobility Models) như **Random Waypoint (RWP)** và **Gauss-Markov (GM)**.
    2. So sánh giữa các phương pháp học máy truyền thống (**MLP**, **XGBoost**) và các phương pháp học sâu trên đồ thị (**GraphSAGE**, **GAT**).

---

## 🛠️ 2. Các Công Việc Đã Hoàn Thành (Completed)

### 🔹 Luồng dữ liệu (Data Pipeline)
*   **Simulator:** Sinh dữ liệu mạng UAV động (`simulation/main.py`), xuất dữ liệu dưới dạng CSV (`nodes.csv`, `edges.csv`, `traffic_log.csv`).
*   **Tiền xử lý & Xử lý mất cân bằng:**
    *   Tự động tiền xử lý dữ liệu đồ thị đồ sộ (`src/preprocessing/run_preprocessing.py`).
    *   Chuẩn hóa dữ liệu phi đồ thị (`src/preprocessing/non-gnn/standardize_baseline_data.py`).
    *   Giải quyết mất cân bằng phân lớp dữ liệu phi đồ thị (`src/preprocessing/non-gnn/handle_imbalance.py`).
    *   Chuyển đổi dữ liệu thành các file PyTorch Geometric dạng `.pt` (`src/preprocessing/gnn/build_graph_dataset.py`) cho huấn luyện GNN.

### 🔹 Mô hình hóa & Huấn luyện (Modeling & Training)
*   **Mô hình Baseline (Học máy truyền thống):**
    *   Tích hợp kịch bản MLP (`src/training/baselines/mlp_baseline.py`) và XGBoost (`src/training/baselines/xgb_baseline.py`).
*   **Mô hình GNN (Graph Deep Learning):**
    *   Xây dựng mô hình phân lớp cạnh đồ thị trong [src/models/gnn/edge_gnn.py](file:///Users/dtam.21/Code/DACN/src/models/gnn/edge_gnn.py) gồm:
        *   **GraphSAGEEdgeClassifier**: Trích xuất đặc trưng nút bằng `SAGEConv`, chuẩn hóa qua `BatchNorm1d` (rất quan trọng do tọa độ 0-1000m và RSSI -90dB có độ lệch scale rất lớn), giải mã cạnh bằng cách nối vector (concat) các nút đầu/cuối cùng đặc trưng cạnh (`labeled_edge_attr`) rồi đưa qua một mạng MLP phân loại.
        *   **GATEdgeClassifier**: Kiến trúc tương tự nhưng dùng cơ chế Attention (`GATConv`).
    *   Hoàn thiện mã nguồn huấn luyện [src/training/gnn/train_gnn.py](file:///Users/dtam.21/Code/DACN/src/training/gnn/train_gnn.py) hỗ trợ Early Stopping lưu mô hình tốt nhất, ghi nhận và xuất metrics (`metrics.csv`) cũng như kết quả dự đoán.

### 🔹 Tổng hợp & Đánh giá (Evaluation & Aggregation)
*   Tích hợp kịch bản Bash tự động hóa chạy hàng loạt:
    *   [run_all_gnn_for_runs.sh](file:///Users/dtam.21/Code/DACN/scripts/train/gnn/run_all_gnn_for_runs.sh): Huấn luyện GNN (GraphSAGE/GAT) cho nhiều run.
    *   [run_all_mlp_for_runs.sh](file:///Users/dtam.21/Code/DACN/scripts/train/mlp/run_all_mlp_for_runs.sh) & [run_all_xgb_for_runs.sh](file:///Users/dtam.21/Code/DACN/scripts/train/xgb/run_all_xgb_for_runs.sh): Huấn luyện baseline cho nhiều run.
*   **Tổng hợp kết quả:**
    *   [src/evaluation/aggregate_all_metrics.py](file:///Users/dtam.21/Code/DACN/src/evaluation/aggregate_all_metrics.py) kết hợp với kịch bản [aggregate_all.sh](file:///Users/dtam.21/Code/DACN/scripts/train/aggregate_all.sh) để gom file `metrics.csv` của cả Baseline và GNN.
*   **Trực quan hóa:**
    *   [src/evaluation/plot_comparison.py](file:///Users/dtam.21/Code/DACN/src/evaluation/plot_comparison.py) tự động vẽ biểu đồ dạng cột so sánh trực tiếp hiệu năng (`Accuracy`, `F1`, `Macro F1`, `Recall`) giữa các mô hình trên tập kiểm thử (Test Split).

---

## 📌 3. Số Liệu Đánh Giá Hiện Tại (Key Numbers)

Sau khi tích hợp đầy đủ 7 đặc trưng cạnh (bao gồm cả `snr` và `throughput`):

### 1. Tính trung bình trên cả 10 run (bao gồm cả các run dị biệt):
*   **XGBoost Baseline:** **0.7504** F1-Macro
*   **MLP Baseline:** **0.6487** F1-Macro
*   **GAT (GNN mới):** **0.6473** F1-Macro
*   **GraphSAGE (GNN cũ):** **0.6194** F1-Macro

### 2. Tính trung bình trên các run cân bằng tốt (well-balanced runs: 01, 04, 05, 07):
*   **XGBoost Baseline:** **0.9095** F1-Macro
*   **MLP Baseline:** **0.8816** F1-Macro
*   **GAT (GNN mới):** **0.8584** F1-Macro (Tiệm cận rất sát với MLP và XGBoost!)
*   **GraphSAGE (GNN cũ):** **0.8045** F1-Macro
*   *Nhận xét:* Trên tập Validation của các run cân bằng, **GAT đạt 0.9068** F1-Macro, bám đuổi rất sát với MLP (0.9280) và XGBoost (0.9253). Thể hiện sự vượt trội so với GraphSAGE cũ.

---

## ⚠️ 4. Các Công Việc Chưa Hoàn Thành & Hướng Cải Tiến (Pending/Ongoing)

> [!WARNING]
> Đây là các vấn đề trọng tâm cần tập trung giải quyết trong các phiên làm việc tiếp theo:

1.  **Thiếu Đặc Trưng Cạnh Trong GNN (Feature Gap):**
    *   *Hiện trạng:* Các mô hình Baseline sử dụng 7 đặc trưng cạnh (bao gồm cả `snr` và `throughput`). GNN trước đây chỉ dùng 5 đặc trưng.
    *   *Đã sửa một phần:* Đã thêm `snr` vào `EDGE_FEATURES` trong [build_graph_dataset.py](file:///Users/dtam.21/Code/DACN/src/preprocessing/gnn/build_graph_dataset.py) nâng số đặc trưng lên 6.
    *   *Cần làm:* 
        *   Tái sinh toàn bộ tập dữ liệu PyG dạng `.pt` bằng cách chạy lại tiền xử lý để nhúng đặc trưng `snr` mới cập nhật này vào các tensor.
        *   Tìm cách bổ sung đặc trưng `throughput` vào dữ liệu đồ thị để GNN có đầy đủ 7 đặc trưng như Baseline nhằm đảm bảo tính công bằng khi so sánh.
2.  **Đánh giá mô hình GAT (Graph Attention Network):**
    *   *Hiện trạng:* Cấu trúc mạng GAT đã được xây dựng sẵn trong code (`GATEdgeClassifier`), nhưng chưa từng được chạy huấn luyện thực tế và đánh giá hiệu năng so sánh.
    *   *Cần làm:* Thử nghiệm huấn luyện bằng cách truyền tham số `--model gat`.
3.  **Xử lý các Run Dị Biệt (Degenerate Runs):**
    *   *Hiện trạng:* Một số run sinh dữ liệu bị lệch phân lớp cực đoan (Ví dụ: Run 03 tập Val có 0% nhãn 1, Run 04 tập Val có 100% nhãn 1, Run 02 có 99% nhãn 1).
    *   *Ảnh hưởng:* Điều này khiến lớp `BatchNorm1d` trong GNN bị lỗi hoặc không hội tụ tốt vì tính toán phương sai bị suy biến khi chỉ có 1 nhãn. Các mô hình ML tĩnh như XGBoost ít bị ảnh hưởng bởi lỗi toán học này hơn.
    *   *Cần làm:* Viết cơ chế lọc bỏ các run dị biệt này trước khi huấn luyện GNN, hoặc điều chỉnh chiến lược tính toán trọng số loss weight và cấu hình `BatchNorm` khi huấn luyện.
4.  **Tối Ưu Siêu Tham Số GNN (Hyperparameter Tuning):**
    *   Tinh chỉnh các thông số: tốc độ học (`--lr`), suy hao trọng số (`--weight-decay`), tỷ lệ dropout (`--dropout`), số tầng GNN (`--num-layers`), số chiều ẩn (`--hidden`) để thu hẹp khoảng cách F1-Macro 0.12 với XGBoost.

---

## 🚀 5. Hướng Dẫn Các Lệnh Vận Hành

### 💡 Bước 1: Tiền xử lý dữ liệu đồ thị (nếu cần cập nhật feature `snr`)
```bash
# Ví dụ chạy tiền xử lý cho một run cụ thể
python3 -m src.preprocessing.run_preprocessing \
  --nodes data/raw_snapshots/<RUN_NAME>/nodes.csv \
  --edges data/raw_snapshots/<RUN_NAME>/edges.csv \
  --output-root data/graph_dataset/<RUN_NAME>
```

### 💡 Bước 2: Huấn luyện hàng loạt mô hình
```bash
# Huấn luyện GraphSAGE
./scripts/train/gnn/run_all_gnn_for_runs.sh 'olsr_dataset_*' graphsage

# Huấn luyện GAT
./scripts/train/gnn/run_all_gnn_for_runs.sh 'olsr_dataset_*' gat

# Huấn luyện MLP & XGBoost
./scripts/train/mlp/run_all_mlp_for_runs.sh 'olsr_dataset_*'
./scripts/train/xgb/run_all_xgb_for_runs.sh 'olsr_dataset_*'
```

### 💡 Bước 3: Tổng hợp kết quả & vẽ biểu đồ so sánh
```bash
# Tổng hợp toàn bộ metrics của các mô hình
./scripts/train/aggregate_all.sh

# Vẽ biểu đồ cột so sánh
python3 -m src.evaluation.plot_comparison
```
Biểu đồ so sánh sẽ được lưu tại: `outputs/aggregates/all_models/model_comparison.png`.
