## Thư mục Scripts (Kịch bản vận hành)

Chào mừng bạn đến với thư mục quản lý kịch bản (`scripts/`) của dự án **UAV Link Quality Routing Support**. Thư mục này chứa các kịch bản Bash shell tự động hóa toàn bộ luồng công việc: từ sinh dữ liệu mô phỏng, tiền xử lý, huấn luyện mô hình MLP, XGBoost, GNN (GraphSAGE, GAT) cho đến việc tổng hợp đánh giá kết quả.

Để bắt đầu nhanh với các lệnh vận hành hàng ngày, bạn có thể tham khảo trực tiếp tài liệu hướng dẫn: [quick\_start.md](file:///Users/dtam.21/Code/DACN/docs/quick_start.md).

## Cấu trúc Thư mục Scripts

```plaintext
scripts/
├── dataset/
│   ├── generate_batch.py              # Sinh batch dataset deterministic theo params.yaml (stage `generate` của DVC)
│   ├── run_one_dataset.sh             # Chạy mô phỏng Python &amp; tiền xử lý 1 dataset đơn lẻ
│   ├── run_many_random_datasets.sh    # Chạy tự động nhiều dataset với tham số ngẫu nhiên
│   ├── run_one_dataset_ns3.sh         # Như run_one_dataset.sh nhưng dùng simulator ns-3
│   └── run_many_random_datasets_ns3.sh # Batch dataset ngẫu nhiên bằng ns-3
├── train/
│   ├── aggregate_all.sh             # Tổng hợp metrics của cả mô hình Baseline &amp; GNN
│   ├── aggregate_baselines.sh       # Tổng hợp metrics của riêng mô hình Baseline
│   ├── baselines/
│   │   └── run_all_baselines_for_runs.sh # Huấn luyện cả 5 baseline (logreg, rf, threshold, mlp, xgb) cho nhiều run
│   ├── gnn/
│   │   ├── run_all_gnn_for_runs.sh  # Huấn luyện GNN (graphsage|gat), thêm tham số `noedge` cho ablation
│   │   ├── run_edge_sage_for_runs.sh # Huấn luyện Edge-Aware GraphSAGE (mô hình đề xuất) cho nhiều run
│   │   └── run_loro.sh              # Đánh giá Leave-One-Run-Out (cross-run) cho cả GNN &amp; baseline
│   ├── mlp/
│   │   └── run_all_mlp_for_runs.sh  # Huấn luyện mô hình MLP cho nhiều run
│   └── xgb/
│       └── run_all_xgb_for_runs.sh  # Huấn luyện mô hình XGBoost cho nhiều run
├── routing/
│   └── run_routing_for_runs.sh      # Inference + replay routing + aggregate cho nhiều run
├── utils/
│   ├── clean_data_outputs.sh        # Xóa data/output cũ (có xác nhận)
│   └── list_run_names.sh            # Tiện ích liệt kê tên các run khả dụng
└── docs/
    ├── README.md                    # Tài liệu chi tiết về pipeline sinh dữ liệu
    └── RUN_EXAMPLES.md              # Các ví dụ kịch bản chạy mô phỏng mẫu
```

> [!TIP]
> Các script này được **pipeline DVC** ([dvc.yaml](../dvc.yaml)) gọi tự động theo đúng
> thứ tự — luồng chuẩn chỉ cần `dvc repro`. Chạy tay từng script khi cần debug.

## 1\. Bộ Kịch Bản Sinh Dữ Liệu (Dataset)

Các kịch bản này điều khiển việc sinh dữ liệu topo UAV động từ simulator và thực hiện tiền xử lý đồ thị.

\> \[!NOTE\]  
\> Các kịch bản này sẽ tự động tìm kiếm môi trường Python thích hợp. Nếu đang kích hoạt Virtualenv, chúng sẽ dùng `python3` hiện tại; nếu không, chúng tự động fallback về môi trường ảo mặc định tại `simulation/.venv/bin/python`.

### [run\_one\_dataset.sh](file:///Users/dtam.21/Code/DACN/scripts/dataset/run_one_dataset.sh)

**Mục đích:** Sinh một tập dữ liệu đơn lẻ từ simulator và thực hiện tiền xử lý đồ thị + xử lý mất cân bằng dữ liệu (class imbalance) cho baseline.

**Các bước tự động chạy:**

1.  Chạy simulator sinh dữ liệu mạng UAV động (`simulation/main.py`).
2.  Chạy tiền xử lý dữ liệu đồ thị (`src/preprocessing/run_preprocessing.py`).
3.  Chuẩn hóa dữ liệu phi đồ thị cho Baseline (`src/preprocessing/non-gnn/standardize_baseline_data.py`).
4.  Xử lý mất cân bằng phân lớp dữ liệu (`src/preprocessing/non-gnn/handle_imbalance.py`).

**Cú pháp:**

**Ví dụ:**

**Tham số môi trường tùy chọn:**  
Bạn có thể truyền các biến cấu hình mô phỏng khi chạy script:

### [run\_many\_random\_datasets.sh](file:///Users/dtam.21/Code/DACN/scripts/dataset/run_many_random_datasets.sh)

*   **Mục đích:** Sinh ngẫu nhiên hàng loạt dataset để phục vụ mục đích thống kê, nghiên cứu nhiều kịch bản khác nhau.

### [run\_one\_dataset\_ns3.sh](file:///Users/dtam.21/Code/DACN/scripts/dataset/run_one_dataset_ns3.sh) / [run\_many\_random\_datasets\_ns3.sh](file:///Users/dtam.21/Code/DACN/scripts/dataset/run_many_random_datasets_ns3.sh)

*   **Mục đích:** Phiên bản **ns-3** của hai script trên — sinh dữ liệu bằng stack 802.11 + OLSR thật (RSSI sniff từ PHY, delay/loss đo từ probe UDP, fading Nakagami) thay cho công thức. Cùng interface (`RUN_NAME [SEED] [MOBILITY]`, env `SIM_NUM_UAVS`, `SIM_COMM_RANGE`, …) và cùng schema output nên các bước phía sau không đổi.
*   **Yêu cầu:** `brew install ns-3`; binary tự build ở lần chạy đầu. Chi tiết: [simulation/ns3/README.md](file:///Users/dtam.21/Code/DACN/simulation/ns3/README.md).

## 2\. Bộ Kịch Bản Huấn Luyện (Training)

Hỗ trợ huấn luyện hàng loạt 5 mô hình Baseline (Logistic Regression, Random Forest, RSSI/SNR Threshold, MLP, XGBoost) và 6 mô hình GNN (graphsage, gat, edge-sage + 3 bản ablation `-noedge`).

### [run\_all\_baselines\_for\_runs.sh](file:///Users/dtam.21/Code/DACN/scripts/train/baselines/run_all_baselines_for_runs.sh)

*   **Mục đích:** Huấn luyện **cả 5 baseline** (logreg, rf, threshold, mlp, xgb) trên các run khớp với mẫu tên chỉ định — đây là script được stage `train_baselines` của DVC gọi.
*   **Ví dụ:** `./scripts/train/baselines/run_all_baselines_for_runs.sh 'ns3big_*'`

### [run\_all\_mlp\_for\_runs.sh](file:///Users/dtam.21/Code/DACN/scripts/train/mlp/run_all_mlp_for_runs.sh)

*   **Mục đích:** Huấn luyện mô hình Baseline MLP (Multi-Layer Perceptron) trên các run khớp với mẫu tên chỉ định.

### [run\_all\_xgb\_for\_runs.sh](file:///Users/dtam.21/Code/DACN/scripts/train/xgb/run_all_xgb_for_runs.sh)

*   **Mục đích:** Huấn luyện mô hình Baseline XGBoost trên các run khớp với mẫu tên chỉ định.

### [run\_all\_gnn\_for\_runs.sh](file:///Users/dtam.21/Code/DACN/scripts/train/gnn/run_all_gnn_for_runs.sh)

*   **Mục đích:** Huấn luyện mô hình học sâu đồ thị GNN (`graphsage` hoặc `gat`) trên các run khớp với mẫu tên chỉ định.
*   **Ablation:** Thêm tham số thứ ba `noedge` để train bản bỏ edge features: `./run_all_gnn_for_runs.sh 'ns3big_*' graphsage noedge` → kết quả vào `outputs/gnn/graphsage-noedge/`.

### [run\_edge\_sage\_for\_runs.sh](file:///Users/dtam.21/Code/DACN/scripts/train/gnn/run_edge_sage_for_runs.sh)

*   **Mục đích:** Huấn luyện **Edge-Aware GraphSAGE** (mô hình đề xuất) trên các run khớp với mẫu tên chỉ định, với cấu hình chuẩn `hidden=128`, `dropout=0.3`, LR scheduler.
*   **Tùy chọn:** Override qua biến môi trường: `HIDDEN=64 DROPOUT=0.4 EPOCHS=200 ./run_edge_sage_for_runs.sh`; `NOEDGE=1` để chạy ablation `edge-sage-noedge`. Trong pipeline DVC, `HIDDEN`/`EPOCHS` được truyền từ `params.yaml` (`train.gnn_hidden`, `train.gnn_epochs`).

### [run\_loro.sh](file:///Users/dtam.21/Code/DACN/scripts/train/gnn/run_loro.sh)

*   **Mục đích:** Đánh giá khả năng **tổng quát hóa cross-run** theo giao thức Leave-One-Run-Out: với mỗi balanced run, train trên các run còn lại và test trên toàn bộ run bị giữ lại. Chạy đủ 8 model: 3 GNN (`graphsage`, `gat`, `edge-sage`) + 5 baseline (`xgb`, `mlp`, `logreg`, `rf`, `threshold`).
*   **Tùy chọn:** Đổi tập fold qua `BALANCED_IDS="007 012 035 046 077 084" ./run_loro.sh` (mặc định: 6 fold cân bằng nhãn nhất của batch ns3big seed-42, trộn 3 rwp + 3 gm).
*   **Kết quả:** `outputs/loro/<MODEL_ID>/<TEST_RUN>/`, tổng hợp bằng `python3 -m src.evaluation.aggregate_all_metrics --loro`.

## 📈 3. Bộ Kịch Bản Tổng Hợp Kết Quả (Aggregation)

### [aggregate\_all.sh](file:///Users/dtam.21/Code/DACN/scripts/train/aggregate_all.sh)

**Mục đích:** Tổng hợp file `metrics.csv` của **tất cả** các mô hình đã chạy (bao gồm Baseline MLP, XGBoost và GNN GraphSAGE, GAT) từ các run đã chỉ định.

### [aggregate\_baselines.sh](file:///Users/dtam.21/Code/DACN/scripts/train/aggregate_baselines.sh)

*   **Mục đích:** Chỉ tổng hợp `metrics.csv` của các mô hình Baseline truyền thống (MLP, XGBoost).

## 4\. Tiện Ích (Utilities)

### [list\_run\_names.sh](file:///Users/dtam.21/Code/DACN/scripts/utils/list_run_names.sh)

*   **Mục đích:** Liệt kê nhanh danh sách các tên run (`RUN_NAME`) hiện có trong thư mục dữ liệu `data/graph_dataset/` theo mẫu tìm kiếm.
*   **Ví dụ:**

## Tài Liệu Tham Khảo Thêm

*   [Hướng dẫn chạy chi tiết và lưu ý pipeline](file:///Users/dtam.21/Code/DACN/scripts/docs/README.md)
*   [Các kịch bản mẫu phục vụ vẽ biểu đồ & báo cáo](file:///Users/dtam.21/Code/DACN/scripts/docs/RUN_EXAMPLES.md)
*   [Tài liệu Quick Start tổng quan của dự án](file:///Users/dtam.21/Code/DACN/docs/quick_start.md)

```plaintext
./scripts/utils/list_run_names.sh 'batch_*'
```

```plaintext
./scripts/train/aggregate_baselines.sh 'xgb' 'exp01_*'
```

```plaintext
# Tổng hợp tất cả mô hình và tất cả run
./scripts/train/aggregate_all.sh

# Tổng hợp cụ thể
./scripts/train/aggregate_all.sh '*' 'batch_*'
```

```plaintext
./scripts/train/gnn/run_all_gnn_for_runs.sh 'batch_*' graphsage
./scripts/train/gnn/run_all_gnn_for_runs.sh 'batch_*' gat
```

```plaintext
./scripts/train/xgb/run_all_xgb_for_runs.sh 'batch_*'
```

```plaintext
./scripts/train/mlp/run_all_mlp_for_runs.sh 'batch_*'
```

```plaintext
./scripts/dataset/run_many_random_datasets.sh 10 exp01
```

```plaintext
./scripts/dataset/run_many_random_datasets.sh [COUNT] [PREFIX]
```

```plaintext
SIM_NUM_UAVS=8 SIM_COMM_RANGE=245 SIM_TIME_STEPS=120 ./scripts/dataset/run_one_dataset.sh custom_run 42 random-waypoint
```

```plaintext
./scripts/dataset/run_one_dataset.sh seed_42_rwp 42 random-waypoint
```

```plaintext
./scripts/dataset/run_one_dataset.sh RUN_NAME [SEED] [MOBILITY_MODEL]
```