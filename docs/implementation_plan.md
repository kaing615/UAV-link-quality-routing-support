## Bổ sung MLOps cho UAV-GNN Project

## Phân tích hiện trạng

| Thành phần MLOps | Trạng thái | Chi tiết |
| --- | --- | --- |
| **Data Versioning** | ✅ Đã có | DVC + Google Drive remote |
| **Pipeline as Code** | ✅ Đã có | [dvc.yaml](file:///Users/dtam.21/Code/DACN/dvc.yaml) — 6 stages, `dvc repro` tái lập toàn bộ |
| **Hyperparameter Management** | ✅ Đã có | [params.yaml](file:///Users/dtam.21/Code/DACN/params.yaml) + DVC params tracking |
| **Containerization** | ✅ Đã có | [Dockerfile](file:///Users/dtam.21/Code/DACN/Dockerfile) multi-stage (ns-3 + Python) |
| **Reproducibility** | ✅ Đã có | `dvc.lock` hash-pinned, deterministic seeding |
| **Experiment Tracking** | ❌ Chưa có | Chỉ lưu `metrics.csv` + `metadata.json` thủ công |
| **CI/CD** | ❌ Chưa có | Không có `.github/workflows/` |
| **Model Registry** | ❌ Chưa có | Model `.pt` nằm trực tiếp trong `outputs/` |
| **Monitoring / Alerting** | ❌ Chưa có | Không có data drift/model performance monitoring |
| **Automated Testing** | ❌ Chưa có | Không có unit/integration tests |

\> \[!TIP\]
\> Nền tảng DVC + Docker + params.yaml của bạn đã rất tốt cho một ĐACN. Các bước dưới đây sẽ nâng project lên mức **MLOps maturity level 1-2** (theo Google's MLOps maturity model).

## User Review Required

\> \[!IMPORTANT\]
\> Bạn cần chọn **mức độ bổ sung** phù hợp với thời gian còn lại của ĐACN. Tôi chia thành 5 tier — từ dễ nhất đến nâng cao. Hãy cho biết bạn muốn làm đến tier nào.

## Open Questions

1.  **Repo host ở đâu?** GitHub hay GitLab? (ảnh hưởng CI/CD setup)
2.  **Thời gian còn lại** để bổ sung MLOps là bao lâu?
3.  **Bạn có muốn dùng experiment tracking có UI** (MLflow/Weights & Biases) hay chỉ cần DVC-native (DVCLive + DVC Studio)?
4.  **Bạn có server/GPU** để deploy CI runner hay chỉ dùng GitHub Actions free tier?

## Proposed Changes

### Tier 1: Experiment Tracking

Hiện tại training code chỉ `print()` và lưu `metrics.csv` thủ công. Bổ sung **DVCLive** (tích hợp sẵn với DVC đã có) để tự động log metrics, params, plots mỗi epoch.

#### \[MODIFY\] [train\_gnn.py](file:///Users/dtam.21/Code/DACN/src/training/gnn/train_gnn.py)

Thêm DVCLive logging vào training loop:

```python
from dvclive import Live

with Live(dir=str(output_dir / "dvclive"), report="auto") as live:
    # Log hyperparameters
    live.log_param("model_id", model_id)
    live.log_param("hidden", args.hidden)
    live.log_param("lr", args.lr)
    live.log_param("dropout", args.dropout)
    live.log_param("num_layers", args.num_layers)
    live.log_param("use_edge_features", not args.no_edge_features)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(...)
        val_metrics, _ = evaluate_split(...)

        live.log_metric("train/loss", train_loss)
        live.log_metric("val/macro_f1", val_metrics["macro_f1"])
        live.log_metric("val/f1", val_metrics["f1"])
        live.log_metric("val/recall", val_metrics["recall"])
        live.next_step()

    # Log final test metrics
    live.log_metric("test/macro_f1", test_metrics["macro_f1"])
    live.summary["best_epoch"] = best_epoch
    live.summary["threshold"] = threshold
```

#### \[MODIFY\] [requirements.txt](file:///Users/dtam.21/Code/DACN/requirements.txt)

```diff
+dvclive&gt;=3.0.0
```

**Kết quả đạt được:**

*   Tự động sinh biểu đồ training curves (loss, F1 theo epoch)
*   So sánh các experiment bằng `dvc plots diff`
*   Nếu dùng DVC Studio (free) → dashboard web xem lịch sử experiment

### Tier 2: CI/CD Pipeline

#### \[NEW\] `.github/workflows/ci.yml`

```plaintext
name: CI Pipeline
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Install dependencies
        run: |
          pip install torch --index-url https://download.pytorch.org/whl/cpu
          pip install -r requirements.txt
          pip install pytest ruff
      - name: Lint
        run: ruff check src/ scripts/ simulation/
      - name: Unit tests
        run: pytest tests/ -v

  dvc-repro-check:
    runs-on: ubuntu-latest
    needs: lint-and-test
    steps:
      - uses: actions/checkout@v4
      - uses: iterative/setup-dvc@v1
      - name: Check pipeline is up to date
        run: dvc status  # fails nếu pipeline outdated
```

#### \[NEW\] `tests/` directory

Tạo unit tests cơ bản:

```plaintext
tests/
├── __init__.py
├── test_preprocessing.py    # test feature extraction, graph construction
├── test_models.py           # test model forward pass shapes
└── test_evaluation.py       # test metric aggregation logic
```

### Tier 3: Model Registry & DVC Experiments

#### Sử dụng `dvc exp` để quản lý experiment

```plaintext
# Chạy experiment với param khác nhau
dvc exp run -S train.gnn_hidden=256 -S train.gnn_epochs=500

# So sánh experiments
dvc exp show --sort-by val/macro_f1

# Promote best experiment vào git
dvc exp apply <exp-name>
git add . &amp;&amp; git commit -m "best: hidden=256, F1=0.92"
```

#### \[NEW\] `scripts/mlops/promote_model.sh`

Script tự động tag best model:

```plaintext
#!/bin/bash
# Tag phiên bản model tốt nhất
BEST_MODEL=$1
VERSION=$(date +%Y%m%d_%H%M%S)
git tag -a "model-v${VERSION}" -m "Promote ${BEST_MODEL}, macro_f1=${2}"
dvc push
git push origin "model-v${VERSION}"
```

### Tier 4: Data Quality & Validation

#### \[NEW\] `src/validation/data_quality.py`

Kiểm tra data quality trước khi train:

```python
"""Data quality checks chạy trước mỗi training run."""

def check_feature_distribution(graph_dataset_path):
    """Detect feature drift giữa các runs."""
    # - Kiểm tra range của RSSI, SNR, distance
    # - Phát hiện NaN/Inf
    # - Cảnh báo nếu class imbalance &gt; threshold

def check_graph_integrity(graph_dataset_path):
    """Validate graph structure."""
    # - Mỗi graph phải có ≥ 2 nodes
    # - edge_index phải consistent với node count
    # - Không có self-loops trừ khi intentional
```

#### \[MODIFY\] [dvc.yaml](file:///Users/dtam.21/Code/DACN/dvc.yaml)

Thêm stage `validate` giữa `generate` và `train_*`:

```plaintext
  validate:
    cmd: python -m src.validation.data_quality
    deps: [data/graph_dataset/, src/validation/]
    metrics: [reports/data_quality.json]
```

### Tier 5: Advanced

Các thành phần nâng cao nếu có thời gian:

| Thành phần | Công cụ gợi ý | Mục đích |
| --- | --- | --- |
| **Hyperparameter tuning** | Optuna + DVCLive | Tự động tìm best hyperparams |
| **Model serving API** | FastAPI + Docker | Expose model qua REST endpoint |
| **Monitoring dashboard** | Grafana + Prometheus | Track inference latency, model drift |
| **Feature store** | DVC + custom pipeline | Version và share features |

## Lộ trình đề xuất (Roadmap)

```plaintext
flowchart LR
    T1["Tier 1\nExperiment Tracking\n(DVCLive)"] --&gt; T2["Tier 2\nCI/CD\n(GitHub Actions)"]
    T2 --&gt; T3["Tier 3\nModel Registry\n(dvc exp + git tag)"]
    T3 --&gt; T4["Tier 4\nData Validation"]
    T4 --&gt; T5["Tier 5\nAdvanced\n(Serving, Monitoring)"]

    style T1 fill:#22c55e,color:#fff
    style T2 fill:#3b82f6,color:#fff
    style T3 fill:#8b5cf6,color:#fff
    style T4 fill:#f59e0b,color:#000
    style T5 fill:#6b7280,color:#fff
```

\> \[!IMPORTANT\]
\> **Đề xuất cho ĐACN:** Làm **Tier 1 + Tier 2** là đủ để thể hiện tư duy MLOps trong báo cáo. Tier 3 là bonus đẹp. Tier 4-5 chỉ nên làm nếu còn thời gian.

## Verification Plan

### Automated Tests

*   `dvc repro` — toàn bộ pipeline chạy thành công
*   `pytest tests/ -v` — unit tests pass
*   `ruff check src/` — không có linting errors
*   `dvc exp show` — experiments được track đúng

### Manual Verification

*   Kiểm tra DVCLive reports sinh ra trong `outputs/gnn/*/dvclive/`
*   Xem biểu đồ training curves bằng `dvc plots show`
*   (Nếu có CI) Pull Request trigger workflow thành công trên GitHub Actions
