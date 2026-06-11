# Utils

Thư mục này dành cho các hàm và công cụ tiện ích dùng chung trong dự án.

## Danh sách công cụ

### 1. `src/utils/list_run_names.py`
Công cụ liệt kê danh sách các thư mục `RUN_NAME` đang có trong thư mục `data/graph_dataset/` khớp với một mẫu (glob pattern) cho trước.

**Cách chạy trực tiếp:**
```bash
python3 -m src.utils.list_run_names
python3 -m src.utils.list_run_names 'olsr_dataset_*'
```

### 2. `scripts/utils/list_run_names.sh`
Wrapper bằng Shell Script của công cụ trên, giúp chạy nhanh từ thư mục gốc.

**Cách chạy:**
```bash
./scripts/utils/list_run_names.sh
./scripts/utils/list_run_names.sh 'olsr_dataset_*'
```

