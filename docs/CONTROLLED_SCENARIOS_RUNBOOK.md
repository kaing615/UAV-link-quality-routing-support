# Runbook: sinh 400 run Fast/Sparse/Dense bằng ns-3

## 1. Mục tiêu và đầu ra bắt buộc

Mục 1--14 dành cho worker sinh dữ liệu và gửi về máy tổng hợp. Nếu được giao
chạy luôn toàn bộ thực nghiệm, worker dùng launcher tại Mục 15 sau khi đã có đủ
400 run; không cần tự ghép các lệnh train và routing riêng lẻ.

Ma trận cuối cùng gồm **100 seed ghép cặp**, mỗi seed sinh đủ bốn scenario:

| Scenario | Run | UAV | Vùng bay | Range | Tốc độ |
|---|---:|---:|---:|---:|---:|
| `baseline` | 100 | 20 | 800×800 m | 240 m | 3–8 m/s |
| `fast` | 100 | 20 | 800×800 m | 240 m | 12–20 m/s |
| `sparse` | 100 | 20 | 800×800 m | 160 m | 3–8 m/s |
| `dense` | 100 | 20 | 800×800 m | 300 m | 3–8 m/s |

Tổng cộng: **400 run**. Mỗi run có 120 snapshot. Seed lẻ dùng Random Waypoint,
seed chẵn dùng Gauss–Markov; bốn scenario của cùng một seed luôn dùng cùng
mobility để có thể so sánh ghép cặp.

## 2. Yêu cầu máy chạy

- Ubuntu/Linux hoặc GitHub Codespaces.
- Docker hoạt động: `docker version` không báo lỗi daemon.
- Khuyến nghị tối thiểu 4 CPU, 16 GB RAM, 64 GB storage.
- Repository phải chứa phiên bản mới nhất của `Dockerfile`, `simulation/ns3/`,
  `scripts/dataset/run_controlled_scenarios.py` và
  `scripts/analysis/summarize_controlled_scenarios.py`.
- Không cần tải bộ `ns3big_*` cũ. Tuy nhiên, nếu nhóm đã đưa một phần
  `stress_*` lên DVC thì phải tải và giải nén phần đó trước khi chạy để runner
  nhận ra các run hoàn chỉnh và tự bỏ qua.

Trước khi chạy:

```bash
cd /workspaces/UAV-link-quality-routing-support
git pull origin main
git status --short
df -h .
docker version
```

Nếu tên thư mục Codespace khác, dùng `pwd` rồi `cd` vào thư mục chứa
`Dockerfile`.

### 2.1. Tải các controlled run đã có từ DVC

Nếu thư mục `deliverables/` có các metadata
`controlled_*_data.tar.gz.dvc`, tải archive từ remote `storage`:

```bash
dvc pull -r storage deliverables/controlled_*_data.tar.gz.dvc
```

Kiểm tra checksum tương ứng nếu có, sau đó giải nén tại thư mục gốc repository:

```bash
for archive in deliverables/controlled_*_data.tar.gz; do
  [ -f "$archive" ] || continue
  checksum="${archive%_data.tar.gz}_SHA256SUMS.txt"
  [ -f "$checksum" ] && sha256sum -c "$checksum"
  tar -xzf "$archive"
done
```

`dvc pull` chỉ tải archive vào `deliverables/`; bước giải nén mới đưa các file
vào `data/raw_snapshots/` và `data/graph_dataset/`. Sau đó vẫn chạy đủ dải index
được giao. Runner sẽ in `SKIPPED` cho run đã có đủ raw data và ba split graph,
và chỉ sinh các run còn thiếu hoặc chưa hoàn chỉnh. Nếu không có metadata
controlled `.dvc`, bỏ qua Mục 2.1 và chạy bình thường.

## 3. Build môi trường

```bash
docker build -t uav-ns3-runner .
```

Build có thể lâu ở lần đầu vì phải tải và biên dịch ns-3, PyTorch và các thư
viện Python. Không hủy nếu terminal vẫn đang có log mới.

Kiểm tra image:

```bash
docker image inspect uav-ns3-runner --format '{{.Id}}'
```

## 4. Dry-run bắt buộc

Dry-run không sinh dữ liệu, chỉ xác nhận đúng ma trận 400 job:

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  uav-ns3-runner \
  python -m scripts.dataset.run_controlled_scenarios \
    --start-index 1 \
    --end-index 100 \
    --dry-run
```

Kết quả phải ghi:

```text
total_jobs=400
baseline    100
dense       100
fast        100
sparse      100
```

Nếu không đúng 400 thì dừng, không chạy thật.

## 5. Chạy thử một seed trước

Lệnh sau sinh 4 run (`baseline`, `fast`, `sparse`, `dense`) cho seed đầu tiên:

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  uav-ns3-runner \
  bash -lc '
    set -euo pipefail
    mkdir -p simulation/ns3/build
    cp /app/simulation/ns3/build/uav-olsr-dataset simulation/ns3/build/
    chmod +x simulation/ns3/build/uav-olsr-dataset
    python -m scripts.dataset.run_controlled_scenarios \
      --start-index 1 --end-index 1
  '
```

Kiểm tra bốn run:

```bash
find data/raw_snapshots -maxdepth 2 \
  -path '*/stress_*_s60001/scenario.json' | sort

find data/graph_dataset -maxdepth 3 \
  -path '*/stress_*_s60001/graph_dataset/test.pt' | sort
```

Mỗi lệnh phải in đúng bốn đường dẫn. Kiểm tra dung lượng và thời gian trong
manifest trước khi quyết định số máy cần dùng:

```bash
du -sh data/raw_snapshots/stress_*_s60001
du -sh data/graph_dataset/stress_*_s60001
cat reports/controlled_scenarios/workers/worker_001_001.csv
```

## 6. Chạy toàn bộ trên một máy

Lệnh có resume. Khi chạy lại, run có đủ raw CSV và train/val/test graph sẽ được
đánh dấu `SKIPPED`.

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  uav-ns3-runner \
  bash -lc '
    set -euo pipefail
    mkdir -p simulation/ns3/build
    cp /app/simulation/ns3/build/uav-olsr-dataset simulation/ns3/build/
    chmod +x simulation/ns3/build/uav-olsr-dataset
    python -m scripts.dataset.run_controlled_scenarios \
      --start-index 1 --end-index 100
  '
```

Không đóng hoặc xóa Codespace khi job đang chạy. Nếu job dừng, chạy lại đúng
lệnh trên; không xóa output đã có.

## 7. Chia nhiều máy

Chia theo index, không chia theo scenario. Mỗi máy luôn sinh đủ bốn scenario
cho các seed được giao.

| Máy | `--start-index` | `--end-index` | Tổng run |
|---|---:|---:|---:|
| 1 | 1 | 25 | 100 |
| 2 | 26 | 50 | 100 |
| 3 | 51 | 75 | 100 |
| 4 | 76 | 100 | 100 |

Mỗi máy dùng lệnh ở Mục 6 và thay hai index. Ví dụ máy 2:

```bash
python -m scripts.dataset.run_controlled_scenarios \
  --start-index 26 --end-index 50
```

Không cho hai máy chạy trùng index. Tên run chứa scenario, mobility và seed nên
các dải không trùng có thể giải nén vào cùng một repository.

## 8. Theo dõi và xử lý lỗi

Manifest của từng worker được cập nhật sau mỗi run:

```text
reports/controlled_scenarios/workers/worker_START_END.csv
```

Xem tiến độ gần nhất:

```bash
tail -n 20 reports/controlled_scenarios/workers/worker_001_100.csv
```

Đếm run hoàn chỉnh hiện tại:

```bash
for scenario in baseline fast sparse dense; do
  raw=$(find data/raw_snapshots -maxdepth 2 \
    -path "*/stress_${scenario}_*/scenario.json" | wc -l)
  graph=$(find data/graph_dataset -maxdepth 3 \
    -path "*/stress_${scenario}_*/graph_dataset/test.pt" | wc -l)
  echo "${scenario}: raw=${raw}/100 graph=${graph}/100"
done
```

Nếu một run báo `FAILED` hoặc `INCOMPLETE`:

1. Giữ nguyên thư mục output để điều tra.
2. Ghi lại tên run và phần stack trace cuối terminal.
3. Kiểm tra `df -h .` và `docker version`.
4. Chạy lại đúng dải index. Runner sẽ bỏ qua run hoàn chỉnh.
5. Không tự sửa CSV, không đổi seed và không đổi tham số scenario.

## 9. Đóng gói dữ liệu từ từng worker

Mỗi worker chỉ đóng gói dải seed mình đã chạy. Ví dụ máy 1 chạy index 1–25,
tương ứng seed 60001–60025:

```bash
mkdir -p deliverables

find data/raw_snapshots data/graph_dataset -mindepth 1 -maxdepth 1 \
  -type d -name 'stress_*_s*' | while IFS= read -r path; do
    seed=${path##*_s}
    if [ "$seed" -ge 60001 ] && [ "$seed" -le 60025 ]; then
      printf '%s\n' "$path"
    fi
  done | sort > deliverables/worker_001_025_files.txt

test "$(wc -l < deliverables/worker_001_025_files.txt)" -eq 200

tar -czf deliverables/controlled_worker_001_025_data.tar.gz \
  -T deliverables/worker_001_025_files.txt

cp reports/controlled_scenarios/workers/worker_001_025.csv deliverables/

sha256sum deliverables/controlled_worker_001_025_data.tar.gz \
  > deliverables/controlled_worker_001_025_SHA256SUMS.txt
```

Nếu câu lệnh chọn seed không in đúng thư mục, dừng và kiểm tra bằng:

```bash
find data/raw_snapshots data/graph_dataset -maxdepth 1 -type d \
  -name 'stress_*' | sort | head
```

Con số `200` ở lệnh `test` là 25 index × 4 scenario × 2 thư mục dữ liệu
(`raw_snapshots` và `graph_dataset`). Với worker khác, thay dải seed, tên file và
số thư mục mong đợi tương ứng. Gửi file `.tar.gz`, manifest `.csv`, danh sách
`_files.txt` và checksum `.txt`. Không chỉ gửi report; nhóm viết paper cần cả
raw snapshot và graph dataset để chạy mô hình.

## 10. Ghép dữ liệu từ nhiều worker

Trên máy tổng hợp, đặt các archive vào `incoming/`, rồi chạy:

```bash
for archive in incoming/controlled_worker_*_data.tar.gz; do
  tar -xzf "$archive"
done
```

Kiểm tra không thiếu/trùng:

```bash
for scenario in baseline fast sparse dense; do
  echo -n "${scenario}: "
  find data/raw_snapshots -maxdepth 2 \
    -path "*/stress_${scenario}_*/scenario.json" | wc -l
done
```

Mỗi dòng phải bằng `100`.

## 11. Sinh report cuối

Chạy report generator sau khi đã ghép đủ dữ liệu:

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  uav-ns3-runner \
  python -m scripts.analysis.summarize_controlled_scenarios \
    --expected-per-scenario 100 \
    --fail-on-check
```

Report kiểm tra:

- 100 run hoàn chỉnh cho từng scenario;
- 100 bộ seed–mobility có đủ bốn scenario;
- tốc độ Fast lớn hơn Baseline;
- topology churn của Fast lớn hơn Baseline;
- mean degree của Sparse thấp hơn Baseline;
- mean degree của Dense cao hơn Baseline;
- connected-edge rate và positive label ratio theo scenario.

Đầu ra:

```text
reports/controlled_scenarios/
├── REPORT.md
├── run_inventory.csv
├── scenario_summary.csv
├── manipulation_checks.csv
├── figures/
│   ├── scenario_manipulation_checks.png
│   └── scenario_manipulation_checks.pdf
└── workers/
    └── worker_*.csv
```

Nếu command kết thúc với `[CONTROLLED SCENARIOS] FAIL`, không được tuyên bố bộ
dữ liệu hoàn tất. Gửi `REPORT.md`, `manipulation_checks.csv` và log lỗi cho
người phụ trách code.

## 12. Data-quality check cuối

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  uav-ns3-runner \
  python -m src.validation.data_quality \
    --data-dir data/graph_dataset \
    --output reports/controlled_scenarios/data_quality.json \
    --fail-on-error
```

Warning về class imbalance được phép giữ lại và phải gửi kèm. Error về missing
split, NaN, Inf hoặc invalid edge index phải được xử lý trước khi bàn giao.

## 13. Gói bàn giao cuối cùng

```bash
mkdir -p deliverables

tar -czf deliverables/controlled_scenarios_reports.tar.gz \
  reports/controlled_scenarios

sha256sum deliverables/controlled_scenarios_reports.tar.gz \
  > deliverables/controlled_scenarios_reports_SHA256SUMS.txt
```

Người chạy gửi:

1. Bốn archive dữ liệu hoặc đường dẫn cloud chứa toàn bộ 400 run.
2. `controlled_scenarios_reports.tar.gz`.
3. Các file `SHA256SUMS.txt`.
4. Commit SHA đã dùng: `git rev-parse HEAD`.
5. Docker image ID: `docker image inspect uav-ns3-runner --format '{{.Id}}'`.

## 14. Đưa dữ liệu đã nghiệm thu lên DVC remote

Chỉ thực hiện bước này trên **máy tổng hợp**, sau khi report và data-quality đều
đạt. Không để từng worker tự `dvc push`, vì mỗi worker chỉ giữ một phần dataset.

Các thư mục `data/raw_snapshots/` và `data/graph_dataset/` đang là output của
stage `generate` dành cho bộ dữ liệu nền. Vì controlled scenarios được chạy bằng
runner riêng, không dùng `dvc commit generate` và không `dvc add` đè lên hai thư
mục này. Thay vào đó, DVC theo dõi các archive đã kiểm tra checksum.

Đặt đủ bốn archive worker vào `deliverables/`, rồi kiểm tra:

```bash
test "$(find deliverables -maxdepth 1 -type f \
  -name 'controlled_worker_*_data.tar.gz' | wc -l)" -eq 4

sha256sum -c deliverables/controlled_worker_001_025_SHA256SUMS.txt
sha256sum -c deliverables/controlled_worker_026_050_SHA256SUMS.txt
sha256sum -c deliverables/controlled_worker_051_075_SHA256SUMS.txt
sha256sum -c deliverables/controlled_worker_076_100_SHA256SUMS.txt
sha256sum -c deliverables/controlled_scenarios_reports_SHA256SUMS.txt
```

Máy tổng hợp phải có quyền ghi vào remote `storage` đã cấu hình trong
`.dvc/config`. Không ghi credential vào Git. Xác nhận remote và trạng thái đăng
nhập trước khi tạo metadata:

```bash
dvc remote list
dvc remote default
```

Đưa bốn archive dữ liệu và archive report vào DVC cache:

```bash
dvc add deliverables/controlled_worker_*_data.tar.gz \
  deliverables/controlled_scenarios_reports.tar.gz

git add deliverables/*.dvc deliverables/*SHA256SUMS.txt \
  deliverables/.gitignore
git commit -m "data: version controlled scenario dataset"
```

Đẩy cache lên Google Drive rồi mới đẩy metadata Git:

```bash
dvc push -r storage deliverables/*.dvc
git push origin main
```

`dvc push` chỉ tải các object đã có trong DVC cache; nó không tự đọc và tải mọi
file mới trong workspace. Vì vậy `dvc add` ở trên là bước bắt buộc. Nếu push báo
lỗi xác thực Google Drive, dừng lại và nhờ chủ remote cấp quyền; không đổi URL
remote và không commit file credential.

Kiểm tra từ một clone/Codespace khác:

```bash
dvc pull deliverables/*.dvc
sha256sum -c deliverables/controlled_worker_001_025_SHA256SUMS.txt
tar -tzf deliverables/controlled_worker_001_025_data.tar.gz | head
```

Sau khi kiểm tra thành công, người viết paper chỉ cần clone đúng commit, chạy
`dvc pull deliverables/*.dvc`, giải nén bốn archive rồi thực hiện Mục 10--12.

## 15. Chạy toàn bộ flow nghiên cứu trên máy tổng hợp

Chỉ một máy có đủ 400 run mới chạy mục này. Launcher thực hiện tuần tự:

```text
controlled ns-3 → data quality → multi-horizon QoS/survival
→ persistence/logreg/XGBoost/Edge-SAGE → LORO/cross-mobility/ablation
→ routing replay → inference benchmark → closed-loop ns-3
→ hình và bảng cho paper
```

Xem trước tám pha mà không chạy tính toán:

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace \
  uav-ns3-runner bash scripts/run_controlled_paper_pipeline.sh --dry-run
```

Chạy toàn bộ và lưu log:

```bash
mkdir -p reports/controlled_paper
set -o pipefail
docker run --rm -v "$PWD:/workspace" -w /workspace \
  uav-ns3-runner bash scripts/run_controlled_paper_pipeline.sh \
  2>&1 | tee reports/controlled_paper/pipeline.log
```

Launcher dùng riêng `data/multihorizon_controlled/`,
`outputs/multihorizon_controlled/` và `reports/controlled_paper/`, vì vậy không
trộn kết quả với pilot `ns3big_*`. Dữ liệu vẫn gồm đủ 100 seed cho mỗi scenario
(400 run) tại từng mốc `t+1`, `t+2`, `t+3`, `t+5`. Để tránh hàng nghìn lần
huấn luyện GNN không cần thiết, mặc định within-run dùng 10 run/scenario, LORO
giữ ra 5 fold/scenario, ablation dùng 10 run/scenario và closed-loop dùng 10
run baseline. Các fold LORO vẫn huấn luyện trên toàn bộ run còn lại. GNN chạy
200 epoch với patience 20. Run đại diện được chọn theo seed tăng dần; seed
lẻ/chẵn của bộ controlled xen kẽ Random Waypoint và Gauss–Markov.

Có thể tăng hoặc giảm khối lượng huấn luyện mà không sinh lại dữ liệu:

```bash
export WITHIN_RUNS_PER_SCENARIO=10
export LORO_FOLDS_PER_SCENARIO=5
export ABLATION_RUNS_PER_SCENARIO=10
export CLOSED_LOOP_RUNS=10
export GNN_EPOCHS=200 GNN_PATIENCE=20
set -o pipefail
docker run --rm \
  -e WITHIN_RUNS_PER_SCENARIO \
  -e LORO_FOLDS_PER_SCENARIO \
  -e ABLATION_RUNS_PER_SCENARIO \
  -e CLOSED_LOOP_RUNS -e GNN_EPOCHS -e GNN_PATIENCE \
  -v "$PWD:/workspace" -w /workspace \
  uav-ns3-runner bash scripts/run_controlled_paper_pipeline.sh \
  2>&1 | tee reports/controlled_paper/pipeline.log
```

Nếu bị ngắt, chạy lại đúng lệnh. Các bước sinh controlled data, dựng
multi-horizon, huấn luyện, routing, inference và closed-loop đều nhận diện output
đã hoàn chỉnh để tiếp tục; các bước tổng hợp nhẹ có thể chạy lại. Hoàn tất khi
launcher in:

```text
[OK] controlled paper pipeline completed: reports/controlled_paper
```

Không đặt ba giới hạn theo scenario thành 100 trừ khi thật sự có cụm tính toán:
cấu hình đó tạo lại hàng nghìn lần huấn luyện Edge-SAGE nhưng không tạo thêm dữ
liệu. Máy CPU Codespaces vẫn không phù hợp cho flow này; dùng máy có GPU bằng
cách thêm `--gpus all` vào `docker run` nếu Docker/NVIDIA đã được cấu hình.

## 16. Phân biệt publish paper và promote model serving

Đây là hai việc độc lập:

```text
Data/kết quả paper → dvc add + dvc push thủ công
Model chạy API      → promote_model.sh tự dvc add + dvc push model
```

### 16.1. Publish dữ liệu và kết quả paper

Sau khi launcher in `[OK]`, đóng gói data, outputs và reports thành
`deliverables/controlled_paper_full.tar.gz`, rồi chạy:

```bash
dvc add deliverables/controlled_paper_full.tar.gz
dvc push -r storage deliverables/controlled_paper_full.tar.gz.dvc

git add deliverables/controlled_paper_full.tar.gz.dvc \
  deliverables/controlled_paper_full_SHA256SUMS.txt \
  deliverables/controlled_paper_full_files.txt \
  deliverables/.gitignore
git commit -m "data: publish controlled paper experiment"
git push origin main
```

Đây là bước bàn giao kết quả paper. Không chạy `promote_model.sh` chỉ để upload
data.

### 16.2. Promote model serving (không bắt buộc cho paper)

Chỉ thực hiện sau khi đã phân tích metric và chọn một thư mục model có
`best_model.pt` cùng `metadata.json`:

```bash
bash scripts/mlops/promote_model.sh \
  outputs/multihorizon_controlled/edge-sage/survival/k3/<RUN_NAME> \
  <MACRO_F1>
```

Script tự tạo artifact DVC nhỏ dành cho serving, push artifact, commit pointer
và tạo tag `model-v*` để kích hoạt workflow build image. Không cần `dvc push`
model thêm lần nữa. Nếu chỉ viết paper, bỏ qua toàn bộ Mục 16.2.

## 17. Tiêu chí nghiệm thu

Chỉ xem là hoàn tất khi:

- `baseline`, `fast`, `sparse`, `dense` đều có `raw=100/100` và
  `graph=100/100`;
- `REPORT.md` có `Overall status: PASS`;
- data-quality có `total_errors = 0`;
- archive giải nén được và checksum hợp lệ;
- manifest không có trạng thái `FAILED` hoặc `INCOMPLETE`;
- `dvc push -r storage deliverables/*.dvc` hoàn tất trên máy tổng hợp;
- metadata `.dvc` đã được commit và push lên Git;
- người nhận có cả raw data, graph data và report, không chỉ có hình.
