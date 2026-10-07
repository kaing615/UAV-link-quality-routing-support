# Audit lại đồ án - 2026-10-07

**Phạm vi:** working tree hiện tại trên HEAD `c168b19`, bao gồm các sửa đổi chưa commit sau audit 05/10. Rà soát simulation, preprocessing, training, routing, thống kê, artifact nghiên cứu và serving/deployment. Không sửa mã ứng dụng, không train lại hoặc deploy.

**Kết luận:** nhiều lỗi vận hành cũ đã được sửa, nhưng chưa đủ cơ sở dùng artifact hiện có để chốt kết luận khoa học. Còn **11 phát hiện: 4 P1, 7 P2**. P1 ảnh hưởng phép đánh giá chính hoặc khả năng tái lập; P2 ảnh hưởng một luồng hoặc một cấu hình cụ thể. Phát hiện 1-2 áp dụng cho bộ controlled ghép seed; không quy chúng cho bộ `ns3big` chưa kiểm tra quan hệ seed.

## Phát hiện

### 1. P1 - Controlled LORO không giữ toàn bộ seed ngoài tập train

[run_stage6_benchmark.py](../../scripts/train/run_stage6_benchmark.py#L191) chỉ bỏ `run != test_run`. Trong khi [run_controlled_scenarios.py](../../scripts/dataset/run_controlled_scenarios.py#L42) tạo bốn scenario với cùng seed, mobility và số UAV. Baseline/sparse/dense còn dùng cùng cấu hình tốc độ và vùng bay, chỉ đổi radio range.

Probe gọi chính runner và chụp command: test `stress_baseline_rwp_s60001`, nhưng train chứa `stress_dense_rwp_s60001`, `stress_fast_rwp_s60001`, `stress_sparse_rwp_s60001`. Đây là giữ ra một biến thể scenario của seed đã thấy, không phải kiểm tra seed/mobility realization hoàn toàn mới. Chưa chạy ns-3 để so tọa độ từng timestamp; quan hệ seed và cấu hình đã xác minh từ command.

**Sửa:** giữ ra cả nhóm seed khi đo tổng quát hóa sang realization mới; các scenario cùng seed phải cùng fold. Lưu danh sách seed train/val/test trong manifest. Có thể giữ phép đánh giá hiện tại nếu đổi tên và mô tả đúng estimand, nhưng không gọi nó là khả năng tổng quát hóa sang seed chưa thấy.

### 2. P1 - Thống kê controlled coi các scenario cùng seed là quan sát độc lập

[paired_statistics.py](../../src/evaluation/paired_statistics.py#L88) pivot theo `run_name`; [run_multihorizon_benchmark.py](../../scripts/train/run_multihorizon_benchmark.py#L86) bootstrap từng dòng metric. Khi gộp bốn scenario, một seed đóng góp bốn đơn vị được resample/sign-flip riêng, dù thiết kế controlled cố ý ghép chúng.

400 condition runs chỉ có 100 nhóm seed. Với lựa chọn mặc định 10 run/scenario, 40 condition runs chỉ có 10 nhóm seed; 5 LORO folds/scenario tương ứng 20 condition folds trên 5 seed. Số scenario runs vẫn có thể báo cáo là 400/40/20, nhưng không dùng chúng làm số lần lặp seed độc lập. CI và p-value hiện tại không tính cấu trúc phụ thuộc này.

**Sửa:** phân tích từng scenario riêng hoặc bootstrap/sign-flip theo cụm seed, giữ cả bốn scenario cùng một lần resample. Giữ rõ trọng số scenario và báo cả số condition runs lẫn số seed. Đây là kết luận từ thiết kế và cách resampling; audit không ước lượng lại CI cuối.

### 3. P1 - Artifact LORO hiện có vẫn so khác số mẫu sau khi loader được sửa

[loro_baselines.py](../../src/training/baselines/loro_baselines.py#L150) đã lọc train/val/test, loại purged times. Tuy nhiên [loro_summary.csv](../stage6/loro_summary.csv) hiện có vẫn lệch số mẫu ở **40 tổ hợp target/horizon/run**.

Ví dụ cùng run `ns3big_001_rwp_s17296_n18_c211_t108`:

| Target/Horizon | Edge-SAGE | LogReg | XGB |
| --- | ---: | ---: | ---: |
| qos/k1 | 2.793 | 2.855 | 2.855 |
| qos/k5 | 2.541 | 2.855 | 2.855 |
| survival/k1 | 2.793 | 2.855 | 2.855 |
| survival/k5 | 2.541 | 2.855 | 2.855 |

[run_stage6_benchmark.py](../../scripts/train/run_stage6_benchmark.py#L194) chỉ kiểm tra `metrics.csv` tồn tại để skip. Chạy lại command mặc định sau khi sửa loader sẽ giữ metric cũ. Các bảng/figure suy ra từ CSV này cũng cần được tạo lại, không chỉ đọc lại CSV cũ.

**Sửa:** chạy lại các thí nghiệm bị ảnh hưởng với `--force`, đối chiếu cả danh tính `(run,time,src,dst)` rồi sinh lại aggregate, paired effects, figure và báo cáo. Khi resume, kiểm tra revision dữ liệu/code/config và danh sách fold thay vì chỉ sự tồn tại file. CSV là artifact local, không phải bằng chứng kết quả tracked đã được cập nhật.

### 4. P1 - Thay source ns-3 không bảo đảm thí nghiệm dùng binary mới

[run_one_dataset_ns3.sh](../../scripts/dataset/run_one_dataset_ns3.sh#L42) chỉ chạy CMake khi executable chưa tồn tại. [dvc.yaml](../../dvc.yaml#L4) theo dõi source C++, nên thay source làm stage generate chạy lại, nhưng script vẫn dùng binary cũ. Các sửa priority OLSR/FlowMonitor hiện tại vì thế chưa chắc đi vào dữ liệu được tái sinh trên máy đã build.

Stage generate còn không khai báo dependency `scripts/dataset/run_one_dataset_ns3.sh` hoặc `simulation/ns3/CMakeLists.txt`. Sửa các file này riêng lẻ không làm DVC nhận ra generate cần chạy lại.

**Sửa:** gọi incremental `cmake --build` trước batch khi dùng local build và khai báo đủ dependency. Với binary lấy từ image, khóa image/binary revision cùng source và dataset manifest. Đây là kiểm chứng luồng mã; checkout này không có binary ns-3 để chứng minh một batch thực đã dùng bản cũ.

### 5. P2 - Image serving vẫn ép kiến trúc Edge-SAGE

[Dockerfile.serve](../../Dockerfile.serve#L16) vẫn đặt `ENV MODEL_ID=edge-sage`. [app.py](../../src/serving/app.py#L40) từ chối khi biến môi trường khác `metadata.model_id`. Bỏ MODEL_ID khỏi ConfigMap không xóa ENV được bake trong image.

Probe dùng checkpoint GAT hợp lệ và môi trường đúng như image, nhận `ValueError: MODEL_ID=edge-sage disagrees with artifact gat`. GraphSAGE và các ablation có model_id khác cũng gặp cùng nhánh. Luồng promote/stage đã nhận metadata của các model này nhưng image sẽ không khởi động theo mặc định.

**Sửa:** bỏ default MODEL_ID khỏi image để metadata quyết định; chỉ override có chủ đích. Smoke check phải load checkpoint staged, không chỉ import class GNN. Đây là phần còn sót của phát hiện 18 ở audit trước.

### 6. P2 - Sửa lọc cohort làm mất Persistence khỏi bảng within-run

[run_multihorizon_benchmark.py](../../scripts/train/run_multihorizon_benchmark.py#L229) lọc model theo `models`, nhưng MODELS/default chỉ gồm logreg/xgb/edge-sage. [launcher](../../scripts/run_controlled_paper_pipeline.sh#L59) chạy Persistence riêng rồi gọi benchmark mặc định.

Probe đặt metric của đủ bốn model trên cùng cohort: summary chỉ giữ ba model. `paired_model_comparisons` trả `n_pairs=0` cho LogReg/Persistence và XGB/Persistence; Stage 7 loại các comparison này. Vì vậy đối chứng last-state đã được tính nhưng biến mất khỏi bảng/figure cuối.

**Sửa:** ghép metric Persistence trên đúng manifest cohort được chọn, hoặc truyền nó vào tập model cần báo cáo mà không yêu cầu train lại. Giữ việc loại các run ngoài cohort.

### 7. P2 - Stage 6 và routing vẫn gom kết quả ngoài phạm vi command

[run_stage6_benchmark.py](../../scripts/train/run_stage6_benchmark.py#L231) quét toàn protocol output sau khi chỉ chạy target/horizon/model/fold đã chọn. Probe yêu cầu qos/k1/logreg nhưng summary vẫn chứa artifact cũ survival/k5/xgb.

[multihorizon_eval.py](../../src/routing/multihorizon_eval.py#L144) cũng gọi aggregate toàn routing root; [collector](../../src/routing/multihorizon_eval.py#L152) không nhận manifest run/model/cost mode của invocation. Giảm `--runs-per-scenario`, đổi model hoặc chạy subset trong output root có dữ liệu cũ không bảo đảm báo cáo phản ánh subset đó. Việc sửa collector within-run chưa xử lý hai đường này.

**Sửa:** lọc aggregate bằng manifest của invocation, kiểm tra uniqueness và so cohort giữa các model. Probe tái hiện trực tiếp nhánh Stage 6; nhánh routing được xác định từ caller và glob.

### 8. P2 - Cache inference benchmark thiếu target, horizon và cấu hình đo

[benchmark_inference_resources.py](../../scripts/analysis/benchmark_inference_resources.py#L285) dùng tên `model__run.json` rồi skip nếu file tồn tại. Đổi `--target`, `--horizon`, `--threads`, warmups/repeats hoặc model artifact vẫn dùng measurement cũ.

Probe yêu cầu survival/k5/4 threads trong cùng report directory và nhận lại qos/k1/1 thread. Sửa lọc danh sách run/model ở collector không kiểm tra các tọa độ này.

**Sửa:** so metadata cache với toàn cấu hình và artifact revision trước skip, hoặc đưa các tọa độ vào output path. Khác với benchmark mới, `--force` là cách chạy lại thủ công hiện có.

### 9. P2 - Replay vẫn đổi endpoint H thành horizon ngắn hơn ở cuối trace

[replay_eval.py](../../src/routing/replay_eval.py#L159) chỉ yêu cầu hỗ trợ `t+1`; [dòng 188](../../src/routing/replay_eval.py#L188) dùng `min(horizon, max_time-t)`, rồi báo `survival_at_horizon` theo H bị rút ngắn.

Probe với raw chỉ có t=0,1, yêu cầu H=5, nhận `horizon=1`, `survival_at_horizon=1`. Không đủ dữ liệu để kết luận sống tới t+5. Lỗi giữ empty snapshot đã sửa, nhưng không giải quyết censoring khi trace thật kết thúc. Luồng multi-horizon có common support 5 giảm nguy cơ này; replay legacy/DVC và horizon sweep vẫn bị ảnh hưởng.

**Sửa:** yêu cầu `t+H <= max_time` cho endpoint cố định, hoặc ghi rõ censored và không tính nó là sống đủ H. Ghi riêng endpoint survival dựa QoS hay connectivity khi diễn giải kết quả.

### 10. P2 - Route plan báo không tìm thấy đường nhưng ns-3 vẫn cài route gửi trực tiếp

[uav-olsr-dataset.cc](../../simulation/ns3/uav-olsr-dataset.cc#L414) khi `!entry.found` gọi `AddHostRouteTo(destination, sourceInterface)` rồi return. Overload này là route trực tiếp ra interface, không phải blackhole. Static routing được ưu tiên cao cho planned strategies.

Khi graph không tìm được đường nhưng một lần unicast trực tiếp vẫn có thể nhận được, FlowMonitor có thể ghi packet delivery dù plan báo không có route. Khi một đoạn planned path thiếu route, connected-subnet route cũng là fallback có thể làm đường thực khác kế hoạch.

**Sửa:** định nghĩa rõ hành vi no-route và bắt buộc packet path khớp plan; dùng cơ chế drop/gating hoặc cấu hình routing tránh fallback ngoài kế hoạch. Nếu muốn direct fallback, ghi nó thành chính sách được đánh giá và đo đường thực.

**Mức xác minh:** suy luận từ mã repo và [source chính thức của ns-3](https://www.nsnam.org/doxygen/d6/d09/ipv4-static-routing_8cc_source.html), phần `AddHostRouteTo` không gateway; thứ tự protocol được mô tả trong [routing overview](https://www.nsnam.org/docs/models/html/routing-overview.html). Chưa chạy kiểm thử packet path trên binary 3.48 của dự án, không kết luận mức thay đổi PDR thực.

### 11. P2 - Biên API vẫn nhận số vượt miền biểu diễn của model

[schemas.py](../../src/serving/schemas.py#L21) nhận float không có miền giá trị. Node references/duplicate IDs đã được kiểm tra, nhưng payload JSON hợp lệ có `distance=1e100` vẫn được chấp nhận. Khi [app.py](../../src/serving/app.py#L122) tạo tensor float32, giá trị thành infinity, model sinh score NaN và response validation ném exception.

Probe dùng GAT thật ở eval mode, tái hiện lỗi validation trên `stability_score`. Qua HTTP đây là lỗi server, thay vì từ chối payload ở request. Loss ngoài [0,1] và đặc trưng vật lý âm cũng đang được nhận.

**Sửa:** từ chối non-finite và giá trị vượt miền float32, thêm bounds theo feature ở schema; không tự clamp telemetry sai thành dự đoán có vẻ hợp lệ. Thêm một kiểm tra HTTP cho payload lớn với kỳ vọng 422.

## Đối chiếu audit 05/10

| Nhóm lỗi cũ | Trạng thái hiện tại |
| --- | --- |
| 13 script Bash sai cú pháp | Đã sửa; 21/21 script pass `bash -n` |
| Docker thiếu requirements include | Đã sửa thứ tự COPY; chưa build full image |
| Writer/reader GNN metadata và edge modes | Đã đồng bộ trong loader đã đọc; suite có kiểm tra model/serving |
| Cohort test GNN/baseline LORO | Loader đã sửa; artifact cũ chưa được sinh lại, xem #3 |
| Priority static/OLSR | Source đã sửa; cần kiểm tra binary mới và packet path, xem #4/#10 |
| Serving port 8000/8080, image name chữ hoa, threshold override | Đã sửa trong Docker/Kustomize/workflow đang đọc |
| Biên QoS, timestamp disconnected cuối trace, fallback reroute | Đã sửa; horizon cuối trace còn #9 |
| Throughput/zero-reception delay | Source chuyển sang offered window/NaN; chưa đo lại FlowMonitor |
| Closed-loop resume khi đổi src/dst/rate/packet size | Đã thêm so bốn trường; thay model/data/code vẫn không được fingerprint |
| Lọc cohort within-run/ablation reference | Đã sửa phần đã nêu; xem phát sinh #6 và phần còn sót #7 |
| Scenario density, danh tính run, p_th duplicate subset | Đã sửa trong source |
| Query node không tồn tại, dashboard URL/sample | Đã sửa schema và sample; validation numeric còn #11 |
| Serving architecture | Loader đã sửa; ENV trong image còn #5 |
| DVC credential trong archive, nuốt lỗi git commit | Source đã sửa; hai lifecycle checks pass trên Linux |
| DB password hardcode, AZ cố định | Source đã đổi sang managed password và provider AZ; chưa Terraform validate/apply |
| Workspace DVC trong image/hướng dẫn mount | Đã sửa ignore và hướng dẫn; chưa chạy `dvc repro` trong image |
| GDrive/OpenSSL dependencies | Đã pin và thêm CI import smoke; chưa cài mới để xác minh resolver Linux trong lượt này |

## Kiểm chứng

- Pytest trên Windows/Python 3.12.14: lượt đầu **83 pass, 3 fail**. Một fail do sandbox chặn SQLite cache DVC; chạy lại đúng test ngoài sandbox **pass**. Hai fail còn lại do Bash/WSL bị chặn, không tính là lỗi sản phẩm.
- Hai test lifecycle Bash được gọi nguyên test function trong container `python:3.12-slim`, mount repo read-only, network disabled: **2 pass**. Tổng cộng **84 kiểm thử pytest Windows và 2 kiểm thử lifecycle Linux đã pass**, không tuyên bố đã chạy toàn suite trên Linux.
- Ruff check `src/ simulation/ tests/ scripts/`: **pass**. Ruff format-check `src/ simulation/ tests/`: **82 files đúng format**.
- Bash syntax trên nội dung chuẩn hóa LF: **21 pass, 0 fail**.
- [probe_reaudit.py](./probe_reaudit.py): **8 kiểm tra xác nhận thành công**, gồm seed overlap, Persistence bị loại, scope Stage 6, cache inference, replay H bị rút, image MODEL_ID, numeric input và artifact LORO lệch mẫu. Probe assert hành vi đang lỗi; đây là bằng chứng audit, không phải regression suite yêu cầu hành vi đúng.
- Probe chỉ ghi dữ liệu tạm dưới `.cache`, không train hoặc ghi đè data/model/report nghiên cứu.
- Không có `data/`, `outputs/` hoặc binary ns-3 trong checkout chính. Không train 400 controlled runs, không kiểm chứng cloud, không build toàn image nghiên cứu, không apply Terraform. Các phát hiện C++ được gắn rõ mức kiểm chứng.

Chạy lại probe từ repo root:

```powershell
rtk proxy .cache\audit-venv\Scripts\python.exe reports/audit-2026-10-07/probe_reaudit.py
```

## Tinh giản

1. `delete:` Có thể bỏ 5 wrapper CLI trong [src/models/baselines](../../src/models/baselines/Logistic_Regression_Baseline.py), tổng 60 dòng; dùng trực tiếp `python -m src.training.baselines.<model>`. Tìm trong source, tests, docs, scripts và tracked/untracked project content không thấy caller tới đường cũ. Chỉ bỏ khi không cần giữ tương thích với command bên ngoài repo; không ưu tiên việc này trước các lỗi nghiên cứu.

net: -60 lines, -0 deps possible.

## Thứ tự xử lý

Khóa seed group và đơn vị thống kê (#1/#2), bảo đảm binary/source cùng revision (#4), sửa cohort và resume (#3/#6/#7/#8), rồi sinh lại artifact. Sửa image/validation (#5/#11) trước demo triển khai. Kiểm chứng ns-3 ba node và no-route packet path (#10), sau đó mới dùng PDR đo thực để kết luận hệ thống tốt hơn OLSR. Các bảng hiện có nên giữ nhãn exploratory cho đến khi hoàn tất bước này.
