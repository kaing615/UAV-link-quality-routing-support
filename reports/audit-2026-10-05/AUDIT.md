# Audit repository — 2026-10-05

**Revision:** `c168b19` (`perf: bound controlled paper training`). Phạm vi gồm simulation Python/ns-3, preprocessing, models/training, evaluation/routing, scripts, serving/dashboard, Docker, CI/CD, Kustomize và Terraform. Đây là audit; không sửa mã ứng dụng, không deploy, không thay đổi dữ liệu thí nghiệm.

**Kết luận:** có lỗi chặn pipeline và lỗi làm sai phép so sánh khoa học. Cần sửa luồng sinh dữ liệu, hợp đồng artifact và tập đánh giá trước khi dùng bảng kết quả hiện tại để kết luận GNN tốt hơn baseline. P1 = chặn luồng chính hoặc làm sai đối chứng quan trọng; P2 = lỗi cần sửa; P3 = lỗi phạm vi hẹp.

**Tổng hợp:** 26 phát hiện — 7 P1, 18 P2, 1 P3. Mức độ được xếp theo tác động, không phải mọi lỗi đều đã xuất hiện trên hệ thống cloud đang chạy.

## Phát hiện ưu tiên cao

1. **P1 — 13/21 script Bash không qua kiểm tra cú pháp.** Ví dụ `scripts/dataset/run_one_dataset_ns3.sh:10` còn `if [[ $`, `scripts/train/gnn/run_all_gnn_for_runs.sh:35` và `scripts/routing/run_routing_for_runs.sh:20` còn `if [[ ${`; nhiều chuỗi `${#...}`/`$#` đã bị cắt. Cả DVC và controlled launcher gọi những script này nên dừng trước khi sinh dữ liệu hoặc train. Đã chạy `bash -n` trên nội dung chuẩn hóa LF, do đó không phải chỉ do CRLF của Windows; bản HEAD của script ns-3 cũng chứa lỗi. **Sửa:** khôi phục các biểu thức bị cắt, đưa Bash syntax check vào CI. Danh sách đầy đủ ở cuối báo cáo.

2. **P1 — Dockerfile nghiên cứu và dashboard thiếu file requirements được include.** `Dockerfile:29` và `Dockerfile.dashboard:5` chỉ COPY `requirements.txt`, trong khi `requirements.txt:15` chứa `-r requirements-serve.txt`. File này chưa có ở lớp chạy pip; COPY toàn repo diễn ra sau đó. Cả hai image sẽ dừng khi cài dependency. **Sửa:** COPY cả hai requirements trước RUN pip. Xác minh từ thứ tự lớp Dockerfile; chưa build toàn image ns-3.

3. **P1 — Model vừa train không được các loader downstream đọc lại.** `src/training/gnn/train_gnn.py:243` và `src/training/gnn/train_gnn_loro.py:216` ghi hai cờ `use_message_edge_features`/`use_decoder_edge_features`, bỏ `use_edge_features`. `src/routing/predict_edges.py:69`, `src/evaluation/refresh_gnn_metrics.py:29`, `scripts/analysis/benchmark_inference_resources.py:129` vẫn truy cập khóa cũ và phát sinh `KeyError`. Decoder-only/message-only còn không được map đúng model ID. Controlled pipeline vì thế có thể train xong rồi chết ở resource benchmark. **Sửa:** đồng bộ writer/reader cho mọi edge mode, kiểm thử train artifact → load/infer.

4. **P1 — GNN và baseline dùng khác tập test trong LORO/cross-mobility.** `src/training/gnn/train_gnn_loro.py:85` nối train/val/test PT; `src/preprocessing/gnn/build_graph_dataset.py:38` bỏ thời điểm purged. `src/training/baselines/loro_baselines.py:150` lại lấy mọi hàng labeled, kể cả purged. File kết quả có sẵn tại máy `reports/stage6/loro_summary.csv` xác nhận cùng run `ns3big_001_rwp_s17296_n18_c211_t108`, qos/k1: GNN **2.793**, baseline **2.855** mẫu; qos/k5: GNN **2.541**, baseline **2.855** mẫu. CSV này là artifact local bị gitignore, không phải file tracked. **Sửa:** đánh giá trên cùng danh sách `(run,time,src,dst)` rồi sinh lại metric, CI và bảng paper.

5. **P1 — Lưu lượng đối chứng OLSR bị static routing bắt trước.** `simulation/ns3/uav-olsr-dataset.cc:875` đặt static priority 20, OLSR 10; dòng 883 gán tất cả node vào `10.0.0.0/24`. Static routing tự thêm route trực tiếp cho subnet đó, nên mọi đích được match trước OLSR, thay vì dùng relay do OLSR tìm. Đây là suy luận từ mã repo và mã nguồn/tài liệu ns-3, chưa chạy kiểm chứng C++ thực tế. [Quy tắc thứ tự routing](https://www.nsnam.org/docs/models/html/routing-overview.html) và [NotifyInterfaceUp tạo route mạng trực tiếp](https://www.nsnam.org/doxygen/d6/d09/ipv4-static-routing_8cc_source.html) xác nhận cơ chế. **Sửa:** ưu tiên OLSR trong đối chứng; giữ static ưu tiên cao chỉ cho planned routing. Bắt buộc kiểm thử chuỗi ba node mà hai đầu không nghe nhau trực tiếp.

6. **P1 — Kubernetes dùng cổng 8000, server nghe 8080.** `deploy/kustomize/base/deployment-serve.yaml:19` khai báo `http:8000`; probes và Service dùng cổng có tên đó. `Dockerfile.serve:21` chạy `${PORT:-8080}` và overlays không đặt PORT. Kết quả là health/readiness fail và Service không tới API. Đã render cả staging/production để đối chiếu. **Sửa:** thống nhất 8080 hoặc đặt PORT=8000; cập nhật lệnh Docker ở `docs/quick_start.md:634`.

7. **P1 — Tên image staging chứa chữ hoa và workflow tiếp tục tạo lại tên đó.** `deploy/kustomize/overlays/staging/kustomization.yaml:28` chứa `UAV-link-quality-routing-support`; `.github/workflows/build-and-push.yml:122` tái tạo từ `github.repository`. Build dùng metadata-action hạ chữ thường, nhưng deployment dùng tên chưa chuẩn hóa, gây invalid image reference. [Quy tắc metadata-action](https://github.com/docker/metadata-action#image-name-and-tag-sanitization). **Sửa:** truyền image reference/digest thực từ job build sang job cập nhật manifest; sửa overlay hiện tại.

## Sai lệch metric, routing và tái lập

8. **P2 — Replay và nhãn train khác nhau ở biên QoS.** `src/routing/replay_eval.py:33` dùng `>18`, `<0.1`, `<10`, còn `src/preprocessing/gnn/build_labels.py:23` chấp nhận dấu bằng. Link có loss=0.1 được train là ổn định nhưng replay đánh chết. **Sửa:** dùng chung predicate và threshold; probe audit kiểm tra trực tiếp trường hợp này.

9. **P2 — Mất kết nối ở cuối trace bị biến thành sống sót đủ horizon.** `src/routing/replay_eval.py:31` bỏ timestamp không còn connected edge; dòng 155 dùng `max(raw)` làm cuối trace và dòng 185 rút ngắn horizon. Chuỗi connected tại t=0,1, disconnected tại t=2, yêu cầu H=2 thành H=1 và survival=1. `src/routing/closed_loop.py:119` cũng dùng giới hạn bị rút ngắn này. **Sửa:** giữ timestamp với tập cạnh rỗng và xác định support từ toàn bộ raw times.

10. **P2 — Reroute không giữ chính sách fallback của chế độ non-strict.** Khởi tạo tại `src/routing/replay_eval.py:197` fallback sang graph không lọc nếu p_th làm mất đường; reroute dòng 227 không làm vậy. Mạng còn đường vẫn bị ghi disconnected, vòng lặp dừng và số route change giảm giả. **Sửa:** áp dụng cùng lựa chọn strict/fallback cho mọi lần chọn tuyến.

11. **P2 — Throughput và delay ưu ái run bị mất kết nối.** `simulation/ns3/uav-olsr-dataset.cc:649` chia bytes nhận cho `lastRx-firstTx`, loại toàn bộ thời gian outage cuối phiên khỏi mẫu số. Dòng 648 còn ghi delay=0 khi không nhận được gói nào; aggregation coi đây là delay tốt. **Sửa:** throughput theo cửa sổ offered traffic chung; delay không có quan sát phải là missing và báo số run hợp lệ. Chưa chạy FlowMonitor thực tế; lỗi được xác định từ công thức.

12. **P2 — Resume closed-loop bỏ qua thay đổi cấu hình thí nghiệm.** `src/routing/closed_loop.py:286` chỉ dùng run/target/horizon/cost/strategy làm đường dẫn, rồi skip khi metrics tồn tại. Thay `--source`, `--destination`, `--app-rate-kbps`, `--packet-size` vẫn nhận kết quả cũ, mặc dù các trường này là tọa độ thống kê ở dòng 170. **Sửa:** so manifest cấu hình hoặc đưa các tọa độ vào khóa output trước khi cho skip.

13. **P2 — Giới hạn tập run không được giữ khi tổng hợp benchmark.** `scripts/train/run_multihorizon_benchmark.py:226` gọi collector quét toàn output, kể cả run/model ngoài lựa chọn hiện tại. Launcher chạy persistence trên cả 400 run nhưng mặc định chỉ train các model khác trên 40 run; các mean và đường biểu diễn trong bảng within-run vì vậy không cùng cohort. Tương tự `scripts/train/run_stage6_benchmark.py:54` chọn full ablation reference chỉ theo target/horizon, không theo run, nên khi hai giới hạn within/ablation khác nhau sẽ so khác tập run. **Sửa:** tổng hợp đúng manifest run đã chọn, báo cả số run và giao cohort; paired comparisons hiện có chỉ ghép phần chung nhưng không sửa được các aggregate mean khác cohort.

14. **P2 — Controlled worst-group gộp mọi scenario thành sparse.** `scripts/analysis/analyze_multihorizon_stage6.py:42` chia density theo số UAV/diện tích. Cả baseline/fast/sparse/dense controlled đều là 20 UAV/800×800; chúng khác radio range. Probe với cả bốn cấu hình trả về `density_group=sparse` cho tất cả, kể cả dense. **Sửa:** giữ scenario identity để đánh giá manipulation, hoặc dùng chỉ số graph density/degree phù hợp; không gắn nhãn spatial density như thể đó là controlled scenario.

15. **P2 — Legacy aggregator ghi đè danh tính run test.** `src/evaluation/aggregate_all_metrics.py:83` thay mọi `run_name` bằng tên thư mục chứa metrics. Với `--test-runs` nhiều run, một file chứa nhiều dòng test bị gom thành một run giả; scenario và balanced filtering cũng sai. **Sửa:** giữ run_name từ từng dòng CSV, chỉ fill giá trị thiếu; collector Stage 6 đã có cách xử lý này.

16. **P2 — p_th sweep đếm thêm subset như một run.** `src/routing/plot_pth_sweep.py:24` glob `summary*.csv` lấy cả `summary_olsr_pair.csv`. Đây là subset cùng nguồn đã có trong `summary.csv`, làm p_th=0 bị trọng số khác các ngưỡng còn lại. **Sửa:** chỉ lấy summary all-pairs và các summary_pth tương ứng, kiểm tra uniqueness theo run/strategy/p_th.

## API, triển khai và bảo mật

17. **P2 — Query có node không tồn tại gây lỗi server.** `src/serving/app.py:114` lọc query khi dựng tensor nhưng dòng 130 vẫn enumerate danh sách gốc để lấy score. Với nodes={1,2}, queries=[(999,2),(1,2)], score chỉ có một phần tử nhưng code đọc scores[1]. Schema hiện cho phép payload này. **Sửa:** validate tham chiếu node và duplicate IDs ở biên request, hoặc dùng cùng danh sách query hợp lệ xuyên suốt.

18. **P2 — Serving bỏ qua kiến trúc được promote.** `src/serving/app.py:33` chọn model từ env mặc định Edge-SAGE; metadata chỉ cập nhật kích thước/threshold. Promotion nhận GAT/GraphSAGE/noedge nhưng Docker/ConfigMap vẫn cố khởi tạo Edge-SAGE rồi load state_dict không tương thích; các edge mode mới cũng chưa được hỗ trợ. **Sửa:** dựng model từ metadata, kiểm tra override rõ ràng hoặc từ chối artifact không hỗ trợ trước promotion.

19. **P2 — Kubernetes bỏ threshold đã tune.** `deploy/kustomize/base/configmap.yaml:9` luôn đặt THRESHOLD=0.5; `src/serving/app.py:45` chỉ đọc threshold metadata nếu env không tồn tại. Mô hình deployed vì vậy có thể phân loại khác model được chấm điểm. **Sửa:** bỏ default override, chỉ đặt khi chủ động hiệu chỉnh.

20. **P2 — Dashboard mặc định không gọi được API và sample không hợp schema.** `src/dashboard/app.py:8` hardcode localhost:8000, bỏ qua API_URL mà Cloud Run script truyền ở dòng 104. API/dashboard chạy container riêng nên địa chỉ này trỏ về dashboard. Sample tại dòng 22–24 còn dùng speed và thiếu load bắt buộc bởi `src/serving/schemas.py:15`, nên Predict trả 422 ngay cả sau khi sửa URL. **Sửa:** đọc API_URL từ env và dùng sample theo schema chuẩn; hai lỗi này cần sửa độc lập.

21. **P2 — Credential DVC được đưa vào Cloud Build source archive.** `.github/workflows/deploy.yml:64` ghi key JSON vào checkout; `scripts/mlops/deploy_cloud_run.sh:64` upload toàn checkout; `.gcloudignore` không loại `dvc_creds.json`. Đây là mở rộng nơi lưu credential không cần thiết, không phải kết luận key đã nằm trong inference image hay bị lộ công khai. [Cơ chế gcloudignore](https://docs.cloud.google.com/sdk/gcloud/reference/topic/gcloudignore). **Sửa:** dùng RUNNER_TEMP, dọn sau staging và thêm exclude cụ thể trước upload.

22. **P2 — Promotion nuốt lỗi commit rồi tag version cũ.** `scripts/mlops/promote_model.sh:40` coi mọi lỗi git commit là “pointer unchanged”, sau đó tag/push HEAD. Nếu hook hoặc signing làm commit thất bại, artifact mới chỉ nằm staged trong máy nhưng release tag trỏ version cũ. **Sửa:** kiểm tra empty staged diff riêng; lỗi commit thật phải dừng script.

23. **P2 — Terraform đặt mật khẩu DB cố định trong repo.** `terraform/modules/storage_db/main.tf:51` hardcode password; dòng 38 cho cả VPC vào PostgreSQL. Người triển khai dùng mặc định sẽ tạo DB với credential đã công khai trong source. Tài liệu gọi phần này là kế hoạch, nên không có bằng chứng DB thực đang bị ảnh hưởng. **Sửa:** dùng RDS-managed master credentials/Secrets Manager và giới hạn security group.

24. **P3 — aws_region có thể đổi nhưng AZ bị cố định.** `terraform/modules/vpc/main.tf:8` dùng AZ us-east-1 trong khi environment cho cấu hình region. Chuyển region sẽ làm subnet provisioning lỗi. **Sửa:** lấy availability zones từ provider đang cấu hình.

25. **P2 — Image nghiên cứu không chứa workspace DVC như hướng dẫn sử dụng.** `.dockerignore:4`, `:11`, `:19` loại `.dvc/`, `scripts/`, `dvc.yaml`, `dvc.lock` và project config trước `Dockerfile:35` COPY. Dù sửa lỗi requirements ở mục 2, lệnh standalone `docker run -it uav-gnn dvc repro` tại `README.md:159` vẫn thiếu DVC repo và stage scripts. **Sửa:** đồng bộ hướng dẫn với workspace bind mount có sẵn hoặc đóng gói workspace DVC đầy đủ. Controlled runbook bind-mount `/workspace` không chịu lỗi này.

26. **P2 — Cài requirements thành công nhưng GDrive authentication không import được.** Cài mới `requirements.txt` trong Python 3.12.14/Windows ngày audit resolve thành DVC 3.67.1, dvc-gdrive 3.0.1, PyDrive2 1.21.2, pyOpenSSL 22.0.0 và cryptography 50.0.2. `pip check` PASS nhưng `import pydrive2.auth` lỗi `AttributeError: module 'lib' has no attribute 'GEN_EMAIL'` trong OpenSSL. Do default remote là GDrive, môi trường này không dùng được luồng auth cho pull/push. `requirements.txt:11` không khóa bộ dependency này, trong khi workflow deploy có pin riêng. **Sửa:** dùng bộ constraints tương thích và thêm import smoke check; cần kiểm chứng riêng trên Linux thay vì suy rằng kết quả resolver Windows giống mọi nền tảng. Probe không dùng credential hay gọi remote.

## Kiểm chứng và giới hạn

- `bash -n`: **8 pass, 13 fail**; kiểm tra nội dung chuẩn hóa LF, không chạy các thao tác trong script.
- Render Kustomize staging/production: thành công; xác nhận cổng, threshold và image reference sai như trên.
- Ruff 0.15.17 check `src/ simulation/ tests/`: **PASS**; format check: **82 file đúng format**. Ruff check `scripts/`: **PASS**; format check bổ sung cho scripts: **5 file cần format**, 7 file đúng. CI hiện không kiểm tra Bash và không format-check scripts.
- Hai test chức năng staging/promotion hiện có: **2 PASS** khi gọi chính các test function trong container Linux, dùng fake git/DVC và thư mục tạm. Không chạm repository/remote thật.
- `probe_reporting.py`: đã tái hiện density grouping, ablation lệch cohort, metadata thiếu khóa; đọc và đối chiếu số mẫu trong artifact LORO local.
- `probe_serving.py`: đã tái hiện `IndexError` ở query, 3 node sample thiếu load, và `KeyError` khi đọc metadata mới bằng routing loader.
- `reproduce_routing_findings.py`: đã tái hiện đủ ba lỗi biên QoS, mất timestamp disconnected cuối trace và mất fallback khi reroute, chạy lại bằng dependency pin của repo.
- `dvc dag --dot`: **PASS**, gồm 7 stage. Sandbox ban đầu chặn database cache; đã chạy lại với quyền bình thường. DAG hợp lệ không chứng minh các command trong stage chạy được.
- `pip check`: **PASS**, nhưng GDrive authentication import **FAIL**, chi tiết ở mục 26.
- Pytest Python 3.12.14 + torch 2.11.0 CPU + requirements của repo: **77 PASS**, 0 fail, 2 cảnh báo deprecated torch.jit.script, thời gian 156,38 giây. Hai test trong `test_serving_model_lifecycle.py` được chạy riêng trên Linux như trên vì dùng Bash/POSIX. Lượt pytest sandbox đầu có 38 lỗi setup do không được truy cập thư mục temp; chạy lại với quyền bình thường đã hết toàn bộ, nên không tính chúng là lỗi repo.
- Coverage của lượt pytest thành công: **44%**, 1.084/2.447 statements đo được trong `src`. Không phải coverage toàn repo; Bash, C++ và `scripts/` không nằm trong mẫu số này. `train_gnn.py` 15%, `train_gnn_loro.py` 11%; legacy metric aggregation và refresh chưa được suite hiện tại chạy tới. Báo cáo máy đọc: `pytest.xml`, `coverage.json` trong cùng thư mục.
- Không có `data/`, `outputs/` hay binary ns-3 trong checkout này. Không train lại 400 run, không tái tạo kết quả paper, không chạy Terraform apply/Cloud Run deploy. Không thể xác nhận trạng thái hệ thống cloud từ repository.
- Chỉ thêm tài liệu/script trong `reports/audit-2026-10-05/` và môi trường tạm `.cache/audit-venv/`. Giữ nguyên các file người dùng đang có.
- Phần kiểm tra độ phức tạp không đưa ra đề xuất xóa module chỉ vì ít caller: các lỗi hợp đồng và đánh giá ở trên có tác động rõ ràng hơn việc giảm dòng mã.

**13 script lỗi cú pháp:**

```text
scripts/dataset/run_many_random_datasets.sh
scripts/dataset/run_many_random_datasets_ns3.sh
scripts/dataset/run_one_dataset.sh
scripts/dataset/run_one_dataset_ns3.sh
scripts/routing/run_pth_sweep.sh
scripts/routing/run_routing_for_runs.sh
scripts/train/baselines/run_all_baselines_for_runs.sh
scripts/train/gnn/run_all_gnn_for_runs.sh
scripts/train/gnn/run_edge_sage_for_runs.sh
scripts/train/gnn/run_loro.sh
scripts/train/mlp/run_all_mlp_for_runs.sh
scripts/train/xgb/run_all_xgb_for_runs.sh
scripts/utils/clean_data_outputs.sh
```

**Thứ tự sửa đề nghị:** Bash/Docker → hợp đồng artifact → OLSR và cohort test → định nghĩa metric/replay → tái tạo bảng nghiên cứu → serving/deployment. Mỗi phần cần một kiểm thử xuyên qua ranh giới đang lỗi, thay vì chỉ kiểm tra helper riêng lẻ.

**Chạy lại các probe** từ repo root sau khi kích hoạt môi trường đủ dependency:

```powershell
rtk proxy .cache\audit-venv\Scripts\python.exe reports/audit-2026-10-05/probe_reporting.py
rtk proxy .cache\audit-venv\Scripts\python.exe reports/audit-2026-10-05/probe_serving.py
rtk proxy .cache\audit-venv\Scripts\python.exe reports/audit-2026-10-05/reproduce_routing_findings.py
rtk proxy .cache\audit-venv\Scripts\python.exe reports/audit-2026-10-05/check_dependencies.py
```

Ba probe đầu thành công khi tái hiện đúng lỗi hiện tại; `check_dependencies.py` trả mã 1 khi import GDrive lỗi. `check_shell_syntax.py` và `check_shell_lifecycle.py` cần môi trường Linux có Bash; hai script chỉ dùng Python stdlib, có thể chạy trong container mount repo read-only.
