# BÁO CÁO TIẾN ĐỘ ĐỀ TÀI NCKH SINH VIÊN

**Tên đề tài:** Dự đoán chất lượng liên kết đa thời điểm và hỗ trợ định tuyến trong mạng UAV bằng Graph Neural Network  
**Sinh viên thực hiện:** Nguyễn Đình Tâm (23521389), Võ Công Vinh (23521800)  
**Giảng viên hướng dẫn:** ThS. Đặng Lê Bảo Chương  
**Ngày báo cáo:** 08/09/2026  
**Mốc đối chiếu:** Thuyết minh đề tài NCKH sinh viên 2026 và tài liệu “Giải thích công thức, chỉ số, ngưỡng và thuật ngữ”

## 1. Tóm tắt tiến độ

Nhóm đã triển khai prototype end-to-end từ mô phỏng ns-3, tạo graph dataset, gán nhãn QoS/survival tại các mốc `t+1, t+2, t+3, t+5`, huấn luyện mô hình, đánh giá within-run/LORO/cross-mobility/ablation, đến routing replay và thí nghiệm định tuyến ns-3 dựa trên trace. Có 79/79 kiểm thử phần mềm chạy thành công ngày 08/09/2026; kết quả này kiểm tra hành vi code, không thay thế kiểm định dữ liệu, thống kê hoặc tái lập artifact.

Phần triển khai pipeline chủ yếu đã có. Các phần chưa hoàn thành là chạy và nghiệm thu bộ **400 controlled runs**, tái lập toàn bộ artifact theo cùng một phiên bản code–data, bổ sung temporal baseline đã cam kết, thực hiện phân tích xác nhận đã khóa trước, rồi chốt kết luận và bài báo.

Thông điệp khoa học hiện tại là: **các chiến lược dùng dự đoán cho kết quả routing thăm dò tốt hơn, nhưng chưa có cơ sở để khẳng định GNN luôn tốt hơn mô hình dạng bảng**. Ở `t+1`, phép đo trực tiếp của chính cạnh đã mang phần lớn tín hiệu. Trong phép so sánh LORO tại `k=5`, Edge-SAGE cao hơn Logistic Regression; tuy nhiên chưa có paired contrast tương ứng với XGBoost, chưa thể quy chênh lệch cho graph context, kết quả không lặp lại ở cross-mobility và thí nghiệm định tuyến dựa trên trace vẫn có khoảng tin cậy rộng.

## 2. Tiến độ theo mục tiêu trong thuyết minh

| Hạng mục | Trạng thái | Bằng chứng đã có | Việc còn lại |
|---|---|---|---|
| Mô phỏng và dữ liệu nền | Đã có code và artifact bản nền | Pipeline ns-3 với 802.11 ad-hoc, OLSR, Random Waypoint/Gauss–Markov, UDP probe và PHY RSSI; DVC lock ghi nhận bộ 100 run nền | Tải/tái lập lại data trong checkout hiện tại |
| Controlled scenarios | Đã có code, chưa nghiệm thu dữ liệu cuối | Runner và runbook cho 100 seed ghép cặp × 4 scenario: baseline/fast/sparse/dense | Chạy đủ 400 run; manipulation check phải PASS; đóng gói và đưa lên DVC |
| Nhãn đa thời điểm | Đã triển khai | QoS và survival cho `k ∈ {1,2,3,5}`; common support; purge theo horizon; 8 báo cáo pilot (2 target × 4 horizon, mỗi báo cáo 3 run) đều PASS | Chạy kiểm định trên toàn bộ controlled dataset; khóa định nghĩa nhãn và sensitivity protocol trước test cuối |
| Mô hình dự đoán | Hoàn thành nhóm lõi | Persistence, Logistic Regression, XGBoost, GraphSAGE, GAT, Edge-SAGE và các ablation đã có ở các mức khác nhau | Multi-horizon hiện tập trung Persistence/LogReg/XGB/Edge-SAGE; chưa có Edge-SAGE–GRU temporal baseline như thuyết minh |
| Giao thức đánh giá | Đã triển khai, số liệu còn ở quy mô đại diện | Within-run, LORO, cross-mobility, ablation, bootstrap CI theo run, paired test và Holm correction | Chạy lại trên controlled data; chốt trước primary estimand, số run và correction family |
| Routing support | Đã triển khai, kết quả còn thăm dò | Chi phí `-log(p)`, Dijkstra, lọc cạnh, cùng session, multi-horizon replay và thí nghiệm định tuyến ns-3 dựa trên trace | Ấn định số run theo mục tiêu precision/power; khóa ngưỡng quyết định và cost mode trước test cuối |
| Tái lập và triển khai | Phần code đã có | DVC/DVCLive, Docker, FastAPI, CI/CD, GitOps; 79/79 test pass | Checkout hiện thiếu data/outputs và `dvc status` chưa sạch; cần một lần `pull/repro` cuối và cập nhật lock/artifact |
| Báo cáo/bài báo | Đang thực hiện | Đã có bảng LaTeX, hình Stage 7, định tuyến dựa trên trace và benchmark tài nguyên | Viết phần kết quả–thảo luận, giới hạn, threat to validity và hoàn thiện bản thảo bài báo |

## 3. Kết quả sơ bộ từ artifact hiện có — chưa tái lập trên checkout hiện tại

Các số dưới đây được đọc từ artifact đang được Git theo dõi tại commit `c168b19aefd9c8f92f1725ecc27ae721bae5d53c`. Bảng Stage 7 dùng 10 run đại diện; thí nghiệm routing replay sơ khởi dùng 100 run và 240.276 session nhưng đơn vị suy luận phải là run; bảng định tuyến dựa trên trace dùng 10 seed ghép cặp. Vì raw data và output DVC không có trong checkout hiện tại, nhóm chưa kiểm tra lại độc lập run ID, class count và sample count. Bản cuối phải ghi commit, DVC revision, cấu hình, run ID, số run độc lập, training seed và trạng thái exploratory/confirmatory cho từng bảng.

### 3.1. Dự đoán đa thời điểm

Trong các cấu hình được hiển thị, Macro-F1 của QoS tại `k=5` thấp hơn `k=1`. Mỗi snapshot ns-3 dài 1 giây nên `k=5` tương ứng nhìn trước 5 giây. Đây là xu hướng mô tả trên artifact hiện có, chưa phải kiểm định xu hướng theo toàn bộ horizon của bộ controlled cuối:

| Giao thức | Mô hình | Macro-F1 `k=1` | Macro-F1 `k=5` |
|---|---|---:|---:|
| Within-run | Logistic Regression | 0.871 | 0.800 |
| Within-run | XGBoost | 0.863 | 0.798 |
| Within-run | Edge-SAGE | 0.848 | 0.755 |
| LORO | Logistic Regression | 0.877 | 0.819 |
| LORO | XGBoost | 0.881 | 0.835 |
| LORO | Edge-SAGE | 0.880 | **0.856** |
| Cross-mobility | Logistic Regression | 0.883 | 0.825 |
| Cross-mobility | XGBoost | 0.883 | **0.833** |
| Cross-mobility | Edge-SAGE | 0.879 | 0.813 |

Ở LORO QoS `k=5`, Edge-SAGE cao hơn Logistic Regression **0.037 Macro-F1**, run-level percentile-bootstrap 95% CI `[0.023, 0.052]`, two-sided paired sign-flip test với Holm-adjusted `p=0.007812`. Với nhãn survival ở LORO `k=5`, Edge-SAGE đạt 0.847 so với 0.805 của Logistic Regression, chênh **0.042**, 95% CI `[0.030, 0.057]`, Holm-adjusted `p=0.007812`. Trong artifact hiện tại, Holm được áp dụng riêng ở mỗi tổ hợp target–horizon–split trên 8 phép kiểm định: 4 model contrast × 2 metric (Macro-F1 và PR-AUC); correction family này phải được khóa lại trước phân tích xác nhận.

Kết quả này chỉ hỗ trợ phát biểu hẹp: **Edge-SAGE cao hơn Logistic Regression trong hai phép đánh giá LORO `k=5` nói trên**. Báo cáo chưa có paired contrast với XGBoost hoặc ablation dưới cùng giao thức LORO, nên chưa thể nói graph context gây ra chênh lệch. Ở cross-mobility `k=5`, Edge-SAGE cũng chưa vượt XGBoost.

### 3.2. Ablation: tín hiệu thực sự đến từ đâu

Với QoS `k=1`, Edge-SAGE full đạt 0.848 Macro-F1, decoder-only đạt 0.847, message-only đạt 0.693 và no-edge đạt 0.541. Ở `k=5`, full đạt 0.755, decoder-only đạt 0.758, message-only đạt 0.637 và no-edge đạt 0.516.

Kết quả này cho thấy:

- Edge telemetry là nguồn tín hiệu chính.
- Đưa edge features trực tiếp vào decoder gần như đủ để đạt kết quả của mô hình full; decoder-only thậm chí cao hơn full 0.003 tại `k=5`.
- Với số liệu hiện có, ablation **không cho thấy lợi ích của message passing**. Cần paired CI theo run trước khi kết luận về từng thành phần.
- Đóng góp nên được định vị là benchmark và pipeline end-to-end, không tuyên bố phép message passing là một kiến trúc hoàn toàn mới.

### 3.3. Routing replay và định tuyến ns-3 dựa trên trace

Trong routing replay offline sơ khởi trên 100 run và 240.276 session, các chiến lược dùng dự đoán đi kèm route lifetime trung bình tăng từ 0.353 lên khoảng 0.50 snapshot liên tiếp và proxy PDR tại `t+1` tăng từ khoảng 0.10 lên 0.14–0.15, đổi lại khoảng một hop và dưới 1 ms delay. Route lifetime ở đây là số bước liên tiếp trong `[0,5]` mà tuyến chọn tại `t` còn hợp lệ; giá trị thập phân dưới 1 là trung bình qua nhiều session, phần lớn hỏng ngay ở bước đầu. Các session trong cùng run không độc lập; kết quả phụ thuộc giả định replay, chưa chứng minh cải thiện PDR end-to-end hoặc ưu thế riêng của GNN.

Codebase mới đã bổ sung thí nghiệm định tuyến ns-3 dựa trên trace với 10 seed ghép cặp. Cần kiểm tra thêm mức phản hồi của quyết định routing lên traffic và telemetry trước khi gọi đây là closed-loop đầy đủ:

| Chiến lược | PDR | Delay trung bình (ms) | Throughput (Mbps) | Route changes |
|---|---:|---:|---:|---:|
| OLSR | 0.341 | 102.544 | 0.091 | 1.7 |
| Shortest-hop | 0.513 | 111.945 | 0.140 | 5.2 |
| Logistic Regression | 0.532 | 121.380 | 0.144 | 3.6 |
| XGBoost | 0.527 | 95.098 | 0.145 | 5.9 |
| Edge-SAGE | 0.558 | 103.250 | 0.152 | 4.2 |

Edge-SAGE có PDR trung bình cao nhất trong bảng, nhưng chênh so với shortest-hop là `0.044`, 95% CI `[-0.010, 0.098]`, Holm-adjusted `p=0.75`. Khoảng này chứa cả tác động bất lợi nhỏ và tác động có lợi; với 10 run, chưa có bằng chứng về ưu thế.

### 3.4. Khả năng chạy thực tế

Benchmark CPU một luồng trên 10 run cho thấy trung bình của median latency theo snapshot là:

- Logistic Regression: 0.032 ms/snapshot;
- XGBoost: 0.258 ms/snapshot;
- Edge-SAGE: 0.925 ms/snapshot.

Đây chỉ là microbenchmark có mô tả môi trường chưa đầy đủ trong artifact hiện tại; chưa đủ để kết luận đáp ứng thời gian thực trên UAV. Bản cuối phải kèm CPU, batch size, quy mô graph, phân tán và tail latency.

## 4. Các giới hạn cần nói rõ với giảng viên

1. **Artifact Stage 7 hiện dựa trên 10 run đại diện**, trong khi bộ controlled 400 run chưa có report nghiệm thu trong repository.
2. **Data-quality đa thời điểm đã PASS ở pilot 3 run**, chưa thay thế kiểm định toàn bộ controlled data.
3. **Checkout hiện tại thiếu các thư mục data/outputs do DVC quản lý** và nhiều dependency đã thay đổi so với `dvc.lock`; vì vậy các số hiện có là exploratory và cần tái lập lại toàn bộ pipeline trước khi chốt.
4. **Threshold 18 dB, loss 0.10 và delay 10 ms chỉ là cấu hình khởi tạo được nêu trong thuyết minh**, không phải hằng số phổ quát. Giữ chúng làm cấu hình chính để đối chiếu và báo cáo sensitivity; chỉ tune ngưỡng quyết định của classifier/routing trên validation.
5. **Proxy PDR trong replay không phải PDR end-to-end đo trực tiếp**. Thí nghiệm ns-3 dựa trên trace là bằng chứng hệ thống mạnh hơn, nhưng hiện có ít run và cần mô tả rõ cơ chế phản hồi.
6. **Edge-Aware GraphSAGE–GRU temporal baseline chưa được triển khai**, dù đã xuất hiện trong ma trận mô hình của thuyết minh.
7. **Controlled scenario hiện mới bao phủ baseline/fast/sparse/dense**; partition, high-load, failure và telemetry noise vẫn là mở rộng nếu thời gian cho phép.

## 5. Kế hoạch hoàn thành đề xuất

### Tuần 1 — Khóa dữ liệu và giao thức xác nhận

- Lấy lại artifact DVC hoặc sinh lại dữ liệu cần thiết.
- Chạy đủ 400 controlled runs; xác nhận đủ bốn scenario cho từng seed, không tăng/dừng theo p-value quan sát được.
- Chạy manipulation checks và data-quality; chỉ tiếp tục khi report PASS.
- Trước khi mở test cuối, đăng ký một primary estimand gồm target, horizon, protocol, comparator, metric, analysis unit và correction family. Ứng viên cần xin thầy duyệt: Edge-SAGE so với XGBoost, survival `k=5`, LORO, Macro-F1 theo run.
- Giữ bộ ngưỡng nhãn 18 dB/0.10/10 ms làm cấu hình chính và báo cáo sensitivity; chỉ tune ngưỡng quyết định classifier/routing trên validation.

### Tuần 2 — Chạy benchmark chính

- Chạy multi-horizon Persistence/LogReg/XGB/Edge-SAGE trên số run đại diện đã định trước.
- Chạy LORO, cross-mobility, ablation, calibration và threshold sensitivity; prediction đưa vào routing phải là out-of-fold hoặc từ model không học trên session test.
- Bổ sung Edge-SAGE–GRU ở mức baseline tối thiểu để hoàn thành cam kết RQ4; không mở rộng thêm kiến trúc nếu chưa có bằng chứng cần thiết.

### Tuần 3 — Chốt routing và thống kê

- Chạy multi-horizon routing replay trên cùng session.
- Ấn định trước số run định tuyến từ mục tiêu độ rộng CI hoặc power; báo cáo effect size, 95% CI và p-value đã hiệu chỉnh; kiểm tra thêm biến thiên theo training seed.
- Chọn một cấu hình cuối dựa trên validation, không chọn theo test.

### Tuần 4 — Đóng băng artifact và viết báo cáo

- Chạy lại pipeline từ commit đã khóa; cập nhật DVC lock và checksum.
- Hoàn thiện bảng/hình, phần thảo luận, giới hạn và threat to validity.
- Viết bản thảo bài báo và luyện phần trình bày/Q&A.

## 6. Ba vấn đề nên xin ý kiến thầy

1. Có chấp nhận thu hẹp ma trận multi-horizon chính còn Persistence, Logistic Regression, XGBoost và Edge-SAGE; các mô hình còn lại để ở đối chứng `t+1`/phụ lục hay không?
2. Với thí nghiệm định tuyến ns-3 dựa trên trace, 10 run hiện chỉ cho kết quả thăm dò. Thầy muốn ưu tiên ấn định và tăng số run hay ưu tiên hoàn thành temporal baseline trước?
3. Có thống nhất định vị đóng góp là **benchmark tái lập + đánh giá end-to-end + kiểm chứng khi graph context có tạo thêm giá trị**, thay vì cố chứng minh GNN luôn thắng baseline hay không?

## 7. Nội dung nói miệng trong 3–4 phút

> Thưa thầy, so với thuyết minh ban đầu, nhóm đã hoàn thành phần lớn pipeline kỹ thuật từ ns-3 đến đánh giá routing. Dữ liệu được biểu diễn theo snapshot đồ thị; nhóm đã hỗ trợ hai loại nhãn là QoS tại đúng thời điểm tương lai và survival liên tục đến thời điểm đó, với các mốc một, hai, ba và năm bước. Temporal split đã được bổ sung purge để hạn chế leakage.
>
> Về mô hình, kết quả hiện tại không cho thấy GNN luôn thắng baseline. Trong within-run, Logistic Regression và XGBoost vẫn rất mạnh vì telemetry tại thời điểm hiện tại đã chứa nhiều thông tin về liên kết ở bước kế tiếp. Ở LORO tại k bằng năm, Edge-SAGE cao hơn Logistic Regression khoảng 0.037 Macro-F1 cho QoS và 0.042 cho survival, với khoảng tin cậy không đi qua không. Đây là tín hiệu riêng của Edge-SAGE so với Logistic Regression; chưa xác lập lợi ích của graph context hoặc ưu thế so với baseline dạng bảng mạnh nhất.
>
> Ablation cũng cho thấy edge features ở decoder là nguồn tín hiệu chính; message passing có xét edge chưa tạo lợi thế rõ. Vì vậy nhóm định vị đóng góp ở pipeline benchmark và đánh giá end-to-end, không đặt nặng tuyên bố kiến trúc mới.
>
> Ở mức hệ thống, routing replay cho thấy tuyến có hỗ trợ dự đoán đi kèm route lifetime trung bình cao hơn shortest-hop. Thí nghiệm định tuyến dựa trên trace với 10 run cũng cho Edge-SAGE có PDR trung bình 0.558 so với 0.513 của shortest-hop, nhưng khoảng tin cậy còn rộng và kiểm định chưa có ý nghĩa thống kê. Công việc tiếp theo là nghiệm thu 400 controlled runs, chạy lại toàn bộ benchmark theo cùng phiên bản dữ liệu, bổ sung temporal baseline, rồi mới chốt kết luận và bài báo. Nhóm xin ý kiến thầy về phạm vi mô hình chính và ưu tiên tài nguyên cho tăng số run định tuyến.

## 8. Gợi ý slide trình bày 8–10 phút

| Slide | Tiêu đề theo dạng kết luận | Nội dung/visual chính | Thời lượng |
|---:|---|---|---:|
| 1 | Topology UAV thay đổi nhanh làm thông tin định tuyến lỗi thời | Một hình `t` và `t+k`: cạnh B–D đang tốt rồi bị đứt | 40 giây |
| 2 | Dự đoán hỗ trợ routing có triển vọng; lợi thế của GNN chưa được xác lập | Ba câu ngắn: dự đoán xa, run chưa thấy, tác động routing | 45 giây |
| 3 | Pipeline đã chạy end-to-end; bằng chứng cuối còn chờ 400 controlled runs | Sơ đồ 6 khối phía trên, thanh tiến độ xanh/vàng phía dưới | 75 giây |
| 4 | Chất lượng dự đoán giảm khi nhìn xa hơn | Dùng `reports/stage7/figures/multihorizon_macro_f1.pdf`; chỉ highlight `k=1` và `k=5` | 60 giây |
| 5 | Edge-SAGE hơn Logistic Regression trên run chưa thấy, chưa hơn khi kiểu di chuyển thay đổi | Dùng `ood_macro_f1.pdf`; chỉ ghi `k=5`, để LORO/OOD trong chú thích nhỏ | 70 giây |
| 6 | Phép đo trực tiếp của liên kết giải thích gần hết hiệu năng Edge-SAGE | Dùng `edge_sage_ablation.pdf`; gọi message passing là “thông tin lân cận” | 60 giây |
| 7 | Predictive routing có PDR trung bình cao hơn, nhưng chênh lệch chưa chắc chắn | Dùng `closed_loop_comparison.pdf`; ghi `n=10`, hiện CI và không tô đậm model thắng | 70 giây |
| 8 | Ba việc còn lại: controlled data, temporal baseline, artifact cuối | Timeline bốn tuần; nói một câu hỏi ưu tiên, để hai câu còn lại trên slide | 60 giây |

### Cách dùng thuật ngữ từ tài liệu Definitions/Explanations

Chỉ giải thích mục nào cần cho biểu đồ đang chiếu; chuyển công thức, purge, Holm và calibration chi tiết sang slide dự phòng để không vỡ thời lượng.

- Trên slide đầu tiên có ký hiệu, nói: **snapshot là ảnh chụp trạng thái mạng trong cửa sổ 1 giây; `k` là số giây/bước nhìn về tương lai trong cấu hình hiện tại**.
- Phân biệt: **QoS `t+k` chỉ kiểm tra đúng một thời điểm; survival yêu cầu liên kết sống liên tục từ `t+1` đến `t+k`**.
- Khi hiện `-log(p)`, chỉ giải thích một câu: tổng chi phí nhỏ tương ứng với tích xác suất sống của cả tuyến lớn, dưới giả định rủi ro cạnh độc lập.
- Khi hiện Macro-F1, nói đây là F1 trung bình của hai lớp với trọng số bằng nhau, phù hợp hơn accuracy khi lệch lớp.
- Mỗi biểu đồ phải ghi số run và 95% CI; không dùng từng edge phụ thuộc nhau như quan sát độc lập.
- Luôn gọi PDR của replay là **proxy PDR**; chỉ gọi PDR đo trực tiếp khi lấy từ traffic ns-3 và mô tả rõ protocol.

### Slide dự phòng cho phần hỏi đáp

1. Định nghĩa nhãn QoS và survival.
2. Nguồn gốc RSSI/SNR/loss/delay trong ns-3 và khác biệt với simulator Python.
3. Temporal split và purge chống leakage.
4. Công thức `-log(p)` và trade-off route-found rate.
5. Bảng latency/RAM.
6. Bảng paired effect, 95% CI và Holm-adjusted p-value.

## 9. Câu trả lời ngắn cho các câu hỏi dễ bị hỏi

**Vì sao vẫn dùng GNN khi Logistic Regression/XGBoost đang mạnh?**  
Vì câu hỏi nghiên cứu là khi nào graph context tạo thêm giá trị, không phải mặc định GNN phải thắng. Kết quả LORO `k=5` cho tín hiệu tích cực, còn ablation và cross-mobility giúp xác định giới hạn.

**Edge-Aware GraphSAGE có phải kiến trúc hoàn toàn mới không?**  
Không. Việc đưa edge features vào message passing có liên hệ với MPNN và edge-conditioned convolution. Đóng góp chính là cách chuẩn hóa bài toán UAV, thiết kế benchmark, ablation và đánh giá tác động lên routing.

**Các ngưỡng 18 dB, 10% loss, 10 ms có cơ sở nào?**  
Đó là cấu hình khởi tạo được nêu trong thuyết minh để đối chiếu pipeline sơ khởi, chưa phải hằng số phổ quát. Nhóm giữ nó làm cấu hình chính, báo cáo sensitivity analysis và không dùng test để chọn lại định nghĩa nhãn; validation chỉ dùng để tune ngưỡng quyết định của classifier/routing.

**PDR tăng đã được chứng minh chưa?**  
Chưa. Replay có trung bình tốt hơn nhưng dùng proxy PDR. Thí nghiệm ns-3 dựa trên trace có PDR trung bình cao hơn, song với 10 run thì CI còn rộng và kiểm định chưa đủ mạnh; cần mô tả rõ cơ chế phản hồi và ấn định số run trước khi tuyên bố chính thức.

**Đã chống data leakage chưa?**  
Code đã có temporal split và purge theo horizon. Việc còn lại là tái lập full controlled pipeline và xác nhận artifact cuối dùng đúng split đã khóa.

## 10. Artifact nên mang theo khi gặp thầy

- [Kết quả multi-horizon và OOD](../reports/stage7/figures/multihorizon_macro_f1.pdf)
- [Ablation Edge-SAGE](../reports/stage7/figures/edge_sage_ablation.pdf)
- [So sánh định tuyến ns-3 dựa trên trace](../reports/closed_loop/artifacts/figures/closed_loop_comparison.pdf)
- [Bảng effect ghép cặp](../reports/stage7/tables/paired_effects.tex)
- [Bảng tài nguyên suy luận](../reports/inference_benchmark/tables/inference_resources.tex)
- [Runbook controlled scenarios](CONTROLLED_SCENARIOS_RUNBOOK.md)
