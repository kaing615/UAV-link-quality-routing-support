# Protocol v3: dự báo suy giảm + cổng giá trị hành động

Trạng thái: **đã chạy và dừng theo protocol (2026-11-20).** Tuning 96/96, validation 160/160 hợp lệ; go = false vì Spearman = −0,19 (tiêu chí 5.1), nên không freeze test. Kết quả đã khóa: `reports/uav_service_v3/results/`. Phần dưới đây là bản đăng ký trước, giữ nguyên. Tài liệu này được viết
sau khi đã biết kết quả formal v2 (âm), trước mọi lần chạy v3. Mọi thay đổi
sau khi chạy tuning phải ghi thành bản sửa đổi có ngày và lý do.

Seed plan: [`reports/uav_service_v3/seed-plan.json`](../reports/uav_service_v3/seed-plan.json)
(sinh bởi `python -m src.control.service_v3.cli seed-plan`, kiểm tra bởi
`tests/test_service_v3_protocol.py`). Code: `src/control/service_v3/`.

## 1. Câu hỏi nghiên cứu

Formal v2 cho thấy thêm dự báo vào controller làm PDR giảm 0,986 điểm %
(CI 95% [-1,99; -0,02]) và tăng 41,22 m/episode so với bản bỏ dự báo. Giả
thuyết v3: lỗi nằm ở **khâu biến cảnh báo thành hành động**, không chỉ ở độ
chính xác dự báo. v3 kiểm tra hai sửa đổi, mỗi sửa đổi là một arm riêng:

- **service-v3-threshold**: giữ nguyên controller v2, chỉ thay ngưỡng rủi ro
  0,50 cố định bằng ngưỡng chọn trên dữ liệu tuning.
- **service-v3-gate**: như trên, thêm `ActionValueGate`: một burst chỉ được
  phép khi PDR gain dự kiến trừ chi phí quãng đường còn dương.

## 2. Những gì giữ cố định

- Simulator ns-3, kịch bản (`scenario_config`), `ServiceV2Config` trừ
  `risk_on`/`risk_off` ở hai arm v3.
- Predictor dịch vụ v2 đã fit (`new-model.json` của formal v2): **không fit
  lại**. Nhờ vậy mọi khác biệt chỉ đến từ ngưỡng và cổng hành động.
- Định nghĩa PDR, endpoint sự kiện, bootstrap khối kịch bản như báo cáo v2.
- Không dùng bất kỳ dữ liệu train/calibration/development/pilot/formal v2 nào
  để chọn tham số v3.

## 3. Pha, arm và seed

Seed mới: layout `2400001–2400096`, radio `2500001–2500096`, đã kiểm toán
không trùng 567 layout seed và 347 radio seed trong 13 seed plan v2
(bao gồm toàn bộ kho lịch sử mà v2 đã kiểm toán).

| Pha | Static/gradual mỗi cấu hình (n4, n5) | Kịch bản | Arm | Episode |
|---|---|---|---|---|
| tuning | 8 / 8, cặp static–gradual dùng chung seed | 32 | hold, service-v2, service-v2-no-forecast | 96 |
| validation | 4 / 12 | 32 | 5 arm (thêm v3-threshold, v3-gate) | 160 |
| test | 4 / 20 (như formal v2) | 48 | 5 arm | 240 |

Tổng: 496 episode. Mỗi ô được phép retry đúng một lần nếu lỗi kỹ thuật
hoặc hạ tầng, giống v2. Không thay seed vì PDR thấp, không arm hay không
có outage.

## 4. Pha tuning: chọn tham số (chỉ dùng tuning)

Thứ tự cố định, mỗi bước ghi kết quả vào `tuning-selection.json` trước bước sau:

1. **Horizon**: tính PR-AUC của predictor v2 trên các frame HOLD hợp lệ
   (khỏe, đã arm, nhãn future-only) cho H10 và H20; chọn horizon có PR-AUC
   cao hơn (hòa → H20).
2. **Ngưỡng τ**: `select_operating_point` trên cùng các frame đó với
   `positive_reward=1`, `false_positive_cost=1`, `false_negative_cost=1`,
   `min_recall=0.30`. Arm v3 dùng `risk_on=τ`, `risk_off=τ/2` (giữ tỉ lệ
   0,25/0,50 của v2). Như v2, `risk_on` áp cho cả hai horizon khi xét điểm
   cắt; gate (mục 4.5) còn yêu cầu rủi ro ở horizon đã chọn ≥ τ. Nếu không
   ngưỡng nào đạt recall 0,30, hoặc τ = 0 → dừng v3, báo kết quả âm.
3. **Nhãn giá trị hành động**: với mỗi burst **forecast** bắt đầu tại bước *t*
   trong episode service-v2 (gate chỉ xét burst forecast; burst phục hồi không
   qua gate, và service-v2-no-forecast không có burst forecast), nhãn
   `y = PDR_policy(t, t+30] − PDR_hold(t, t+30]` trên cùng seed (common
   random numbers). Cửa sổ thiếu dữ liệu ở bất kỳ phía nào → loại burst đó
   và đếm riêng. Đây là phản thực tế xấp xỉ: RNG có thể lệch sau khi UAV
   di chuyển; giới hạn này được nêu trong báo cáo.
4. **Mô hình giá trị hành động**: ridge regression, `alpha=1.0`, intercept
   không phạt, đặc trưng chuẩn hóa theo tuning. 9 đặc trưng, đúng những gì
   controller có lúc quyết định và ghi trong decision log: rủi ro H10, H20;
   tỷ lệ dịch vụ online A1, A5, X5, Q5; `plan.total_motion`, `plan.gain` hình
   học, `plan.tau`. Thiếu đặc trưng → burst bị loại khỏi tuning và gate từ
   chối lúc chạy. Đầu ra là `predicted_service_gain` (đơn vị: tỷ lệ PDR). Ghi
   số burst, R², Spearman ρ theo leave-one-out trên cặp seed (static/gradual
   cùng seed ra cùng nhau).
5. **Chi phí quãng đường λ**: chọn trong lưới
   `{0, 1e-4, 2e-4, 5e-4, 1e-3}` (PDR/m; 40 m ở λ=1e-3 tương đương 0,04 PDR) giá trị làm tổng `y` của các burst
   được phép (dùng dự đoán leave-one-scenario-out) lớn nhất; hòa → λ lớn hơn.
   Các tham số còn lại của gate: `min_predicted_gain=0`, `min_net_benefit=0`,
   `max_motion_m=400`.

Sau bước 5: freeze controller v3, model giá trị hành động, τ, λ và source
cho validation.

## 5. Pha validation: go/no-go (không tune)

Validation **không chọn lại** tham số nào. v3 chỉ được chạy test nếu cả ba
điều kiện sau đúng trên các kịch bản gradual (trung bình cân bằng n4/n5):

1. Spearman ρ > 0 giữa `predicted_service_gain` và nhãn `y` (mục 4.3, so với
   HOLD cùng seed) của các burst forecast trong arm v3-threshold và v3-gate.
2. PDR(service-v3-gate) − PDR(service-v2-no-forecast) ≥ −0,005.
3. Quãng đường(service-v3-gate) ≤ quãng đường(service-v2).

Không đạt → dừng, không freeze test, báo kết quả âm cùng số liệu validation.
Đạt → freeze test với đúng package của validation (source, model, τ, λ).

## 6. Pha test: phân tích đã khóa

- **Endpoint chính**: chênh PDR `service-v3-gate − service-v2-no-forecast`
  trên kịch bản gradual, trung bình cân bằng n4/n5. CI 95% từ bootstrap
  10.000 lần trên khối kịch bản, giữ cả 5 arm trong khối, RNG `20261120`.
  Thành công khi cận dưới CI > 0.
- **Kiểm định theo thứ tự cố định** (chỉ diễn giải bước sau nếu bước trước
  thành công):
  1. PDR v3-gate − v2-no-forecast > 0;
  2. quãng đường v3-gate − v2 < 0;
  3. PDR v3-gate − hold > 0.
- **Mô tả, không kiểm định**: v3-threshold so với v2 và v2-no-forecast;
  recall/precision/false-warning của predictor tại τ trên HOLD test; số burst
  bị gate chặn; lý do chặn; static context.
- **Bất biến an toàn** (giữ như v2, báo cáo dạng đếm): không có forecast start
  trước khi arm, không vượt budget, không vi phạm khoảng cách an toàn,
  burst/cooldown đúng quy định.
- Thiếu dữ liệu: báo cáo cặp đủ dữ liệu và biên nhạy cảm chặn PDR trong [0, 1].

## 7. Điều cấm

- Không chạy test trước khi validation đạt go.
- Không đổi τ, λ, horizon, đặc trưng hoặc model sau khi nhìn validation/test.
- Không gộp tuning/validation vào suy luận test.
- Không dùng `plan["gain"]` hình học làm PDR gain (gate từ chối nếu thiếu
  `predicted_service_gain`).

## 8. Cách chạy

Code đã có (`src/control/service_v3/`, test `tests/test_service_v3_*.py`).
Chạy trên server có ns-3, từ thư mục `source/` của package đã freeze:

1. `python -m src.control.service_v3.cli freeze --phase tuning --seed-plan reports/uav_service_v3/seed-plan.json --model <formal v2 new-model.json> --reference-model reports/uav_prevention_pilot_2026-11-11/predictor.json --launcher <ns-3 binary> --ns3-identity ... --compiler-identity ... --output <pkg>/tuning`
2. Mỗi ô: `... cli run-case --protocol <pkg>/tuning/protocol.json --cell-id <id> --attempt 0 --output <run>/<id>/attempt-0`
3. `... cli select --protocol ... --run-root <run> --output tuning-selection.json`
4. `freeze --phase validation --selection tuning-selection.json`, chạy ô, rồi
   `analyze` → `go.json`.
5. Chỉ khi `go.json` có `go: true`: `freeze --phase test --selection ... --inherits <validation protocol.json> --go go.json`, chạy ô, `analyze`.

## 9. Sửa đổi trước khi chạy tuning (2026-11-20)

Khi code, các điểm sau được làm rõ; chưa có dữ liệu v3 nào được thu:
mục 4.2 (τ áp cho cả hai horizon, gate kiểm horizon đã chọn), 4.3 (chỉ burst
forecast của service-v2), 4.4 (9 đặc trưng cụ thể, leave-one-out theo cặp
seed), 5.1 (burst của hai arm v3).
