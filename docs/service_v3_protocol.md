# Protocol v3: dự báo suy giảm + cổng giá trị hành động

Trạng thái: **đăng ký trước, chưa freeze, chưa chạy.** Tài liệu này được viết
sau khi đã biết kết quả formal v2 (âm), trước mọi lần chạy v3. Mọi thay đổi
sau khi chạy tuning phải ghi thành bản sửa đổi có ngày và lý do.

Seed plan: [`reports/uav_service_v3/seed-plan.json`](../reports/uav_service_v3/seed-plan.json)
(sinh bởi `python -m src.control.service_v3.protocol`, kiểm tra bởi
`tests/test_service_v3_protocol.py`).

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
   0,25/0,50 của v2). Nếu không ngưỡng nào đạt recall 0,30 → dừng v3, báo
   kết quả âm.
3. **Nhãn giá trị hành động**: với mỗi burst bắt đầu tại bước *t* trong
   episode service-v2 hoặc service-v2-no-forecast, nhãn
   `y = PDR_policy(t, t+30] − PDR_hold(t, t+30]` trên cùng seed (common
   random numbers). Cửa sổ thiếu dữ liệu ở bất kỳ phía nào → loại burst đó
   và đếm riêng. Đây là phản thực tế xấp xỉ: RNG có thể lệch sau khi UAV
   di chuyển; giới hạn này được nêu trong báo cáo.
4. **Mô hình giá trị hành động**: ridge regression, `alpha=1.0`, đặc trưng
   chuẩn hóa theo tuning: xác suất rủi ro H10/H20, đặc trưng dịch vụ tại *t*
   (`features.py`), `plan.total_motion`, `plan.gain` hình học. Đầu ra là
   `predicted_service_gain` (đơn vị: tỷ lệ PDR). Ghi số burst, R², Spearman ρ
   theo leave-one-scenario-out.
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

1. Spearman ρ giữa `predicted_service_gain` và nhãn `y` của các burst
   validation > 0.
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

## 8. Việc cần code trước khi freeze tuning

1. Mở rộng freeze/run v2 để nhận phase và policy v3 (`service_v3/protocol.py`
   đã có seed plan và danh sách arm theo pha).
2. Arm `service-v3-threshold`/`service-v3-gate` trong controller factory, với
   `risk_on`/`risk_off` lấy từ `tuning-selection.json`.
3. Trích nhãn burst `y` (mục 4.3) và fit ridge (mục 4.4–4.5).
4. Phân tích validation go/no-go và test theo mục 5–6.

Nếu các việc này làm lộ ra điều cần sửa trong protocol, phải sửa tài liệu này
trước khi chạy tuning.
