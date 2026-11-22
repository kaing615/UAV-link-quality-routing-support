# Chẩn đoán: vì sao di chuyển relay không cải thiện PDR (2026-11-20)

Phân tích đọc lại dữ liệu đã khóa, không chạy mô phỏng mới và không fit mô hình.
Đây là phân tích **khám phá, chưa đăng ký trước**; các con số là tương quan, chưa phải tác động nhân quả.

- Dữ liệu: formal v2, 384 episode, 48 kịch bản × 8 policy (`reports/uav_service_prevention_v2_2026-11-12/runs/formal`).
- Lệnh tạo lại:
  `python scripts/analysis/diagnose_motion_harm.py --validation <formal-analysis/validation.json> --run-root <runs/formal> --output reports/uav_service_v3/diagnosis/v2-formal.json`
- Kết quả đầy đủ: [`v2-formal.json`](v2-formal.json).

## 1. Phần lớn gói bị mất khi OLSR đi thẳng qua một link yếu

Các con số dưới đây là của kịch bản gradual, policy HOLD (24 episode, 9.600 giây).

| Route OLSR đang dùng | Thời gian | PDR | Link yếu nhất (trung vị) | Phần gói mất |
|---|---|---|---|---|
| Đi thẳng nguồn → đích | 59% | 0,885 | 0,75 | 52% |
| Qua relay | 34% | 0,948 | 0,55 | 13% |
| Không có route | 6% | 0,283 | – | 35% |

Khi chỉ xét các giây route đã ổn định hơn 5 giây:

| Loại route | PDR | Phần gói mất |
|---|---|---|
| Đi thẳng | 0,900 | 39% |
| Qua relay | 0,966 | 3% |

Mọi policy trong 8 policy đều có cùng mẫu hình: khoảng 59% thời gian đi thẳng. OLSR chọn route theo số hop. Chừng nào link trực tiếp còn nhận được HELLO, route 1 hop vẫn được giữ, dù link đó mất nhiều gói. Trong phần lớn các giây đi thẳng (khi link trực tiếp ≥ 0,3) có sẵn một đường qua relay với link yếu nhất ≥ 0,9. Khi link trực tiếp < 0,3 thì vẫn có đường qua relay ≥ 0,5, nhưng PDR chỉ còn 0,40–0,76. Đây là vấn đề kinh điển của thước đo min-hop, và là lý do ETX được đề xuất (De Couto và cộng sự, MobiCom 2003).

Di chuyển relay không chạm tới phần mất gói này, vì route lúc đó không đi qua relay.
Cả 94 forecast burst của service-v2 đều bắt đầu khi route **đã** đi qua relay.

## 2. Route churn có làm mất gói, nhưng không phải nguồn chính

- HOLD đổi route trung bình 19,7 lần mỗi episode; các policy có di chuyển đổi 20,2–23,2 lần.
- Trong 1–5 giây sau khi đổi route, PDR thấp hơn (đi thẳng 0,778; không có route 0,296 so với ổn định). Các giây này chiếm khoảng 33% số gói mất của HOLD.
- Di chuyển chỉ làm tăng thêm khoảng 2–3 lần đổi route mỗi episode.

## 3. Nhãn PDR gain của v3 bị nhiễu do lịch sử can thiệp

Tại thời điểm forecast burst bắt đầu, PDR theo từng giây của trace service-v2 đã cao hơn HOLD cùng seed **+0,11 (5 giây trước) và +0,18 (tại t = 0)**. Hai trace đã lệch nhau từ trước vì những lần di chuyển phục hồi trước đó. Nhãn "chênh 30 giây so với HOLD" vì vậy không đo được tác động của riêng burst đang xét. Đây là một lý do hợp lý giải thích vì sao Spearman của mô hình gain âm ở cả tuning (−0,02) và validation (−0,19).

## 4. Giới hạn

- Đây là tương quan. Các giây đi thẳng có thể khác các giây qua relay về hình học hoặc thời điểm trong episode, nên mức chênh 0,90 và 0,966 chưa chắc là lợi ích khi ép route qua relay.
- Thử ước lượng dư địa bằng tích các rx_ratio đo bằng probe: **không dùng được**. Kết quả chỉ 0,57 trong khi PDR thực tế là 0,91, vì probe là broadcast không gửi lại còn dữ liệu unicast được MAC gửi lại.
- Chỉ phân tích dữ liệu v2 formal. Dữ liệu v3 trên server chưa được phân tích.

## 5. Hệ quả cho v4

Biến điều khiển hứa hẹn nhất là **chọn route theo chất lượng link** (ép đi qua relay khi link trực tiếp yếu), không phải vị trí relay. Simulator đã có lớp `PlannedRouting` (route tĩnh đè OLSR), nhưng hiện chỉ đọc kế hoạch soạn sẵn, và bị chặn trong kịch bản relay. Muốn điều khiển route trực tuyến cần sửa C++ và build lại binary.
