# Service prevention v3: kế hoạch tối ưu có kiểm soát

Tài liệu này mô tả lớp tối ưu **opt-in** được thêm sau formal v2. Formal v2,
model đã fit, seed, raw data, report và PDF không bị sửa.

## Những gì đã có trong code

`src/control/service_v2/optimization.py` cung cấp:

- `select_operating_point(...)`: chọn threshold trên validation bằng utility
  có chi phí false positive/false negative và ràng buộc recall tối thiểu.
- `OperatingPoint`: lưu threshold, confusion counts, precision, recall, FPR,
  utility và nguồn chọn threshold.
- `ActionValueGate`: chỉ cho phép một burst khi risk, predicted service gain,
  quãng đường và net benefit cùng đạt ngưỡng.
- `advance_motion(..., action_gate=...)` và `ServiceV2Controller(...,
  action_gate=...)`: gate là tùy chọn; bỏ qua gate thì hành vi v2 giữ nguyên.

`predicted_service_gain` bắt buộc phải đến từ action-value model được đánh giá
trên validation. Không được dùng `plan["gain"]` hình học để giả làm PDR gain.

## Quy trình chạy v3

Protocol đăng ký trước, có số lượng seed và tiêu chí go/no-go cụ thể: [service_v3_protocol.md](service_v3_protocol.md).

1. Giữ formal v2 làm baseline bất biến.
2. Tạo seed validation mới, tách khỏi train/calibration/formal cũ.
3. Sinh xác suất và nhãn future-only trên validation.
4. Dùng `select_operating_point` để chọn threshold. Lưu toàn bộ chi phí,
   confusion counts và recall floor vào manifest.
5. Thu thập các trạng thái có/không can thiệp trong một protocol mới để học
   `predicted_service_gain`. Không fit action-value model từ formal v2.
6. Chạy `ActionValueGate` trên validation; ghi số cảnh báo, số burst bị chặn,
   PDR, movement và net benefit.
7. Chỉ sau khi gate validation đạt yêu cầu mới freeze seed test mới.
8. So sánh v3 với HOLD, v2 và v2-no-forecast bằng scenario-paired bootstrap.

## Mục tiêu đánh giá

- H10 recall tăng so với baseline hiện tại.
- False-warning fraction giảm.
- PDR không giảm khi thêm forecast.
- Movement giảm hoặc PDR gain đủ bù movement cost.
- Không có forecast start trước arming, không vượt budget, không vi phạm
  separation và burst/cooldown.

## Điều chưa được tuyên bố

Code mới chưa chứng minh v3 tốt hơn v2. Chưa có action-value model được fit,
chưa có seed validation/test mới và chưa có formal v3. Các kết quả formal v2
vẫn là kết quả âm đã báo cáo; mọi claim cải thiện cần một protocol mới.
