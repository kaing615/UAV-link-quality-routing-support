# Protocol v5: ghim route tập trung so với OLSR-ETX phân tán

Trạng thái: **đăng ký trước, chưa freeze, chưa chạy** (2026-11-22). Viết sau khi pha test v4 cho kết quả dương
(lq-route − hold = +4,18 điểm %, CI 95% [1,14; 6,36]) và trước mọi lần chạy v5.

- Seed plan: [`reports/uav_service_v5/seed-plan.json`](../reports/uav_service_v5/seed-plan.json); thiết lập cố định:
  [`reports/uav_service_v5/selection.json`](../reports/uav_service_v5/selection.json).
- Code: `src/control/service_v5/`; patch ns-3: `simulation/ns3/patches/ns-3.45-olsr-etx.patch`;
  test: `tests/test_service_v5.py`, `tests/cpp/test_olsr_etx_header.cc`.

## 1. Câu hỏi

Phản biện chắc chắn sẽ hỏi: lợi ích của v4 đến từ việc dùng ETX (đã biết từ De Couto 2003, có sẵn trong olsrd
OLSR-ETX), hay từ cách điều khiển tập trung? v5 so sánh trực tiếp:

- **H1 (chính):** ghim route tập trung (lq-route) cho PDR cao hơn OLSR-ETX phân tán.
- **H2:** OLSR-ETX phân tán cho PDR cao hơn OLSR thường (hold).

## 2. OLSR-ETX phân tán (bản vá ns-3.45)

Bật bằng attribute `ns3::olsr::RoutingProtocol::LinkQualityEtx` (cờ simulator `--olsrLinkQuality=etx`). Khi tắt,
định dạng gói và hành vi không đổi.

- **Đo LQ:** EWMA trên số thứ tự gói OLSR của từng hàng xóm. Gói mất (khoảng trống số thứ tự) và mỗi 1,5 chu kỳ
  HELLO im lặng đều tính là mất. Hệ số `LinkQualityAging` = 0,1. Gói đầu tiên đặt LQ = 1.
- **NLQ:** lấy từ LQ_HELLO (loại 201) của hàng xóm; mỗi địa chỉ có thêm 4 byte (LQ, NLQ, dự phòng), như olsrd.
- **LQ_TC** (loại 202) quảng bá **mọi** hàng xóm đối xứng kèm LQ/NLQ (giống TC_REDUNDANCY = 2 của olsrd); mỗi
  lần gửi tăng ANSN để thay toàn bộ tập quảng bá.
- **Route:** Dijkstra theo ETX = 1/(LQ·NLQ) trên link của chính nút, link 2-hop (từ LQ_HELLO) và topology (từ
  LQ_TC). Link có LQ·NLQ < 10⁻³ không dùng. Chọn MPR vẫn theo RFC 3626.
- HELLO 2 s, TC 5 s (mặc định ns-3), không chỉnh.

Kiểm chứng trước đăng ký (seed formal v2 1200055/1300055, đã dùng):
- Với binary mới, HOLD ở chế độ mặc định nhận đúng 12.596/15.000 gói, như formal v2.
- OLSR-ETX giảm số giây đi thẳng từ 127 xuống 69 và số gói mất trên route đi thẳng từ 284 xuống 0.
- Test round-trip header LQ_HELLO/LQ_TC đạt.

## 3. Pha, arm và seed

Không có tham số nào được chọn: margin của lq-route cố định 1,0 (selection v4 `e498e159…`), OLSR-ETX dùng các
giá trị ở mục 2. Vì vậy không có pha tuning.

| Pha | Static/gradual mỗi cấu hình | Kịch bản | Arm | Episode |
|---|---|---|---|---|
| validation | 4/12 | 32 | hold, olsr-etx, lq-route | 96 |
| test | 4/20 | 48 | như trên | 144 |

Seed: layout 2800001–2800080, radio 2900001–2900080, đã kiểm toán không trùng với 15 seed plan cũ (v2, v3, v4).
Mọi arm chạy cùng binary, cùng thư viện ns-3 đã vá và cùng stack `--routeControl=external`. Relay không di
chuyển ở cả ba arm. Mỗi ô được retry một lần nếu lỗi kỹ thuật.

## 4. Validation: go/no-go (kiểm tra kỹ thuật, không xét PDR)

Chạy test khi cả ba điều kiện đều đúng:
1. Toàn vẹn route của lq-route ở mọi episode (như v4).
2. Dòng `UAVOLSR link_quality=etx` có trong log của **mọi và chỉ** episode olsr-etx.
3. Tỷ lệ số giây đi thẳng (gradual) của olsr-etx thấp hơn của hold, tức là ETX thực sự hoạt động.

Không dùng PDR để quyết định go, nên kết quả test không bị chọn lọc theo chiều nào.

## 5. Test: phân tích đã khóa

Bootstrap 10.000 lần trên khối kịch bản gradual, giữ cả 3 arm trong mỗi khối, RNG `20261120`, trung bình cân
bằng n4/n5. Hai bước kiểm định theo thứ tự cố định; mỗi bước thành công khi cận dưới CI 95% > 0:
1. PDR lq-route − olsr-etx (**endpoint chính**, H1);
2. PDR olsr-etx − hold (H2).

Chỉ mô tả, không kiểm định:
- lq-route − hold (lặp lại v4 trên seed mới);
- tỷ lệ số giây đi thẳng / qua relay / không có route và số gói mất theo từng loại;
- kết quả ở kịch bản static.

## 6. Điều cấm và giới hạn

- Không đổi tham số OLSR-ETX hay margin sau khi đã thấy dữ liệu; không chạy test khi validation chưa đạt go.
- OLSR-ETX ở đây là bản cài đặt của nhóm theo olsrd, không phải chính olsrd. Bản vá được công bố để kiểm tra lại.
- lq-route vẫn là điều khiển tập trung lý tưởng (giới hạn đã nêu ở v4).

## 7. Cách chạy

Như v4, thay `service_v4` bằng `service_v5`. Bước chọn tham số được thay bằng `selection.json` cố định ở trên.
`LD_LIBRARY_PATH` phải trỏ tới bộ thư viện runtime có `libns3.45-olsr-default.so` đã vá (hash `c3777857…`).
