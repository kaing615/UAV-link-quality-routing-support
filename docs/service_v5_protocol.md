# Protocol v5: ghim route tập trung so với OLSR-ETX phân tán

Trạng thái: **đã chạy xong** (2026-11-22). Validation go=true; pha test: H1 không đạt (mục 8). Đăng ký trước khi chạy,
không sửa nội dung mục 1–7 sau đó. Viết sau khi pha test v4 cho kết quả dương
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

## 8. Kết quả (thêm sau khi chạy)

Dữ liệu: [`reports/uav_service_v5/results/`](../reports/uav_service_v5/results/). Freeze validation `7172828d…`,
go.json `9b66857c…`, freeze test `5785879d…`, analysis test `6676f0a4…`. 96 + 144 episode, tất cả status ok,
không retry.

**Validation: go = true.** Toàn vẹn route 32/32; cờ ETX đúng arm; tỷ lệ giây đi thẳng: hold 0,576, olsr-etx 0,402.

**Test (kiểm định đã khóa, gradual, 40 kịch bản):**

| Bước | Chênh lệch PDR | CI 95% | n4 / n5 | Kết luận |
|---|---|---|---|---|
| 1. lq-route − olsr-etx (H1) | +3,89 điểm % | [−1,11; +8,66] | +5,20 / +2,58 | **không đạt** |
| 2. olsr-etx − hold (H2) | +0,93 điểm % | [−3,67; +5,56] | +1,85 / +0,02 | không diễn giải (bước 1 không đạt) |
| lq-route − hold (mô tả) | +4,82 điểm % | [+3,65; +6,01] | +7,04 / +2,60 | lặp lại v4 |

Toàn vẹn route 48/48; cờ ETX đúng arm. Static: ba arm như nhau.

Theo từng kịch bản: ở n4, lq-route hơn olsr-etx ở 19/20 kịch bản (độ lệch chuẩn 4,1 điểm %). Ở n5, độ lệch chuẩn
của chênh lệch là 22 điểm %, khoảng −56 đến +45.

### 8.1 Chẩn đoán sau khi chạy: ARP mất trắng 100 giây đầu (lỗi mô phỏng, không phải định tuyến)

Các kịch bản n5 cực đoan đều có PDR = 0 trong khoảng 100 giây đầu, dù route OLSR là đi thẳng và link còn tốt.
Tái hiện bằng `scripts/analysis/arp_blackout_repro.py` với test-n5-gradual-03 (hold): mặc định PDR 0,497, khớp server
đến từng cửa sổ 30 giây. Thêm `--ns3::ArpCache::DeadTimeout=+1s` thì PDR là 0,955 và không còn mất trắng. Nguyên
nhân: lần phân giải ARP đầu tiên tới hop kế tiếp thất bại, ns-3 đánh dấu entry "dead" trong `DeadTimeout` = 100 s
(mặc định) và bỏ mọi gói tới hop đó.

Episode có PDR < 5% trong giây 10–60 (chỉ xuất hiện ở n5):

| Pha | hold | lq-route | olsr-etx |
|---|---|---|---|
| v4 validation / test | 1/32, 4/48 | 1/32, 4/48 | — |
| v5 validation | 2/32 | 2/32 | 1/32 |
| v5 test | 3/48 (gradual 03, 18; static 04) | như hold | 4/48 (gradual 06, 07, 09; static 04) |

Ở v4 mọi arm cùng chạy OLSR thường nên lỗi rơi vào cả arm của một kịch bản và gần như triệt tiêu trong so sánh
cặp. Riêng kịch bản test-n5-gradual-18 của v4 (−41,87 điểm %) có phần 100 giây đầu do lỗi này. Ở v5, OLSR-ETX đổi
thời điểm gói điều khiển nên lỗi rơi vào arm khác nhau, tạo nhiễu lớn theo cả hai chiều trong so sánh với
olsr-etx.

**Thăm dò, không phải kiểm định** (quyết định loại sau khi đã thấy dữ liệu): bỏ 5 kịch bản gradual có lỗi ở bất kỳ
arm nào (n5 03, 06, 07, 09, 18), cùng bootstrap:
- lq-route − olsr-etx: +3,60 điểm %, CI [+2,35; +4,92];
- olsr-etx − hold: +1,55, CI [+0,21; +2,89];
- lq-route − hold: +5,15, CI [+3,99; +6,35].

Kết luận chính thức của v5 vẫn là H1 không đạt. Muốn có kết luận xác nhận thì cần một nghiên cứu mới, đăng ký
trước, đã sửa lỗi ARP và dùng seed mới.
