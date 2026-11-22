# Protocol v7: baseline cân bằng thông tin, độ trễ điều khiển và kịch bản rộng hơn

Trạng thái: **đăng ký trước, chưa freeze, chưa chạy** (2026-11-23). Viết sau khi v6 đạt cả hai bước kiểm định
(lq-route − olsr-etx = +4,57 điểm %, CI 95% [3,60; 5,55]) và trước mọi lần chạy v7.

- Seed plan: [`reports/uav_service_v7/seed-plan.json`](../reports/uav_service_v7/seed-plan.json).
- Code: `src/control/service_v7/`, queue `orchestration/service_v7_queue.py`; test: `tests/test_service_v7.py`.

## 1. Câu hỏi

v6 có ba điểm yếu mà phản biện sẽ hỏi. v7 trả lời cả ba trong cùng một thí nghiệm:

1. **Thông tin không cân bằng:** lq-route dùng probe mỗi 0,05 s, còn OLSR-ETX đo chất lượng qua HELLO mỗi 2 s.
   v7 thêm arm **olsr-etx-fast**: OLSR-ETX với HELLO nhanh, chu kỳ được chọn ở pha tuning sao cho baseline mạnh
   nhất.
2. **Điều khiển tập trung lý tưởng:** v7 thêm arm **lq-route-delay**, trong đó mỗi lệnh ghim tuyến có hiệu lực
   sau 2 giây, và **đo chi phí báo hiệu** của mọi arm.
3. **Kịch bản hẹp:** v7 chạy trên chuỗi 5 và **7 UAV** (3 và 5 relay), luôn có thêm **2 luồng UDP nền**.

Giả thuyết (kiểm định theo thứ tự cố định ở mục 5):
- **H1 (chính):** lq-route cho PDR cao hơn olsr-etx-fast.
- **H2:** lq-route-delay cho PDR cao hơn olsr-etx-fast.
- **H3:** lq-route cho PDR cao hơn olsr-etx (lặp lại v6 trong kịch bản rộng hơn).

## 2. Thiết lập

- **Cấu hình:** n5 và n7 (bối cảnh static/gradual như trước, khoảng cách ban đầu 0,10 tầm phủ). Mọi episode
  có `--backgroundFlows=2 --backgroundRateKbps=128` (CBR; luồng nền đi giữa các nút trong chuỗi, cạnh tranh với
  luồng chính 256 kbps).
- **ARP:** `--ns3::ArpCache::DeadTimeout=+1s` ở mọi arm (như v6).
- **Binary:** cùng mã nguồn v6, thêm một bộ đếm chỉ-đọc số gói và số byte phát ở tầng IP theo loại (OLSR, probe,
  dữ liệu, luồng nền), in thành dòng `UAVTX` trong stderr. Kiểm chứng trước đăng ký: trên hai seed thử, binary
  mới cho kết quả giống hệt binary v6 (13.289 và 11.783 gói, khớp từng cửa sổ 30 s). Thư viện ns-3 giống v6.

| Arm | Mô tả |
|---|---|
| `hold` | OLSR thường, relay đứng yên |
| `olsr-etx` | OLSR-ETX, HELLO 2 s, TC 5 s (như v6) |
| `olsr-etx-fast` | OLSR-ETX, HELLO = giá trị chọn ở tuning, TC = 2,5 × HELLO |
| `lq-route` | ghim tuyến của v4, margin 1,0 |
| `lq-route-delay` | như lq-route, nhưng tuyến quyết định ở giây t chỉ có hiệu lực từ giây t + 2 |

## 3. Pha và seed

| Pha | Kịch bản mỗi cấu hình | Arm | Episode |
|---|---|---|---|
| tuning | 6 cặp static+gradual | olsr-etx-h050, -h025, -h010 | 72 |
| validation | 2 static + 8 gradual | 5 arm ở mục 2 | 100 |
| test | 4 static + 20 gradual | như trên | 240 |

Seed: layout 3200001+, radio 3300001+, đã kiểm toán không trùng với 17 seed plan cũ (v2–v6).

**Tuning (chỉ chọn baseline):** HELLO ∈ {0,5; 0,25; 0,1} s (TC = 1,25; 0,625; 0,25 s). Chọn giá trị có PDR
gradual trung bình cân bằng n5/n7 cao nhất; nếu bằng nhau chọn chu kỳ dài hơn (ít báo hiệu hơn). Tuning bị dừng
nếu có lỗi kỹ thuật (cờ sai, thiếu bộ đếm, thiếu số liệu luồng nền, episode mất trắng). Margin của lq-route và độ
trễ 2 s không được chọn lại.

## 4. Validation: go/no-go (kỹ thuật, không xét PDR)

Chạy test khi tất cả đều đúng:
1. Toàn vẹn tuyến ở mọi episode lq-route và lq-route-delay (lệnh đã áp dụng khớp log `UAVROUTE`).
2. Lệnh simulator ghi trong `config.json` kết thúc đúng bằng cờ của arm; cờ OLSR-ETX trong stderr đúng các arm ETX;
   cờ HELLO chỉ có ở olsr-etx-fast.
3. Không episode nào mất trắng lúc khởi động (PDR < 5% trong giây 10–60).
4. Mọi episode có dòng `UAVTX` và số liệu của đủ 2 luồng nền.
5. HELLO nhanh thực sự hoạt động: byte OLSR mỗi giây của olsr-etx-fast > 1,5 lần của olsr-etx.

## 5. Test: phân tích đã khóa

Bootstrap 10.000 lần trên khối kịch bản gradual (giữ cả 5 arm trong mỗi khối), RNG `20261120`, trung bình cân
bằng n5/n7. Ba bước theo thứ tự cố định; mỗi bước thành công khi cận dưới CI 95% > 0, và chỉ được diễn giải khi
mọi bước trước đã thành công:
1. PDR lq-route − olsr-etx-fast (**endpoint chính**, H1);
2. PDR lq-route-delay − olsr-etx-fast (H2);
3. PDR lq-route − olsr-etx (H3).

Chỉ mô tả, không kiểm định:
- olsr-etx-fast − olsr-etx; olsr-etx − hold; lq-route − hold; lq-route − lq-route-delay;
- PDR của luồng nền (tổng hai luồng) cho cùng các cặp, để xem ghim tuyến có làm hại luồng khác không;
- **chi phí báo hiệu** (byte/giây, gradual): OLSR đo được; probe đo được; với lq-route và lq-route-delay thêm
  ước lượng báo cáo về bộ điều khiển (mỗi giây mỗi nút gửi N−1 giá trị chất lượng, 4 B/giá trị + 28 B header, về
  bộ điều khiển đặt tại nguồn, tính theo số hop) và lệnh đổi tuyến. Tổng chi phí của phương pháp: OLSR với các arm
  OLSR; OLSR + probe + báo cáo với hai arm ghim tuyến;
- tỷ lệ thời gian và số gói mất theo loại tuyến; kết quả static; kết quả riêng n5 và n7.

## 6. Điều cấm và giới hạn

- Không đổi tham số, chu kỳ HELLO đã chọn, độ trễ, margin hay cờ sau khi thấy dữ liệu; không chạy test khi
  validation chưa đạt go.
- Độ trễ 2 s chỉ mô hình hóa trễ của vòng điều khiển; không mô hình hóa mất thông điệp điều khiển.
- Chi phí báo cáo về bộ điều khiển là ước lượng giải tích, không được mô phỏng thành gói trong ns-3, nên không
  chiếm kênh truyền trong thí nghiệm.
- Probe chạy ở mọi arm (hạ tầng đo của simulator); chỉ hai arm ghim tuyến cần đến chúng.
- OLSR-ETX vẫn là bản cài đặt của nhóm; một họ kịch bản (chuỗi relay, 802.11g).

## 7. Cách chạy

Như v6, thay `service_v6` bằng `service_v7`, thêm pha tuning và bước `select`:
`freeze --phase tuning` → queue → `validate` → `select --output tuning-selection.json`; sau đó validation và test
dùng `--selection tuning-selection.json`. Binary mới (hash trong `build/recovery/source.sha256`) dùng bộ thư viện
runtime của v5/v6.
