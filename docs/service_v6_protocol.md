# Protocol v6: v5 lặp lại sau khi sửa lỗi ARP của ns-3

Trạng thái: **đăng ký trước, chưa freeze, chưa chạy** (2026-11-22). Viết sau khi có kết quả v5 (H1 không đạt) và
chẩn đoán lỗi ARP (mục 8.1 của [protocol v5](service_v5_protocol.md)), trước mọi lần chạy v6.

- Seed plan: [`reports/uav_service_v6/seed-plan.json`](../reports/uav_service_v6/seed-plan.json); thiết lập cố định:
  [`reports/uav_service_v6/selection.json`](../reports/uav_service_v6/selection.json).
- Code: `src/control/service_v6/`, queue `orchestration/service_v6_queue.py`; test: `tests/test_service_v6.py`.

## 1. Câu hỏi và lý do

Ở v5, lỗi ARP của ns-3 làm một số episode n5 mất trắng khoảng 100 giây đầu. Lỗi rơi ngẫu nhiên vào các arm khác
nhau, nên CI của H1 rộng ra và không loại được 0. v6 lặp lại **đúng** thiết kế v5, chỉ thêm bản sửa lỗi này, trên
seed mới. Giả thuyết, arm, tham số và phân tích đều không đổi:

- **H1 (chính):** lq-route cho PDR cao hơn OLSR-ETX phân tán.
- **H2:** OLSR-ETX cho PDR cao hơn OLSR thường (hold).

v6 được quyết định sau khi đã thấy v5, nên v5 vẫn được báo cáo đầy đủ cùng với v6, không thay thế.

## 2. Thay đổi duy nhất so với v5: ARP

Mọi arm chạy với `--ns3::ArpCache::DeadTimeout=+1s` (mặc định của ns-3 là 100 s). Khi một lần phân giải ARP thất
bại, ns-3 giữ entry "dead" suốt `DeadTimeout` và bỏ mọi gói tới hop đó. Linux thì phân giải lại ở gói kế tiếp; 1 s
gần với hành vi này. Binary, thư viện OLSR-ETX đã vá (`c3777857…`), margin 1,0 và tham số OLSR-ETX giống hệt v5.

Kiểm chứng trước đăng ký (seed v5, đã dùng; `scripts/analysis/arp_blackout_repro.py`): cả 10 episode từng mất trắng
ở v5 (hold: test gradual 03, 18, static 04, validation gradual 04, 08; olsr-etx: test gradual 06, 07, 09, static 04,
validation gradual 04) đều hết mất trắng khi có bản sửa; PDR từng episode 0,87–0,98.

## 3. Pha, arm và seed

| Pha | Static/gradual mỗi cấu hình | Kịch bản | Arm | Episode |
|---|---|---|---|---|
| validation | 4/12 | 32 | hold, olsr-etx, lq-route | 96 |
| test | 4/20 | 48 | như trên | 144 |

Seed: layout 3000001–3000080, radio 3100001–3100080, đã kiểm toán không trùng với 16 seed plan cũ (v2–v5). Relay
không di chuyển ở cả ba arm. Mỗi ô được retry một lần nếu lỗi kỹ thuật.

## 4. Validation: go/no-go (kiểm tra kỹ thuật, không xét PDR)

Chạy test khi cả năm điều kiện đều đúng:
1. Toàn vẹn route của lq-route ở mọi episode.
2. Dòng `UAVOLSR link_quality=etx` có trong log của mọi và chỉ episode olsr-etx.
3. Tỷ lệ số giây đi thẳng (gradual) của olsr-etx thấp hơn của hold.
4. Lệnh simulator ghi trong `config.json` của mọi episode có cờ ARP.
5. Không episode nào mất trắng lúc khởi động: PDR < 5% trong giây 10–60, khi relay còn đứng yên và mọi link tốt.

## 5. Test: phân tích đã khóa (giống v5)

Bootstrap 10.000 lần trên khối kịch bản gradual, giữ cả 3 arm trong mỗi khối, RNG `20261120`, trung bình cân
bằng n4/n5. Hai bước kiểm định theo thứ tự cố định; mỗi bước thành công khi cận dưới CI 95% > 0:
1. PDR lq-route − olsr-etx (**endpoint chính**, H1);
2. PDR olsr-etx − hold (H2).

Chỉ mô tả: lq-route − hold; tỷ lệ thời gian và số gói mất theo loại route; kịch bản static; danh sách episode mất
trắng (nếu có).

## 6. Điều cấm và giới hạn

- Không đổi tham số, margin, cờ ARP hay ngưỡng mất trắng sau khi đã thấy dữ liệu; không chạy test khi validation chưa
  đạt go.
- Không loại kịch bản nào khỏi kiểm định ở mục 5.
- OLSR-ETX là bản cài đặt của nhóm theo olsrd; lq-route là điều khiển tập trung lý tưởng (như v4, v5).

## 7. Cách chạy

Như v5, thay `service_v5` bằng `service_v6`; dùng cùng binary (`240123d2…`) và bộ thư viện runtime của v5.
