# Protocol v6: v5 lặp lại sau khi sửa lỗi ARP của ns-3

Trạng thái: **đã chạy xong** (2026-11-22). Validation go=true; pha test: cả hai bước kiểm định đạt (mục 8). Mục 1–7
viết trước khi chạy và không sửa sau đó. Viết ngày 2026-11-22, sau khi có kết quả v5 (H1 không đạt) và
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

## 8. Kết quả (thêm sau khi chạy)

Dữ liệu: [`reports/uav_service_v6/results/`](../reports/uav_service_v6/results/). Freeze validation `3d71846b…`,
go.json `aa348bb6…`, freeze test `9921c242…`, analysis test `7b7d1413…`. 96 + 144 episode, tất cả status ok,
không retry. Ở cả hai pha: toàn vẹn route đạt (32/32, 48/48), cờ ETX và cờ ARP đúng ở mọi episode, **không episode
nào mất trắng**.

**Validation: go = true** (đủ 5 điều kiện). Tỷ lệ giây đi thẳng: hold 0,575, olsr-etx 0,398.

**Test (kiểm định đã khóa, gradual, 40 kịch bản):**

| Bước | Chênh lệch PDR | CI 95% | n4 / n5 | Kết luận |
|---|---|---|---|---|
| 1. lq-route − olsr-etx (H1) | **+4,57 điểm %** | **[+3,60; +5,55]** | +4,71 / +4,42 | **đạt** |
| 2. olsr-etx − hold (H2) | +1,01 điểm % | [+0,10; +1,91] | +1,45 / +0,56 | đạt |
| lq-route − hold (mô tả) | +5,58 điểm % | [+4,53; +6,63] | +6,16 / +4,99 | lặp lại v4 |

PDR trung bình (gradual): hold 0,921; olsr-etx 0,931; lq-route 0,977. Static: ba arm như nhau (≥ 0,999).

Theo từng kịch bản (mô tả): lq-route hơn olsr-etx ở 39/40 kịch bản (n4 19/20, n5 20/20); kịch bản duy nhất thua là n4, −1,1 điểm %.
Độ lệch chuẩn của chênh lệch ở n5 là 2,8 điểm % (v5: 22 điểm %). olsr-etx hơn hold ở 24/40 kịch bản.

Số gói mất theo loại route OLSR quan sát được (gradual, tổng 40 kịch bản):

| Arm | Đi thẳng | Qua relay | Không có route |
|---|---|---|---|
| hold | 17.685 | 9.079 | 20.617 |
| olsr-etx | 880 | 6.185 | 34.271 |
| lq-route | 1.089 | 6.981 | 5.857 |

OLSR-ETX gần như bỏ hết mất gói trên link đi thẳng yếu, nhưng mất nhiều hơn hẳn trong các giây không có route
(chuyển route chậm qua HELLO 2 s / TC 5 s). lq-route giảm được cả hai loại.

**Kết luận:** sau khi sửa lỗi ARP, cả H1 và H2 đều đạt. Ghim route tập trung theo ETX tốt hơn OLSR-ETX phân tán
+4,57 điểm %, và OLSR-ETX tốt hơn OLSR thường một chút (+1,01 điểm %). Giới hạn: lq-route là điều khiển tập trung lý
tưởng (biết chất lượng link của toàn mạng, lệnh tới ngay); OLSR-ETX là bản cài đặt của nhóm; một họ kịch bản
(chuỗi relay 4–5 UAV, 802.11g).
