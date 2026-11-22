# Protocol v4: điều khiển route theo chất lượng link

Trạng thái: **đăng ký trước, chưa freeze, chưa chạy** (2026-11-21). Tài liệu này được viết sau kết quả âm của v2
và v3 và sau phần [chẩn đoán](../reports/uav_service_v3/diagnosis/README.md), trước mọi lần chạy v4.
Mọi thay đổi sau khi chạy tuning phải ghi thành bản sửa đổi có ngày và lý do.

- Seed plan: [`reports/uav_service_v4/seed-plan.json`](../reports/uav_service_v4/seed-plan.json), sinh bởi `python -m src.control.service_v4.cli seed-plan`.
- Code: `src/control/service_v4/`, `simulation/ns3/route-protocol.h`, `OverrideRouting` trong `simulation/ns3/uav-olsr-dataset.cc`.
- Test: `tests/test_service_v4_*.py`, `tests/cpp/test_route_protocol.cc`.

## 1. Câu hỏi và giả thuyết

Phần chẩn đoán trên formal v2 cho thấy:
- Khi HOLD ở kịch bản gradual, OLSR đi thẳng nguồn → đích 59% thời gian, và các giây đó chiếm 52% số gói mất.
- Khi route đã ổn định, đi thẳng có PDR 0,900, còn qua relay là 0,966.
- OLSR chọn route ít hop nhất nên giữ link trực tiếp dù link đó yếu. Vì vậy di chuyển relay không chạm tới phần mất gói này.

**Giả thuyết v4:** ghim route dữ liệu sang đường qua relay khi ETX của đường đó (De Couto và cộng sự, 2003) tốt hơn rõ rệt so với lựa chọn của OLSR sẽ làm tăng PDR, mà không cần di chuyển UAV.

## 2. Những gì cố định

- Kịch bản, simulator ns-3.45 và cấu hình ứng dụng giống v2/v3.
- Mọi arm của v4 chạy cùng một stack routing, `--routeControl=external` với `OverrideRouting`. Khi không có lệnh ghim, stack này để OLSR quyết định như cũ.
  - Kiểm tra trước khi đăng ký: trên seed formal v2 (1200055, 1300055), HOLD với stack mới nhận đúng 12.596/15.000 gói, trùng từng gói với HOLD của formal v2.
- Bộ quyết định route (`routing.py`) có các hằng số cố định:
  - EWMA α = 0,5 trên rx_ratio đo bằng probe hai chiều;
  - không ghim route trong 3 bước đầu (khởi động);
  - thời gian giữ tối thiểu 5 giây giữa hai lần đổi route tự nguyện;
  - link trong cửa sổ đo mới nhất < 0,1 thì nhả hoặc thay route ngay;
  - ETX của một link = 1/(q_ab · q_ba), chặn trên ở 1000.
- Quy tắc của bộ quyết định:
  - **Ghim** đường qua relay P* có ETX nhỏ nhất khi ETX(P*) + m < ETX(route của OLSR); nếu OLSR không có route thì ETX của OLSR là ∞.
  - **Nhả** route sau thời gian giữ tối thiểu khi ETX(OLSR) + m < ETX(route đang ghim).
  - **Đổi** sang P* khi ETX(P*) + m < ETX(route đang ghim).
  - Chỉ margin m được chọn ở pha tuning.
- Arm có phục hồi dùng đúng controller `service-v2-no-forecast` của v2, với predictor formal v2 (`719816f8…`) không fit lại.
- Không dùng dữ liệu của v2 hay v3 để chọn tham số v4. Phần chẩn đoán chỉ dùng để đặt giả thuyết.

## 3. Pha, arm và seed

Seed mới: layout 2600001–2600096, radio 2700001–2700096. Đã kiểm toán, không trùng với 14 seed plan cũ (gồm v2 và v3).

| Pha | Static/gradual mỗi cấu hình | Kịch bản | Arm | Episode |
|---|---|---|---|---|
| tuning | 8/8, cặp dùng chung seed | 32 | hold, lq-route-m0, lq-route-m05, lq-route-m10 | 128 |
| validation | 4/12 | 32 | hold, service-v2-no-forecast, lq-route, lq-route-restore | 128 |
| test | 4/20 | 48 | như validation | 192 |

Ý nghĩa các arm:
- `lq-route`: ghim route, relay không di chuyển (HOLD).
- `lq-route-restore`: ghim route và có di chuyển phục hồi như service-v2-no-forecast.

Mỗi ô được retry một lần nếu lỗi kỹ thuật, như v2/v3.

## 4. Tuning (chỉ dùng tuning)

1. **Kiểm tra toàn vẹn:** ở mọi episode có ghim route, chuỗi lệnh đổi route Python ra lệnh phải trùng hoàn toàn với các dòng `UAVROUTE` simulator ghi, và với `UAVROUTE_CHANGES`. Sai lệch là lỗi kỹ thuật, dừng v4.
2. **Chọn margin** m ∈ {0; 0,5; 1,0}: chọn giá trị cho chênh PDR (arm − hold) lớn nhất, tính trên kịch bản gradual và trung bình cân bằng n4/n5. Hòa thì chọn m lớn hơn (ít đổi route hơn).

## 5. Validation: go/no-go (không tune)

Chỉ chạy test khi cả ba điều kiện đều đúng:
1. PDR(lq-route) − PDR(hold) > 0 (gradual, trung bình cân bằng n4/n5).
2. lq-route có ghim route ở ít nhất 50% số episode gradual, tức là cơ chế thực sự được dùng.
3. Toàn vẹn route đạt ở mọi episode.

## 6. Test: phân tích đã khóa

Bootstrap 10.000 lần trên khối kịch bản gradual, giữ đủ 4 arm trong mỗi khối, RNG `20261120`. Ba bước kiểm định theo thứ tự cố định; mỗi bước thành công khi cận dưới CI 95% > 0, và chỉ diễn giải bước sau khi bước trước thành công:
1. PDR lq-route − hold (**endpoint chính**);
2. PDR lq-route-restore − service-v2-no-forecast;
3. PDR lq-route-restore − hold.

Chỉ mô tả, không kiểm định:
- quãng đường di chuyển;
- số lần đổi route mỗi episode;
- tỷ lệ episode có ghim route;
- PDR trong các giây có ghim so với các giây OLSR tự chọn;
- kết quả ở kịch bản static;
- các bất biến an toàn như v2.

## 7. Điều cấm

- Không chạy test khi validation chưa đạt go.
- Không đổi margin, hằng số của bộ quyết định, seed hay code sau khi đã nhìn kết quả validation/test.
- Không gộp tuning hoặc validation vào suy luận của test.
- Phải báo cáo cả v2, v3 và v4.

## 8. Cách chạy

Binary được build bằng ns-3.45 trong WSL, rồi chép vào `build/recovery/` của repo trên server cùng file `source.sha256`. Bridge sẽ kiểm tra hash nguồn và binary.

Lệnh: `python -m src.control.service_v4.cli freeze|run-case|validate|select|analyze`. Chạy hàng loạt bằng `orchestration/service_v4_queue.py`.

## 9. Sửa đổi

**2026-11-22 – sửa lỗi code bước chọn margin (trước khi có kết quả chọn).** Pha tuning đã thu đủ 128/128 cell hợp lệ
(freeze `packages/tuning`, HEAD `44cc1cc`). Lệnh `select` dừng với `TypeError: dict() got multiple values for keyword argument 'arm'`,
do khóa `arm` bị trùng khi gộp bảng; chưa có bảng margin nào được tạo ra. Commit sửa lỗi tách phần tính bảng thành
`margin_table`/`choose_margin`, thêm test, và không đổi quy tắc chọn ở mục 4. Không chạy lại mô phỏng; dữ liệu tuning giữ nguyên.
Bước `select` được chạy lại từ commit đã sửa. Package validation sẽ freeze từ commit này.
