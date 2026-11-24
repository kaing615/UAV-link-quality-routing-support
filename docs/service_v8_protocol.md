# v8: dự báo đồ thị nhân quả trong bộ điều khiển tuyến

Thiết kế ngày 23/11/2026, trước khi huấn luyện v8 hoặc chạy bất kỳ cell v8 nào.
Đây là nghiên cứu bổ sung sau v7. Không thay đổi kết quả, binary hoặc package v2–v7.

## Câu hỏi và giới hạn

Thông tin chất lượng liên kết dự báo có cải thiện PDR so với chất lượng đo được,
khi giữ nguyên luật ghim tuyến? Đóng góp của tổng hợp thông tin hàng xóm được
đối chiếu bằng một mô hình bỏ truyền thông điệp đồ thị.

Mô hình mới là một biến thể Edge-SAGE hồi quy, không phải checkpoint phân loại
QoS/survival ở giai đoạn 1. Không gán kết quả của mô hình mới cho checkpoint cũ.
UAV relay đứng yên ở v8. Điều khiển vị trí dự báo sẽ là nghiên cứu riêng sau đó.

## Dữ liệu phát triển và học mô hình

Chỉ đọc trace HOLD đã có của v7: 48 episode test v7 để fit và 20 episode
validation v7 để chọn epoch dừng. Các tập này đã được quan sát trong nghiên cứu
cũ, vì vậy hoàn toàn là **dữ liệu phát triển** cho v8, không còn là test v8.
Giữ hash từng trace và hai file validation nguồn; không lấy trace của nhiều arm
trên cùng seed làm các mẫu độc lập. Không sử dụng dữ liệu v8 để fit/chọn epoch.

Mỗi giây dùng quan sát hiện tại và quá khứ: vị trí, vận tốc, ma trận tỷ lệ nhận
probe, RSSI và mặt nạ RSSI. Không dùng trạng thái nhiệm vụ tương lai, PDR tương lai,
nhãn tương lai hoặc thời gian mất kết nối tương lai trong đặc trưng.

- Nút (8 chiều): tọa độ trừ tâm đội hình / 240 m; vận tốc / 10 m/s;
  bậc hàng xóm chia N−1; trung bình tỷ lệ nhận probe.
- Cạnh (7 chiều): khoảng cách / 240; (RSSI+90)/40, RSSI thiếu được mã hóa −100 dBm;
  tỷ lệ nhận hiện tại; EWMA hệ số 0,5; chênh so với giây trước;
  vận tốc tách xa / 10; độ lớn vận tốc tương đối / 10.
- Đồ thị truyền thông điệp có cạnh khi tỷ lệ nhận hiện tại ≥ 0,1, bỏ tự nối.
- Đích hồi quy: trung bình tỷ lệ nhận probe của **5 giây tiếp theo**. Đây là
  dự báo chất lượng trung bình, không phải xác suất liên kết sống sót.
- Kiến trúc: hai lớp tổng hợp trung bình kiểu Edge-SAGE, 64 chiều, ReLU;
  bộ giải mã cặp nút + cạnh, 64 chiều, ReLU, đầu ra sigmoid; đối xứng hóa logits.
- MLP dùng cùng đặc trưng và kích thước lớp nhưng bỏ hai nhánh truyền thông điệp.
  Số tham số có thể khác; đây là ablation thông tin đồ thị, không phải so sánh
  các mô hình có số tham số bằng nhau.
- MSE trung bình theo cạnh không chéo và snapshot, số snapshot cân bằng n5/n7.
  Adam lr=0,001, weight decay=1e−5, batch 128, clip norm=5, tối đa 60 epoch;
  chọn MSE thấp nhất trên tập chọn epoch, patience 10, seed 20261123.
- Xuất trọng số sang NumPy. Kiểm tra 40 snapshot (20 mỗi cấu hình) phải có
  sai số tuyệt đối lớn nhất < 1e−5 so với PyTorch trước khi chạy simulator.

## Các arm và môi trường

1. `lq-route`: luật v7, dùng EWMA của tỷ lệ nhận hiện tại.
2. `gnn-route`: cùng luật, dùng dự báo Edge-SAGE cho chi phí tuyến.
3. `mlp-route`: cùng luật, dùng dự báo của ablation không đồ thị.
4. `olsr-etx-fast`: baseline v7, HELLO 0,5 s và TC 1,25 s đã chọn ở v7.

Margin 1,0, warmup 3 bước, dwell 5 s, kiểm tra liên kết hỏng vẫn dựa trên số đo
hiện tại < 0,1 ở mọi arm ghim tuyến. Không trộn dự báo với tỷ lệ hiện tại bằng
một tham số được chọn sau khi xem kết quả. Không tune lại margin hoặc HELLO.
Binary v7, runtime v5, ARP DeadTimeout 1 s, hai luồng nền 128 kbps, luồng chính
256 kbps; n5/n7, static/gradual, 240 s giữ nguyên như v7.

## Seed, pha và go/no-go

- Seed mới: layout 3400001+, radio 3500001+, kiểm toán với các seed plan trước.
- Validation: mỗi cấu hình 2 static + 8 gradual, 4 arm = **80 cell**.
- Test: mỗi cấu hình 4 static + 20 gradual, 4 arm = **192 cell**.
- Smoke chỉ dùng seed phát triển riêng 3399001/3499001 và 3399002/3499002;
  ghi vào kiểm toán seed, chỉ kiểm tra kỹ thuật. Không chọn lại mô hình theo PDR smoke.
- Go chỉ kiểm tra kỹ thuật: đủ cell, tuyến đúng lệnh; cờ simulator chính xác;
  không mất trắng khởi động theo định nghĩa v7; đủ bộ đếm và dữ liệu hai luồng nền;
  dự báo hữu hạn, trong [0,1], đúng hash mô hình và đúng số bước; không di chuyển relay.
- Không lấy dấu hoặc độ lớn PDR validation làm tiêu chí go. Nếu lỗi, dừng và giữ
  bằng chứng. Không chạy lại một cell chính thức; không thay seed vì kết quả xấu.
- Dùng lock toàn server, 4 worker; chỉ khởi chạy mỗi queue một lần.

## Phân tích test đã định trước

40 khối gradual (20 mỗi cấu hình), giữ cả 4 arm trong một khối; cân bằng n5/n7.
Bootstrap 10.000 lần, RNG 20261120, như v7. Thứ tự cố định, thành công khi cận
dưới CI 95% > 0 và chỉ diễn giải khẳng định khi các bước trước đã thành công:

1. H1 chính: PDR `gnn-route` − `lq-route`.
2. H2: PDR `gnn-route` − `mlp-route` (giá trị thêm của thông tin đồ thị).
3. H3: PDR `gnn-route` − `olsr-etx-fast`.

Nếu H1 không đạt, báo nguyên kết quả; không đổi mục tiêu chính sang so sánh dễ
thắng hơn. Tất cả so sánh vẫn được trình bày, phân biệt khẳng định và mô tả.
Mô tả thêm: PDR luồng nền, static, từng cấu hình, thời gian không có tuyến,
số lần đổi tuyến, phân bố chênh lệch theo seed, thời gian suy luận; MSE dự báo
trên cùng trace `lq-route` của test với cả hai mô hình và EWMA hiện tại.

Chi phí: byte OLSR/probe đo ở tầng IP. Chi phí báo cáo trung tâm và lệnh là ước
lượng, chưa cạnh tranh kênh. Vì GNN cần vị trí/vận tốc/RSSI bổ sung, không dùng
nguyên con số báo hiệu chỉ báo cáo tỷ lệ nhận của v7 để khẳng định tổng chi phí
triển khai GNN. Thời gian suy luận là số đo trên CPU máy chạy, không phải độ trễ
điều khiển thực tế trên UAV.

## Provenance

`selection.json` chứa trọng số GNN/MLP, lịch sử fit, manifest dữ liệu phát triển,
hash mã huấn luyện và kết quả kiểm tra xuất mô hình. Trước validation đóng gói
mã nguồn và selection; test kế thừa đúng package validation. Predictor logistic
v2 được collector sao chép chỉ để tương thích; nó không điều khiển các arm v8.
Mọi điều chỉnh kỹ thuật trước freeze phải được ghi rõ, không đổi thiết kế sau test.
