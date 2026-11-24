# v9: di chuyển relay dựa trên dự báo để giảm chia cắt mạng

Thiết kế ngày 23/11/2026, trước mọi mô phỏng v9 và trước khi xem kết quả test v8.
Yêu cầu của người dùng: bổ sung điều khiển vị trí UAV dựa trên dự báo. Đây là
điều khiển vận tốc relay trong ns-3, chưa tích hợp autopilot/UAV thật.

## Câu hỏi

Với cùng bộ định tuyến và cùng luật chọn vị trí, kích hoạt di chuyển theo dự báo
GNN có giảm thời gian nguồn–đích bị chia cắt so với kích hoạt theo số đo hiện tại?
Không mặc định dự báo hoặc chuyển động phải đem lại lợi ích. Kết quả âm được giữ.

## Mô hình và các arm

Giữ nguyên GNN/MLP hồi quy 5 giây của v8, selection SHA-256
`ad2e0741a0d7509e922bbd8c186cca0477ecd98ec67440c81b4c8cd12ebbe49f`.
Không huấn luyện lại hoặc chọn mô hình theo PDR v8/v9.

Tất cả arm đều dùng **lq-route theo chất lượng hiện tại** của v7, margin 1,0.
Như vậy chỉ khác cơ chế di chuyển, không đồng thời thay cách chọn tuyến:

1. `route-only`: relay đứng yên.
2. `reactive-motion`: kích hoạt khi chất lượng đường tốt nhất hiện tại yếu.
3. `gnn-motion`: kích hoạt theo dự báo GNN hoặc cùng điều kiện hiện tại của reactive.
4. `mlp-motion`: tương tự, thay GNN bằng mô hình không đồ thị (ablation).

Chất lượng đường là min tỷ lệ nhận của các cạnh; đường tốt nhất tối đa hóa giá
trị đó (widest path), tính trên EWMA hiện tại hoặc ma trận dự báo. Cạnh dự báo
chỉ dùng khi số đo hiện tại ≥0,1, không suy diễn liên kết chưa đo được.
Kích hoạt khi chất lượng tốt nhất <0,70 trong 2 frame liên tiếp; 3 frame đầu
chỉ quan sát. GNN/MLP có fallback kích hoạt hiện tại như reactive. Ngưỡng cố định,
không chọn lại theo kết quả.

## Luật di chuyển dùng chung

Xếp relay theo phép chiếu lên đường nối nguồn–đích; vị trí đích là các điểm chia
đều đoạn nối hai vị trí đầu mút **hiện tại**. Không dùng quỹ đạo nhiệm vụ tương
lai. Mỗi bước gửi vận tốc hướng đến vị trí đích, tối đa 8 m/s, deadband 1 m;
trừ quãng đường đã đi khỏi budget 400 m cho mỗi relay. Burst tối đa 10 bước,
sau đó nghỉ 5 bước; có thể dừng sớm nếu chất lượng hiện tại và (với arm dự báo)
chất lượng dự báo đều ≥0,80 trong 3 frame. Hết budget thì dừng.

Kiểm tra khoảng cách nhỏ nhất liên tục trong bước (swept separation) trước khi
gửi lệnh, bao gồm chuyển động đầu mút theo vận tốc quan sát. Nếu <5 m, gửi lệnh
0 cho toàn relay và ghi lý do. Simulator vẫn thực thi giới hạn giống nhau ở mọi
arm. Kiểm tra này không phải bảo đảm an toàn cho UAV thật.

## Kịch bản và seed

Giữ binary/runtime v7, ARP 1 s, hai luồng nền 128 kbps, luồng chính 256 kbps,
240 giây, n5/n7. Đầu mút tách xa từ bước 30 với **2 m/s mỗi đầu mút** ở gradual
(v7/v8 dùng 1 m/s); static giữ đầu mút đứng yên. Mục đích là tạo tình huống liên
kết từ đầu mút tới relay đứng yên có thể đứt, trong khi bố trí lại relay còn khả
thi trong không gian và ngân sách đã định.

Đây cũng là thay đổi phân phối vận tốc so với dữ liệu fit; phải báo cáo như
kiểm tra chuyển miền, không khẳng định GNN đã hiệu chuẩn cho mọi tốc độ.

- Validation: 2 static + 8 gradual mỗi cấu hình, 4 arm, **80 cell**.
- Test: 4 static + 20 gradual mỗi cấu hình, 4 arm, **192 cell**.
- Layout 3600001+, radio 3700001+, kiểm toán với v8 và các vòng trước.
- Smoke: 3599001/3699001 n5 gnn-motion; 3599002/3699002 n7 reactive-motion.
  Chỉ kiểm tra kỹ thuật, không chọn tham số theo kết quả dịch vụ.
- Mỗi cell chạy một lần; queue 4 worker, lock toàn server, dừng khi lỗi, không
  tự retry hoặc đổi seed.

## Validation go/no-go

Đủ cell, tuyến đúng lệnh, cờ đúng arm, không blackout khởi động kiểu v7, đủ
bộ đếm/luồng nền; dự báo hợp lệ/đúng hash và không dùng tương lai. Toàn bộ frame
không vượt budget hoặc khoảng cách tối thiểu; không có safety rejection của
simulator. `route-only` không di chuyển.

Kiểm tra kịch bản có thách thức: ít nhất 8/16 episode gradual `route-only` có
≥5 giây mất kết nối nguồn–đích. Đây là cổng khả năng kiểm tra câu hỏi, chỉ dựa
baseline; không xét lợi ích của arm GNN hoặc PDR để quyết định go. Nếu không đạt,
báo thiết kế chưa tạo đủ chia cắt, dừng test; không tăng tốc độ rồi chạy lại cùng
nghiên cứu. Ghi số can thiệp ở mọi arm, không đòi GNN phải thắng mới chạy test.

## Test và cách diễn giải

40 khối gradual, 20 mỗi cấu hình, bootstrap 10.000 lần theo khối giữ 4 arm,
cân bằng n5/n7, RNG 20261120. Thứ tự cố định; bước sau chỉ khẳng định khi bước
trước đạt. Thành công khi cận dưới CI 95% >0:

1. **H1 chính:** số giây mất kết nối của reactive-motion trừ gnn-motion.
2. **H2:** PDR gnn-motion trừ reactive-motion.
3. **H3:** số giây mất kết nối của mlp-motion trừ gnn-motion.

Mất kết nối: frame 1..240 có `source_dest_connected=false` trong đồ thị probe
với ngưỡng simulator 0,5. Phân biệt chỉ số này với mất tuyến OLSR và mất gói ứng
dụng; cả ba được báo cáo riêng. Không gọi cải thiện PDR là bằng chứng ngăn chia
cắt nếu H1 không đạt.

Mô tả: mọi arm so với route-only; PDR luồng nền, outage dài nhất, số giây không
có tuyến OLSR, quãng đường mỗi relay và toàn đội, số lần bắt đầu burst, số lần
chặn vì an toàn/budget, thời điểm chuyển động đầu tiên so với lần chia cắt đầu
tiên, static, từng cấu hình, thời gian tính quyết định. Không đồng nhất quãng
đường với năng lượng hoặc thời lượng pin. Báo cáo lợi ích và chi phí cùng nhau.

Sử dụng điều khiển tập trung lý tưởng về mất gói báo cáo/lệnh giống v7; không
thêm một kết luận triển khai thực tế từ mô phỏng này. Nghiên cứu không chứng
minh khả năng duy trì kết nối nếu hai đầu mút vượt khả năng phủ của toàn chuỗi.
