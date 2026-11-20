# Thí nghiệm phòng ngừa chia cắt mạng UAV

## Mục tiêu

Kiểm tra xem dự báo liên kết có giúp giảm thời gian mạng bị chia cắt khi các UAV nhiệm vụ bay tách dần hay không. Giới hạn tốc độ và quãng đường của các UAV chuyển tiếp giống nhau giữa các phương pháp.

Đây là phần mở rộng của hệ thống dự báo liên kết và phục hồi kết nối. Kết quả phục hồi sau sự cố không được dùng thay cho kết quả phòng ngừa.

## Cơ chế được kiểm tra

Mỗi giây, bộ điều khiển nhận vị trí, vận tốc và số liệu thăm dò vô tuyến trong giây vừa qua. Predictor ước lượng nguy cơ một liên kết hiện còn hoạt động sẽ mất trong 3, 5, 10 hoặc 20 giây tiếp theo. Bộ điều khiển ưu tiên các cạnh cầu của đồ thị đo được, vì mất một cạnh cầu có thể tách mạng thành hai phần.

Thời điểm can thiệp xét cả thời gian relay tới vị trí mới và một tham số độ trễ lập kế hoạch định tuyến. Mô hình dự báo không biết lịch bay tương lai của UAV nhiệm vụ. Planner hiện đặt relay cách đều trên đoạn nối nguồn–đích quan sát được, rồi hệ thống đo và dự báo lại ở bước kế tiếp. Bộ dự báo chưa đánh giá hậu quả của từng hành động ứng viên.

## Giao thức pilot đã định trước

Ngày chốt: 11/11/2026, trước khi xem kết quả test.

| Thành phần | Quy định |
|---|---|
| Số UAV | 8, gồm 2 UAV nhiệm vụ và 6 relay |
| Mỗi episode | 120 bước, mỗi bước 1 giây |
| Giới hạn relay | 8 m/s; 400 m/relay; khoảng cách an toàn 5 m |
| Train | 12 layout seeds 11001–11012; radio seed = layout seed + 100000 |
| Calibration | 6 layout seeds 21001–21006; radio seed = layout seed + 100000 |
| Test | 20 layout seeds 31001–31020; radio seed = layout seed + 100000 |
| Thu trace train/calibration | hold và reactive trên mỗi scenario |
| So sánh test | hold, reactive, geometry, predictive-simple, predictive-learned và hai ablation |
| Ngưỡng kích hoạt | Chọn trước khi chạy test; mặc định 0.5, mọi thay đổi phải ghi lại cùng lý do và dùng tập calibration |
| Độ trễ định tuyến để lập kế hoạch | Mặc định 3 giây, là tham số lập kế hoạch; đo riêng trạng thái OLSR thực tế |
| Đơn vị thống kê | Episode/scenario; không coi các cửa sổ thời gian là các lần thử độc lập |

Nếu không thu được đủ hai lớp nhãn trên train hoặc calibration, phải ghi là lỗi thiết kế dữ liệu và thu thêm trên seed train/calibration mới. Không dùng test để giải quyết thiếu lớp. Nếu seed không khởi tạo được mạng kết nối, ghi lỗi thiết lập; không thay seed âm thầm.

Pilot dùng forecaster logistic thời gian với hợp đồng feature mới. Các checkpoint GNN hiện có cần bộ chuyển đổi feature và kiểm tra lại trước khi đưa vào vòng điều khiển. Không gọi logistic là GNN và không gọi bộ điều khiển theo quy tắc là PPO.

## Chỉ số

Chỉ số chính là tỷ lệ thời gian toàn mạng bị chia cắt. Báo thêm tỷ lệ thời gian source–destination mất kết nối, thời gian OLSR không có đường dùng được, số lần chia cắt, PDR, độ trễ và quãng đường relay di chuyển. Quãng đường là chi phí chuyển động, chưa phải mô hình năng lượng pin.

Dự báo được đánh giá riêng bằng Brier score, log loss và các nhóm độ tin cậy trên test. Một can thiệp thành công có thể ngăn sự cố xảy ra; không được coi việc sau đó không đứt mạng là bằng chứng nó đã báo động sai.

Mỗi policy chạy cùng grid seed và đủ horizon, kể cả sau khi hết ngân sách. Lưu mọi lỗi kỹ thuật riêng. Chỉ phân tích paired khi đủ grid và không còn lỗi kỹ thuật. Khoảng tin cậy của chênh lệch giữa các policy dùng bootstrap theo scenario, giữ các policy trên cùng scenario cùng nhau.

## Tiêu chí ra quyết định nghiên cứu

Hướng này đáng phát triển tiếp nếu lợi ích giảm chia cắt xuất hiện trên test chưa thấy và vẫn còn khi so sánh chi phí di chuyển. Nếu geometry hoặc predictor đơn giản tốt hơn learned, báo cáo phải nêu rõ; ưu tiên sửa giả thuyết hoặc mô hình trước khi tăng ngân sách PPO.

Các kết quả pilot là bằng chứng ban đầu. Bài gửi tạp chí cần thêm quy mô mạng, kiểu chuyển động, điều kiện vô tuyến và kiểm thử ngoài phân bố phù hợp với phạm vi tuyên bố.

## Nghiên cứu liên quan

- Ladosz và cộng sự (2019), *A Hybrid Approach of Learning and Model-Based Channel Prediction for Communication Relay UAVs in Dynamic Urban Environments*. Dự báo kênh từ tín hiệu quan sát và lập quỹ đạo UAV chuyển tiếp. [DOI](https://doi.org/10.1109/LRA.2019.2903850).
- Tao và Liu (2024), *Online Autonomous Motion Control of Communication-Relay UAV with Channel Prediction in Dynamic Urban Environments*. Học kênh bằng Gaussian process và tìm vị trí relay trực tuyến. [Bài báo](https://www.mdpi.com/2504-446x/8/12/771).
- Dağasan và Karasan (2025), *Resilient Multi-Hop Autonomous UAV Networks With Extended Lifetime for Multi-Target Surveillance*. Lập quỹ đạo relay để kéo dài thời gian mạng UAV còn kết nối. [DOI](https://doi.org/10.1109/TNSM.2025.3528495).
- *Hop-bounded informed-probabilistic roadmap for robust unmanned aerial vehicle relay placement in free-space optical networks* (2026). Dùng dự báo tín hiệu suy giảm để kích hoạt lập lại phương án và điều động relay trong mạng FSO. [DOI](https://doi.org/10.1016/j.eprime.2026.201169).

Các công trình trên là tiền lệ cần đối chiếu. Chúng chưa đủ để khẳng định phần mở rộng này là mới. Đóng góp dự kiến phải được mô tả bằng cơ chế và bằng thực nghiệm, không dựa vào việc ghép tên các thuật toán.

## Trạng thái triển khai ngày 11/11/2026

Đã có kịch bản `relay-prevention`, môi trường riêng, logistic nhân quả, bộ điều khiển và công cụ thu/huấn luyện/đánh giá. Mọi lượt chạy dùng thư mục mới và kiểm tra tập seed, hash trace, nguồn, mô hình và đủ horizon. Dữ liệu mang nhãn test không được nhận làm train hoặc calibration. `load_traces(root, expected_split=...)` kiểm tra cả manifest bao quanh khi đọc shard hoặc thư mục một policy.

Đã thu 24 episode train, 12 episode calibration và hoàn tất pilot 7 phương pháp ×20 scenario. [Báo cáo và bằng chứng](../reports/uav_prevention_pilot_2026-11-11/REPORT.md) ghi cấu hình, mã băm, kết quả từng phương pháp và giới hạn thực tế. Bộ dự báo là logistic; các checkpoint GNN và PPO trước đây chưa được đưa vào vòng phòng ngừa này.

Pilot đầu cho thấy relay di chuyển giữ đồ thị thăm dò kết nối tốt hơn đứng yên, nhưng learned chưa hơn geometry/simple. PDR thực tế vẫn thấp. Không dùng tỷ lệ đồ thị kết nối cao để thay cho bằng chứng mạng truyền dữ liệu tốt. Trạng thái OLSR và gói thật được báo riêng; [chẩn đoán định tuyến](../reports/uav_prevention_pilot_2026-11-11/ROUTING_DIAGNOSTIC.md) chưa xác định đầy đủ nguyên nhân ở từng nút.

Vòng tối ưu đã hoàn tất theo giao thức chốt trước kết quả pilot: simple và learned nhận cùng ngân sách 9 cấu hình, chọn riêng trên calibration rồi chạy seed xác nhận mới cùng hai điều kiện tốc độ cao hơn và 12 UAV. [Giao thức tối ưu](../reports/uav_prevention_pilot_2026-11-11/optimization_protocol.json) giữ nguyên kết quả ban đầu và mô hình đã fit.

Tổng cộng đã chạy 704 episode phòng ngừa, gồm 560 test trên 80 scenario thuộc bốn điều kiện; không có lỗi kỹ thuật. Bộ phân tích đối chiếu lại dữ liệu thô, phiên bản nguồn, luồng gói và kế hoạch thực chạy. Learned có Brier/log loss thấp hơn simple trên hold, nhưng chưa có lợi thế điều khiển ổn định so với geometry/simple. Ở 12 UAV, learned có trung bình chia cắt 0,375%, geometry/simple là 0%; PDR của geometry và learned đều bằng 0. Quy tắc deadline cũng chưa cho lợi ích ổn định khi kiểm tra bằng ablation.

[Báo cáo hoàn chỉnh](../reports/uav_prevention_pilot_2026-11-11/REPORT.md) đã có bảng so sánh paired, khoảng tin cậy, reliability, độ trễ có điều kiện và kết luận mức sẵn sàng. Kết quả đủ làm báo cáo tiến độ; cần xử lý định tuyến và kiểm chứng lại mục tiêu điều khiển trước một bài tuyên bố duy trì truyền thông tốt hơn. [Hướng dẫn tái lập](../reports/uav_prevention_pilot_2026-11-11/REPRODUCE.md) đi kèm bằng chứng và các lệnh đã dùng.
