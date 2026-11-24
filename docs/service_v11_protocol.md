# v11: độ bền theo seed huấn luyện và ablation cùng số tham số

Thiết kế ngày 24/11/2026, **trước** khi chạy bất kỳ cell v11 nào. Không thay đổi
kết quả, binary, mô hình hoặc package v2–v10/v8x.

## Lý do

Bản thảo bài báo (`reports/nckh_paper`) còn hai lỗ hổng mà phản biện chắc chắn
chọc, và cả hai đều đã được chính bài tự ghi nhận là hạn chế:

1. **Một seed huấn luyện.** PDR vòng kín của v8 chỉ được đo với trọng số của
   seed 20261123. Thăm dò `multiseed_service_v8.py` đã fit lại bằng 5 seed
   (20261123, 1, 2, 3, 4) và cho MSE 0,000778 ± 0,000035, GNN tốt hơn MLP ở
   5/5 seed — nhưng đó là **MSE ngoại tuyến**, và kết luận trung tâm của bài là
   MSE không dự đoán được dịch vụ. Vì vậy chính lập luận của bài buộc phải đo
   PDR vòng kín của các seed còn lại.
2. **Ablation bị lẫn với dung lượng mô hình.** GNN có 19.137 tham số, MLP
   13.505. Chênh lệch H2 của v8 (+0,38 điểm %) vì vậy không tách được tác động
   của truyền thông điệp khỏi tác động của số tham số.

## Thứ tự thực hiện: v11 chạy SAU v10

v10 chọn hệ số làm mượt tốt nhất α\* cho họ EWMA trên một tập tuning riêng. Nếu
v11 chạy trước và vẫn lấy EWMA 0,5 làm mốc, nó lặp lại đúng lỗi baseline yếu mà
bài báo đang phê phán. Vì vậy:

- **Bắt buộc:** v10 phải hoàn tất và công bố α\* trước khi v11 sinh seed plan.
- Baseline của v11 là `ewma-tuned` (α\* của v10), không phải EWMA 0,5.
- Nếu v10 kết luận cận trên của H1 < 0 (EWMA đã chỉnh thắng GNN đã chỉnh), v11
  vẫn chạy: khi đó câu hỏi của v11 trở thành kết quả âm đó có bền theo seed
  huấn luyện hay không, và không được đổi endpoint vì lý do này.

## Arm

Mọi arm: luật ghim tuyến v4/v7 (margin 1,0, warmup 3, dwell 5 s, liên kết hỏng
khi số đo hiện tại < 0,1), relay đứng yên, binary v7, runtime v5, ARP
DeadTimeout 1 s, hai luồng nền 128 kbps, n5/n7, 240 s. Không fit lại trên dữ
liệu mới: mọi mô hình học trên đúng tập phát triển v8 (48 trace HOLD test v7 để
fit, 20 trace HOLD validation v7 để chọn epoch) với đúng recipe đã khóa (MSE,
Adam lr 0,001, weight decay 1e−5, batch 128, clip norm 5, tối đa 60 epoch,
patience 10).

| Arm | Nguồn chất lượng |
|---|---|
| `ewma-tuned` | EWMA hệ số α\* của v10 |
| `gnn-s0` | GNN seed 20261123 (trọng số đã khóa của v8) |
| `gnn-s1` | GNN seed 1 |
| `gnn-s2` | GNN seed 2 |
| `gnn-s3` | GNN seed 3 |
| `gnn-s4` | GNN seed 4 |
| `mlp-matched` | MLP không truyền thông điệp, số tham số khớp GNN, seed 20261123 |

`gnn-s1..s4` dùng đúng trọng số do `multiseed_service_v8.py` sinh ra, không fit
lại; hash SHA-256 của từng bộ trọng số được ghi vào package trước khi chạy.

### Quy tắc khớp số tham số cho `mlp-matched`

Giữ nguyên kiến trúc MLP của v8 (bỏ hai nhánh truyền thông điệp học được, giữ
đặc trưng bậc nút), chỉ quét số chiều ẩn $h \in \{64, 72, 80, 88, 96\}$ và
chọn $h^\*$ có $|\text{params}(h) - 19{,}137|$ nhỏ nhất. Hòa: chọn $h$ nhỏ hơn.
Quy tắc này được áp **trước** khi huấn luyện và không được sửa sau đó; số tham
số thu được được ghi vào package. Không chọn $h^\*$ theo MSE hoặc theo PDR.

## Pha và seed

Một package, seed được kiểm toán không trùng mọi kế hoạch trước (gồm v8x và
v10): layout 5 000 001+, radio 5 100 001+.

1. **Validation** (study `v11v`): 2 static + 8 gradual mỗi cấu hình × 7 arm
   (140 episode). Go chỉ xét kỹ thuật: đủ cell, toàn vẹn tuyến, trace dự báo
   hợp lệ, relay đứng yên, không mất trắng lúc khởi động, hash trọng số của
   từng arm khớp manifest. **Không xét PDR.**
2. **Test** (study `v11`): 4 static + 20 gradual mỗi cấu hình × 7 arm
   (336 episode), chỉ khi go.

## Phân tích test (khóa trước)

Bootstrap 10.000 lần trên 40 khối gradual ghép cặp, cân bằng n5/n7, seed
20261120. Thứ tự cố định, cổng tuần tự: chỉ diễn giải khẳng định một bước khi
mọi bước trước đã đi tới kết luận có hướng.

- **H1 (chính, độ bền theo seed).** Với mỗi $k \in \{0,1,2,3,4\}$ tính
  $\Delta_k = \mathrm{PDR}(\texttt{gnn-s}k) - \mathrm{PDR}(\texttt{ewma-tuned})$.
  Đại lượng chính là $\min_k \Delta_k$ cùng CI 95% hai phía.
  - Cận dưới > 0: lợi ích của GNN bền với mọi seed đã thử.
  - Cận trên < 0: tồn tại seed mà GNN kém hơn EWMA đã chỉnh.
  - CI chứa 0: không phát hiện; **không** diễn giải là tương đương.
  Báo cáo đầy đủ cả năm $\Delta_k$ và CI riêng của từng seed, kể cả khi trái
  chiều nhau. Độ phân tán giữa các seed, $\max_k \Delta_k - \min_k \Delta_k$,
  được báo cáo như đại lượng mô tả.
- **H2 (ablation sạch).**
  $\mathrm{PDR}(\texttt{gnn-s0}) - \mathrm{PDR}(\texttt{mlp-matched})$,
  CI 95% hai phía. Cận dưới > 0: truyền thông điệp có giá trị thêm ở cùng dung
  lượng mô hình. Nếu CI chứa 0 thì kết luận là chênh lệch H2 của v8 không tách
  được khỏi số tham số, và câu tương ứng trong bài báo phải được sửa.
- **Mô tả:** số lệnh đổi tuyến, mất kết nối, PDR luồng nền, static, MSE của mỗi
  arm trên trace `ewma-tuned` của test, và tương quan giữa thứ hạng MSE và thứ
  hạng PDR trên 7 arm. Đại lượng mô tả cuối là kiểm chứng trực tiếp của luận đề
  bài báo và phải được báo cáo bất kể dấu.

Mỗi cell chạy lại tối đa một lần khi lỗi kỹ thuật; không thay seed, không chạy
lại vì kết quả xấu. Không thêm arm, không thêm seed huấn luyện, không đổi α\*
sau khi thấy bất kỳ kết quả nào của v11. Mọi kết quả, kể cả bất lợi, được báo cáo.

## Cái protocol này KHÔNG làm

- Không huấn luyện lại GNN trên dữ liệu relay di chuyển (đó là nhánh v9, cần
  một nghiên cứu riêng).
- Không mô phỏng telemetry và lệnh điều khiển thành gói cạnh tranh kênh. Giới
  hạn trên của so sánh với OLSR-ETX vì vậy vẫn giữ nguyên cách đọc như ở v8.
- Không học tăng cường. Phân tích giới hạn hình học v9 cho thấy họ kịch bản này
  không còn dư địa; quyết định đó không đổi.
