# v10: dự báo GNN so với EWMA đã chỉnh, cả hai được làm mượt tối ưu

Thiết kế ngày 24/11/2026, sau khi xem v8 và thăm dò v8x, **trước** khi chạy bất kỳ
cell v10 nào. Không thay đổi kết quả, binary, mô hình hoặc package v2–v9/v8x.

## Lý do

Test v8: `gnn-route` − `lq-route` (EWMA 0,5) = +0,66 điểm % [0,38; 1,00].
Thăm dò v8x (không đăng ký trước, 40 khối mới): EWMA 0,2 đạt bằng GNN (+0,78 mỗi
bên so với EWMA 0,5), EWMA 0,1 hơn GNN 0,31 [0,04; 0,61]. Baseline EWMA 0,5 của v8
chưa được chỉnh. Hệ số 0,1/0,2 được chọn sau khi xem v8 nên v8x không đủ để kết
luận chính thức.

## Câu hỏi

Khi **cả hai** nguồn chất lượng được làm mượt với hệ số tốt nhất chọn trên tập
tuning riêng, dự báo GNN có tăng PDR so với chất lượng đo hiện tại không?
Cho GNN cùng một núm làm mượt để so sánh không nghiêng về bên nào.

## Arm

Mọi arm: luật ghim tuyến v4/v7 (margin 1,0, warmup 3, dwell 5 s, liên kết hỏng khi
số đo hiện tại < 0,1), relay đứng yên, binary v7, runtime v5, ARP DeadTimeout 1 s,
hai luồng nền 128 kbps, n5/n7, 240 s. Trọng số GNN là trọng số đã khóa của v8
(selection SHA-256 `ad2e0741…9f`), không fit lại.

- Họ EWMA: `q_t = α·d_t + (1−α)·q_{t−1}` trên tỷ lệ nhận probe hiện tại.
- Họ GNN: `q_t = β·ĝ_t + (1−β)·q_{t−1}` trên dự báo GNN 5 giây; β = 1 là `gnn-route` v8.

## Pha, seed và quy tắc

Hai package riêng, seed được kiểm toán không trùng mọi kế hoạch trước (gồm v8x):

1. **Tuning** (study `v10t`, layout 4 000 001+, radio 4 100 001+): 8 khối gradual mỗi
   cấu hình × 10 arm: `ewma-a50`, `ewma-a30`, `ewma-a20`, `ewma-a10`, `ewma-a05`,
   `gnn-b100`, `gnn-b50`, `gnn-b30`, `gnn-b20`, `gnn-b10` (160 episode).
   - Chọn α\* = α có PDR gradual trung bình cân bằng n5/n7 cao nhất trong họ EWMA;
     β\* tương tự trong họ GNN. Hòa (chênh < 1e−9): chọn hệ số lớn hơn (ít làm mượt).
   - Không xét kết quả nào khác khi chọn. Không thêm hệ số sau khi xem tuning.
2. **Validation** (study `v10`, layout 4 200 001+, radio 4 300 001+): 2 static + 8 gradual
   mỗi cấu hình × 3 arm (60 episode). Go chỉ xét kỹ thuật: đủ cell, toàn vẹn tuyến,
   trace dự báo hợp lệ, relay đứng yên, không mất trắng lúc khởi động. Không xét PDR.
3. **Test**: 4 static + 20 gradual mỗi cấu hình × 3 arm (144 episode), chỉ khi go.

Arm validation/test: `lq-route` (EWMA 0,5 như v8), `ewma-tuned` (α\*), `gnn-tuned` (β\*).

## Phân tích test (khóa trước)

Bootstrap 10 000 lần trên 40 khối gradual ghép cặp, cân bằng n5/n7, seed 20261120.

- **H1 (chính)**: PDR(`gnn-tuned`) − PDR(`ewma-tuned`), CI 95% hai phía.
  - Cận dưới > 0: dự báo GNN có giá trị thêm sau khi chỉnh làm mượt.
  - Cận trên < 0: EWMA đã chỉnh tốt hơn GNN đã chỉnh.
  - CI chứa 0: không phát hiện khác biệt; **không** diễn giải là tương đương.
- **H2**: PDR(`ewma-tuned`) − PDR(`lq-route`), thành công khi cận dưới > 0. Chỉ được
  khẳng định nếu H1 đi đến một trong hai kết luận có hướng; nếu H1 chứa 0 thì H2 chỉ mô tả.
- Mô tả: số lệnh đổi tuyến, mất kết nối, PDR luồng nền, static, MSE dự báo của
  α\* và β\* trên trace `lq-route` test.

Mỗi cell chạy lại tối đa một lần khi lỗi kỹ thuật; không thay seed, không chạy lại vì
kết quả xấu. Mọi kết quả, kể cả bất lợi, được báo cáo.
