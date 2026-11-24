# Nhật ký thực thi v10

Ngày 24/11/2026. Protocol: `docs/service_v10_protocol.md`, commit `1a026f5`, trước mọi cell v10.

- Workspace: `/home/sac/uav-service/2026-11-24-service-v10`, bản sao repo v8x cộng mã v10 từ commit
  `1a026f5` (`git archive`). Binary v7, runtime v5, predictor v2 khớp hash (preflight).
- Seed: tuning layout 4 000 001–4 000 016 / radio 4 100 001+; validation/test layout 4 200 001–4 200 068 /
  radio 4 300 001+. Kiểm toán không trùng mọi kế hoạch trước, kể cả v8x và tuning v10.
- Hai smoke trước tuning (gnn-b30, ewma-a05) và hai smoke trước validation (gnn-tuned, ewma-tuned) đạt,
  chạy trong thư mục nháp, không dùng trong phân tích.

## Tuning

- Queue 160/160 wrapper thành công, 48,1 phút; validate 160/160 hợp lệ.
- PDR gradual cân bằng: EWMA 0,5 0,9534; 0,3 0,9589; 0,2 0,9592; **0,1 0,9624**; 0,05 0,9605.
  GNN β 1,0 0,9616; 0,5 0,9634; 0,3 0,9633; 0,2 0,9622; **0,1 0,9638**.
- Theo quy tắc khóa trước: α\* = 0,1, β\* = 0,1. Selection SHA-256 `fd69f63153c4815b…`.

## Validation

- Queue 60/60, 17,5 phút; validate 60/60 hợp lệ. Go = true: đủ cell, toàn vẹn tuyến, trace dự báo,
  relay đứng yên, không mất trắng.

## Test

- Queue 144/144, 41,9 phút; validate 144/144 hợp lệ. Mọi kiểm tra kỹ thuật đạt.
- **H1** `gnn-tuned` − `ewma-tuned`: +0,084 điểm % [−0,106; 0,286] → **không phát hiện khác biệt**.
  n5 +0,18, n7 −0,01. GNN cao hơn ở 22/40 khối gradual, hòa 2.
- **H2** `ewma-tuned` − `lq-route`: +0,90 [0,51; 1,35]; chỉ mô tả vì H1 không có kết luận theo một hướng.
- Mô tả: `gnn-tuned` − `lq-route` +0,98 [0,60; 1,42]. PDR gradual 95,08 / 95,98 / 96,07 %;
  tổng đổi tuyến 385 / 83 / 61. MSE 5 s trên trace `lq-route`: EWMA 0,5 0,00129; EWMA 0,1 0,00166;
  GNN β 0,1 0,00123. Static: ba arm bằng nhau.
- Analysis SHA-256 `3db3548fa7fa9cf5…`. Không chạy lại hoặc chỉnh gì sau khi xem test.

Kết quả chính thức đọc từ `reports/uav_service_v10/results/analysis/test/analysis.json`.
v11 (`docs/service_v11_protocol.md`) dùng α\* = 0,1 làm baseline `ewma-tuned`.
