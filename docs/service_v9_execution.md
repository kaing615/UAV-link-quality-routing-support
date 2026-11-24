# Nhật ký triển khai v9

Ngày chuẩn bị: 23/11/2026. Mở rộng theo yêu cầu của người dùng: điều khiển vị
trí relay dựa trên dự báo để phòng chia cắt. Không tích hợp UAV thật hoặc RL.

- Protocol viết trước mô phỏng v9 và trước kết quả test v8.
- Workspace riêng: `/home/sac/uav-service/2026-11-23-service-v9`.
- Clone nội bộ repo v7, commit `deda60b0b667943f77cf424aeb162ed6d0ab62a4`,
  cộng source bundle v8 và v9. Không sửa repo/package đang chạy của v8.
- Sao chép binary v7 đã khóa, không build. Cả 9 mục manifest source/binary OK.
- SHA-256 selection v9:
  `a3219acae53168db831c4907ede73c9f1bd3879f747161d5f0be7aad8cb0ae94`.
- SHA-256 seed plan:
  `78639e204cab91af3af8e58718d07df3db647e72aee9f0ae96d936d4af31b134`.
- SHA-256 protocol:
  `d7342ef9a3938dc852c8997c26a52e6e1ef6d51edf20474413e44306f55c2e99`.
- Năm kiểm thử về đường rộng nhất, khoảng cách trong bước, ngân sách,
  khác biệt trigger và binding kịch bản vào cell đều đạt.
- Dung lượng còn trống khi chuẩn bị: 26 GB. Chưa có simulation v9 tại thời
  điểm ghi các bước chuẩn bị này.

Kết quả chính thức phải đọc từ manifest, queue-finished, validation và analysis
đã khóa. Không suy luận hoàn thành hoặc hiệu quả từ việc triển khai thành công.

## Bắt đầu validation

- Preflight: source 9/9 và runtime manifest OK; 17 GB trống.
- Smoke n5 gnn-motion và n7 reactive-motion đạt integrity, safety, tx, background, no-blackout.
- Freeze validation và dry-run đạt; queue 80 cell khởi chạy một lần, PID 4160735.

## Validation hoàn tất

- Queue validation 80/80 wrapper thành công, 1699,5 giây; validate 80/80 hợp lệ.
- Go=true; baseline challenge 16/16 đạt ít nhất 5 giây chia cắt (yêu cầu 8/16); mọi cổng kỹ thuật và an toàn đạt.
- Test freeze kế thừa nguồn/selection validation và dry-run đạt; queue 192 cell khởi chạy một lần.

## Test hoàn tất

- PID queue test 86341, chạy một lần; 192/192 wrapper thành công, 3874,4 giây; validate 192/192 hợp lệ.
- Endpoint chính reactive trừ GNN: 0,20 giây [-1,05; 1,45], không đạt. H2/H3 chỉ mô tả theo fixed sequence.
- H2 PDR GNN trừ reactive +0,00338 [-0,0031517; 0,0104269]; H3 MLP trừ GNN 1,225 giây [0,10; 2,525], không được khẳng định vì cổng H1 đóng.
- Toàn vẹn tuyến 192/192, mọi cổng kỹ thuật/an toàn test đạt; không chạy lại hoặc tune theo kết quả.
- Archive xuất SHA-256 ffde09706c922596b9bb67ee0bead9187328bac31123c346713a5cdd38c493d7, kiểm toán cục bộ đạt.
