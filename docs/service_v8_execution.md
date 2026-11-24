# Nhật ký triển khai v8

Ngày bắt đầu: 23/11/2026. Mục tiêu do người dùng yêu cầu: hoàn thiện NCKH bằng
thực nghiệm GNN chọn tuyến, tiếp đó thêm điều khiển vị trí UAV dựa trên dự báo.
Các kết quả v2–v7 được giữ nguyên.

- Protocol: `docs/service_v8_protocol.md`, viết trước khi fit hoặc chạy cell v8.
- V8 chỉ kiểm tra dự báo cho định tuyến; điều khiển vị trí là nghiên cứu tiếp theo.
- Server mới: `/home/sac/uav-service/2026-11-23-service-v8`.
- Mã nền: clone nội bộ từ repo v7, commit `deda60b`;
  thêm các module v8, không pull hoặc sửa package của các vòng trước.
- Dữ liệu phát triển: 48 HOLD của test v7 để fit, 20 HOLD của validation v7
  để chọn epoch; tất cả được chuyển thành dữ liệu phát triển cho v8.
- SHA-256 dữ liệu gzip:
  `3a9cceac782ae1dca446ae3e3e27321e09c15b5db58ea32636c5c8409a3168cd`.
- Huấn luyện một lần trên CPU, seed 20261123; không chọn lại kiến trúc sau khi
  quan sát dữ liệu v8. GNN chọn epoch 58 (MSE 0,0007965441), MLP chọn epoch 59
  (MSE 0,0011458147); EWMA trên cùng tập chọn epoch: 0,0013503141.
  Đây là số liệu phát triển, không phải kết quả kiểm định v8.
- Sai số tối đa giữa NumPy xuất mô hình và PyTorch:
  `1.1920928955078125e-07` cho cả hai mô hình, đạt ngưỡng `1e-5`.
- SHA-256 selection:
  `ad2e0741a0d7509e922bbd8c186cca0477ecd98ec67440c81b4c8cd12ebbe49f`.
- Seed plan kiểm toán 20 tệp lịch sử/phát triển:
  `70fe14358bcd87e99c2a7a77d55a1accc079331e2fed3616eb7177ab1bffbfcf`.
- 5 kiểm tra mới về tính nhân quả, bất biến tịnh tiến, xuất mô hình, số cell
  và seed đạt; 24 kiểm tra v4/v7 liên quan đạt.
- Trong chuẩn bị workspace, kiểm tra manifest báo thiếu bản sao
  `build/recovery/uav-olsr-dataset`. Chưa chạy cell tại thời điểm đó. Đã sao chép
  binary v7 đã khóa vào đúng vị trí; không biên dịch. Cả 9 mục manifest sau đó
  đúng, runtime manifest đúng. Đây là lỗi chuẩn bị workspace, không phải lỗi
  mô phỏng hoặc lý do chạy lại một cell.

Trạng thái và số liệu chính thức phải lấy từ `queue-finished.json`, validation
và analysis đã khóa; không suy luận hoàn tất từ nhật ký này.

## Validation và chuyển test

- Queue validation khởi chạy một lần, 80/80 wrapper thành công, 1492,8 giây.
- Validate hoàn tất, 80/80 cell hợp lệ. Analyze cho go=true ở cả bảy tiêu chí kỹ thuật.
- Test freeze kế thừa đúng nguồn và selection validation; dry-run đạt.
- Queue test 192 cell khởi chạy một lần, PID 3263111. Kết quả cuối phải đọc từ bằng chứng trên server.

## Test hoàn tất

- 192/192 wrapper thành công, thời gian queue 4105,5 giây; validate 192/192 hợp lệ.
- Analyze: H1 +0,0066167 PDR [0,0037781; 0,0099851], H2 +0,0037817 [0,0011566; 0,0066602], H3 +0,0575083 [0,0507283; 0,0644601]; cả ba bước đạt.
- Toàn vẹn tuyến 144/144; bảy kiểm tra kỹ thuật đạt.
- Archive xuất kết quả SHA-256 97fccbc12cf4c4ced91586f496739d11fe792943404c3d0f05442b68d7e44a12; kiểm toán hash cục bộ đạt.
