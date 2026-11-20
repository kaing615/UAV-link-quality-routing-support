# Tái kiểm tra báo cáo tiếng Việt

Bản này biên tập lời văn và bố cục từ kết quả formal đã phân tích. Renderer reporting.py đã đóng băng được chạy trước; bản dựng đó vẫn giữ nguyên ở đường dẫn ghi trong reproduce.json. Không thu thập, fit hay tính lại chỉ số khi dựng báo cáo.

## Dữ liệu và kiểm tra

Tính SHA256 từng tệp trong artifact-inventory.json từ thư mục bản báo cáo. tables/analysis.json và engineering.json giữ nguyên byte đầu vào. tables/all-planned-cells.csv giữ mọi ô kế hoạch, số gói, quãng đường từng relay, ngân sách, lần thử và endpoint. Các JSON còn lại giữ các giá trị tổng hợp, so sánh, kiểm duyệt, cảnh báo, thời gian xử lý và subgroup. numerical-source-ledger.json ghi con trỏ nguồn cho từng giá trị được in trong PDF.

## Tái lập ở thư mục mới

Đọc reproduce.json và tables/retained-inventory.json để lấy ánh xạ gốc nghiên cứu sang đường dẫn tương đối hiện có, hash nguồn, protocol, mô hình, runtime, dữ liệu thô và các argument array. Dùng nguồn đã đóng băng; thay đích validation/analysis/report bằng thư mục mới chưa tồn tại. Không sửa đường dẫn bên trong frozen evidence. Chỉ dùng lệnh validate/analyze/report chỉ đọc dữ liệu; README này không tự chạy mô phỏng hoặc fit.

Dữ liệu thô được giữ tại các đường dẫn trong inventory, không nằm hết trong thư mục PDF này. Nếu có tệp thiếu, gói không thể được gọi là đầy đủ. Audit development ban đầu thiếu mã thoát trực tiếp vẫn là giới hạn của hồ sơ, dù các đầu ra đã xác minh được giữ nguyên.
