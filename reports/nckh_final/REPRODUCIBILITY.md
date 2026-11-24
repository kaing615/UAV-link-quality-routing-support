# Hồ sơ tái lập NCKH

Báo cáo chính: `main.tex`; biên dịch bằng XeLaTeX. Nguyễn Đình Tâm là người
thực hiện chính theo xác nhận của nhóm; phân công chi tiết cho các thành viên
trong báo cáo là phương án đề xuất cần đối chiếu với đóng góp thực tế.

## Phân biệt các mô hình và các phép so sánh

- Giai đoạn 1: Edge-SAGE phân loại QoS/survival đa thời điểm, đánh giá ngoại tuyến.
- V2–v7: các nghiên cứu điều khiển dịch vụ, định tuyến và baseline OLSR-ETX.
- V8: Edge-SAGE hồi quy mới, dự báo tỷ lệ nhận probe trung bình năm giây tới;
  giữ UAV relay đứng yên, thay đầu vào cho cùng luật chọn tuyến.
- V9: dùng nguyên trọng số v8; giữ định tuyến theo số đo hiện tại, thay cơ chế
  kích hoạt cùng luật bố trí relay. Chỉ điều khiển vận tốc trong ns-3.

V8/v9 không dùng reinforcement learning. Dự báo học có giám sát; chính sách
định tuyến và chuyển động được ấn định trước. MLP bỏ truyền thông điệp học được
nhưng còn các đặc trưng bậc nút; số tham số không bằng GNN.

## Kế hoạch và dữ liệu

Protocol: `docs/service_v8_protocol.md`, `docs/service_v9_protocol.md` tính từ
gốc kho mã. Chúng ghi giả thuyết, seed, điều kiện go, thứ tự kiểm định và giới hạn
diễn giải. V8 dùng 48 HOLD test v7 để fit và 20 HOLD validation v7 để chọn epoch;
đây là dữ liệu phát triển cho nghiên cứu mới. V8/v9 dùng seed mới, không fit lại
theo PDR hoặc theo dấu kết quả test.

Mô hình và provenance huấn luyện nằm trong
`reports/uav_service_v8/model/selection.json`. Dữ liệu phát triển và hash từng
episode nằm trong `reports/uav_service_v8/development/`. Selection v9 bọc cùng
trọng số v8 với metadata của nghiên cứu vị trí. GNN có 19.137 tham số; MLP có
13.505 tham số. Kiểm tra xuất NumPy so với PyTorch: sai số tối đa 1,19e-7.

## Bằng chứng kết quả

Các kết quả cũ được giữ trong `reports/uav_service_v*/results/`. Với v8/v9,
`export-manifest.json` liệt kê SHA-256 của mọi tệp xuất, gồm:

- `runs/*-validation.json`: tình trạng và endpoint của từng cell;
- `runs/<phase>/<cell>/attempt-0/result.json`, `artifacts.json`, `timings.json`;
- `analysis/<phase>/analysis.json`, `go.json` nếu là validation, `manifest.json`;
- `packages/<phase>/`: protocol, mô hình, mã nguồn được đóng gói và freeze;
- `evidence/queue-<phase>/`: trạng thái cuối và bằng chứng kết thúc queue;
- `descriptive-supplement.json`: thời gian toàn bộ quyết định trên CPU server.

Trace đầy đủ còn trên server trong thư mục study tương ứng; bản xuất cục bộ
giữ hash trace và kết quả, không sao chép toàn bộ dữ liệu thô. Không sửa package
cũ hoặc ghi đè thư mục kết quả khi kiểm toán. Queue nào đã khởi chạy thì kiểm
tra trạng thái và kết quả của queue đó; không khởi chạy lại để theo dõi.

## Kiểm toán và dựng lại hình/báo cáo, không chạy mô phỏng

Từ gốc kho mã:

```text
python reports/nckh_final/tools/prepare_evidence.py
python reports/nckh_final/tools/audit_phase1.py
python reports/nckh_final/tools/prepare_new_evidence.py v8 v9
python reports/nckh_final/tools/render_new_results.py v8 v9
python reports/nckh_final/tools/render_provenance.py
python reports/nckh_final/tools/plot_motion_example.py
python reports/nckh_paper/render_results.py
python reports/nckh_final/tools/audit_tex.py reports/nckh_final/main.tex
cd reports/nckh_final
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Chỉ chạy phần kiểm toán v8/v9 sau khi đã có bản xuất của nghiên cứu tương ứng.
Hình và tệp `evidence/*-audit.json` là tài liệu dẫn xuất, không thay phân tích
đã khóa. Bootstrap giữ cả các arm cùng một khối kịch bản, không xem frame hoặc
gói tin là các lần lặp độc lập. Khi một bước thứ tự cố định không đạt, các bước
sau chỉ mô tả dù CI riêng lẻ có thể dương.

## Giới hạn triển khai

Đây là mô phỏng, chưa là hệ thống bay thực tế. Lưu lượng telemetry và lệnh tập
trung chưa tranh chấp kênh hoặc bị mất. Độ trễ đo quanh `controller.act` gồm
cả luật chọn tuyến, dự báo nếu có và luật chuyển động; không gọi là thời gian
suy luận thuần hoặc độ trễ đầu cuối. Khoảng cách bay chưa quy đổi thành năng
lượng. Trong v9, tốc độ đầu mút danh định 2 m/s khác dữ liệu fit 1 m/s;
cả hai nhân hệ số ngẫu nhiên [0,85; 1,15] theo layout seed. Dữ liệu fit có
relay đứng yên, còn v9 có relay di chuyển; mô hình không được học để dự báo
phản thực theo từng hành động.

## Kiểm tra tài liệu

Skill LaTeX đã được dùng để biên dịch, lint và kiểm tra tham chiếu. Script
`fixrefs.sh` của skill không hiểu `\bibitem` và macro `\Fig` của mẫu hiện tại;
các cảnh báo đó được kiểm tra lại bằng `tools/audit_tex.py`, hỗ trợ cả hai.
Chỉ thành công biên dịch và kiểm tra log cuối mới xác nhận PDF hợp lệ; các
cảnh báo chktex về kiểu dấu gạch, khoảng trắng hoặc đường kẻ bảng không tự
động được coi là lỗi khoa học.
