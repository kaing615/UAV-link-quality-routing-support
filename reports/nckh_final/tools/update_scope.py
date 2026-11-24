"""One-time update of scope prose after adding the prospective v8/v9 studies."""
from pathlib import Path

p=Path(__file__).resolve().parents[1]/'main.tex'
s=p.read_text(encoding='utf-8')
old="Kết quả v4 quay trở lại đúng mục tiêu ``hỗ trợ định tuyến'' của thuyết minh, nhưng với chất lượng liên kết đo được; thay bằng chất lượng liên kết dự báo (GNN) là bước phát triển tiếp theo (Chương~\\ref{chap:conclusion})."
new='V4--v7 kiểm tra hỗ trợ định tuyến bằng chất lượng liên kết đo được. V8 bổ sung một GNN hồi quy tương thích quan sát trực tuyến để so sánh dự báo với số đo hiện tại; v9 kiểm tra riêng dự báo cho kích hoạt bố trí lại relay. Hai nghiên cứu mới giữ các phép so sánh độc lập và không đồng nhất mô hình hồi quy với checkpoint phân loại đa thời điểm của giai đoạn 1.'
assert old in s
s=s.replace(old,new)
old='Phần hỗ trợ định tuyến đã có kết quả vòng kín ở giai đoạn 2, nhưng với chất lượng liên kết \\emph{đo được}. Đề tài chưa đưa GNN vào vòng điều khiển. Kết quả v4 cho thấy nơi một bộ dự đoán chất lượng liên kết có giá trị: thay chất lượng đo được bằng chất lượng dự báo để ghim tuyến \\emph{trước} khi liên kết suy giảm. Đây là cách nối kết quả giai đoạn dự đoán với mục tiêu hỗ trợ định tuyến của thuyết minh.'
new='Phần hỗ trợ định tuyến được kiểm tra trước bằng chất lượng liên kết \\emph{đo được} ở v4--v7, rồi bằng GNN hồi quy mới ở v8. V9 mở rộng câu hỏi sang thời điểm kích hoạt chuyển động relay. Việc tích hợp dự báo vào vòng điều khiển và việc chứng minh lợi ích là hai kết luận riêng: lợi ích chỉ được đánh giá theo endpoint và cổng kiểm định của từng nghiên cứu. Phạm vi mới vẫn chưa hoàn thành các hướng mở rộng đã nêu ở báo cáo tiến độ như GNN theo thời gian, topology phi chuỗi hoặc thử nghiệm bay thật.'
assert old in s
s=s.replace(old,new)
s=s.replace('  \\item Dùng chất lượng liên kết \\emph{dự báo} (mô hình GNN của giai đoạn đầu) để ghim tuyến chủ động.',
            '  \\item Mở rộng dự báo theo thời gian, kiểm tra hiệu chuẩn dưới chuyển động relay và tách riêng ảnh hưởng của số tham số với truyền thông điệp học được. Chỉ học chính sách hành động bằng RL khi đã có baseline luật cố định và thiết kế đánh giá riêng.')
p.write_text(s,encoding='utf-8')
print('Updated scope and future-work prose; outcome conclusions still pending.')
