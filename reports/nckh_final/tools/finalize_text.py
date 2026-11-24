"""One-time editorial patch after both locked studies completed; do not rerun."""
import json
from pathlib import Path

REPORT=Path(__file__).resolve().parents[1]
ROOT=REPORT.parents[1]
PAPER=ROOT/'reports/nckh_paper'
v8=json.loads((REPORT/'evidence/v8-audit.json').read_text(encoding='utf-8'))
v9=json.loads((REPORT/'evidence/v9-audit.json').read_text(encoding='utf-8'))
assert all(s['success'] and s['interpretable'] for s in v8['fixed_sequence'])
assert not v9['fixed_sequence'][0]['success']
assert v8['counts']==v9['counts']==dict(validation=80,test=192)

abstract=r'''Đề tài nghiên cứu việc dùng chất lượng liên kết để hỗ trợ định tuyến và điều khiển vị trí trong mạng phương tiện bay không người lái (UAV). Dự báo chính xác chưa bảo đảm hành động điều khiển có ích, nên nghiên cứu đánh giá riêng dự báo ngoại tuyến và dịch vụ mạng vòng kín. Giai đoạn đầu dùng mạng nơ-ron đồ thị trên 210 lượt mô phỏng và cho thấy ưu thế Macro-F1, đồng thời ghi nhận lỗi dự đoán liên kết hỏng thành ổn định. Các vòng tiếp theo phát triển ghim tuyến theo chất lượng hiện tại; ở vòng kiểm tra độ bền, phương pháp tăng tỷ lệ gói đến đích (PDR) 5,06 điểm phần trăm so với OLSR-ETX đã tinh chỉnh, với chi phí báo hiệu cao hơn. Hai nghiên cứu bổ sung dùng bộ hồi quy đồ thị mới dự báo tỷ lệ nhận probe trung bình năm giây tới, học từ dữ liệu lịch sử và khóa trước mô phỏng mới. Mỗi nghiên cứu có 80 lượt validation và 192 lượt test trên chuỗi năm hoặc bảy UAV; phân tích bootstrap theo khối seed, giữ các nhóm so sánh ghép cặp. Khi giữ relay đứng yên và cùng luật chọn tuyến, dự báo đồ thị tăng PDR 0,66 điểm phần trăm so với chất lượng hiện tại, khoảng tin cậy 95\% [0,38; 1,00], và tăng 0,38 điểm so với mô hình bỏ truyền thông điệp. Khi giữ cùng định tuyến và luật bố trí relay, kích hoạt chuyển động bằng dự báo chỉ giảm 0,20 giây chia cắt so với kích hoạt theo chất lượng hiện tại, khoảng tin cậy [$-1,05$; $1,45$]; endpoint chính không đạt. Di chuyển giảm chia cắt rõ trong so sánh mô tả với đứng yên, nhưng cần khoảng 637 m tổng quãng đường relay mỗi lượt. Kết quả ủng hộ lợi ích dự báo cho định tuyến trong phạm vi đã xét, chưa chứng minh lợi ích thêm cho trigger vị trí. Hệ thống mới được kiểm tra trong ns-3, chưa tích hợp UAV thật; telemetry tập trung chưa cạnh tranh kênh, chỉ có một seed huấn luyện và một họ topology. Dữ liệu, mã nguồn đóng gói và kết quả được liên kết bằng hash để kiểm tra tái lập.'''

paper_abstract=r'''Nghiên cứu đánh giá giá trị của dự báo chất lượng liên kết đối với hai hành động trong mạng phương tiện bay không người lái (UAV): chọn tuyến dữ liệu và kích hoạt bố trí lại relay. Mạng nơ-ron đồ thị dự báo tỷ lệ nhận probe trung bình trong năm giây tiếp theo từ vị trí, vận tốc và số đo liên kết nhân quả. Mô hình được học có giám sát từ dữ liệu lịch sử; mô hình đối chứng bỏ truyền thông điệp học được, nhưng vẫn giữ đặc trưng bậc nút và có ít tham số hơn. Hai nghiên cứu vòng kín được khóa trước khi chạy, mỗi nghiên cứu gồm 80 lượt validation và 192 lượt test mới trong ns-3.45, với chuỗi năm hoặc bảy UAV và các nhóm so sánh ghép cặp theo seed. Nghiên cứu định tuyến giữ relay đứng yên và cùng luật ghim tuyến, chỉ thay chất lượng hiện tại bằng dự báo. Trên 40 khối gradual test, dự báo đồ thị tăng tỷ lệ gói đến đích (PDR) 0,66 điểm phần trăm so với chất lượng hiện tại, khoảng tin cậy 95\% [0,38; 1,00], và tăng 0,38 điểm so với mô hình bỏ truyền thông điệp, khoảng tin cậy [0,12; 0,67]. Nghiên cứu điều khiển vị trí giữ cùng định tuyến và luật bố trí, chỉ thay điều kiện kích hoạt chuyển động. Dự báo giảm trung bình 0,20 giây chia cắt đồ thị probe so với kích hoạt phản ứng, khoảng tin cậy [$-1,05$; $1,45$], nên endpoint chính không đạt. Các nhóm di chuyển có khoảng 31 giây chia cắt so với 123 giây khi đứng yên, nhưng cần khoảng 637 m tổng quãng đường relay; đây là so sánh mô tả, không chứng minh lợi ích riêng của dự báo. Kết quả cho thấy cần đánh giá bộ dự báo cùng hành động cụ thể: GNN cải thiện định tuyến trong phạm vi đã thử, còn trigger vị trí chưa thể hiện lợi ích thêm rõ. Các giới hạn gồm telemetry tập trung chưa tranh chấp kênh, một seed huấn luyện, thay đổi phân phối chuyển động ở nghiên cứu vị trí và chưa có thử nghiệm bay thật.'''

p=REPORT/'main.tex'
s=p.read_text(encoding='utf-8')
start=s.index(r'\addcontentsline{toc}{chapter}{TÓM TẮT NỘI DUNG}')+len(r'\addcontentsline{toc}{chapter}{TÓM TẮT NỘI DUNG}')
end=s.index(r'\clearpage',start)
s=s[:start]+'\n'+abstract+'\n\n'+r'\noindent\textbf{Từ khóa:} mạng UAV; mạng nơ-ron đồ thị; chất lượng liên kết; định tuyến; điều khiển vị trí.'+'\n'+s[end:]
s=s.replace('áp dụng cho sáu vòng v2--v7 (1.980 episode hợp lệ được liệt kê trong bảng tổng quan v2--v7, không cộng smoke và các lần tái hiện chẩn đoán).',
            'áp dụng cho tám vòng v2--v9 (2.524 episode hợp lệ trong các pha chính thức; không cộng 210 lượt ngoại tuyến, smoke hoặc các lần tái hiện chẩn đoán).')
anchor=r'\section{Cấu trúc báo cáo}'
index=s.index(anchor)
before=s[:index]
endlist=before.rfind(r'\end{itemize}')
before=before[:endlist]+r'''  \item Tích hợp GNN hồi quy vào luật chọn tuyến và kiểm tra bằng seed mới: tăng PDR 0,66 điểm phần trăm so với số đo hiện tại; mô hình xuất NumPy, truy vết dự báo và kiểm tra lệnh thực thi.
  \item Bộ điều khiển vị trí relay với cùng luật bố trí và ba trigger đối chiếu; công bố kết quả không đạt về lợi ích thêm của dự báo, cùng số giây chia cắt, quãng đường và kiểm tra an toàn mô phỏng.
'''+before[endlist:]
s=before+s[index:]
s=s.replace(r'v8&GNN dự báo để ghim tuyến&80 + 192&Dương: $+0{,}66$ [0,38; 1,00] so với EWMA hiện tại, 3/3 kiểm định đạt.\\\bottomrule',
            r'v8&GNN dự báo để ghim tuyến&80 + 192&Dương: $+0{,}66$ [0,38; 1,00] so với EWMA hiện tại, 3/3 kiểm định đạt.\\'+'\n'+
            r'v9&Dự báo kích hoạt bố trí relay&80 + 192&Endpoint chính không đạt: giảm 0,20 s chia cắt [$-1{,}05$; $1{,}45$] so với reactive; vẫn báo cáo chi phí bay.\\\bottomrule')
start=s.index(r'\section{Kết luận}',s.index(r'\chapter{Kết luận và hướng phát triển}'))+len(r'\section{Kết luận}')
end=s.index(r'\section{Hướng phát triển}',start)
s=s[:start]+r'''
Đề tài đã nối bài toán dự báo liên kết với hai hành động điều khiển cụ thể
trong mô phỏng mạng UAV. GNN phân loại đa thời điểm ở giai đoạn 1 đạt
Macro-F1 cao hơn các baseline trên seed giữ lại, đồng thời còn lỗi
false-stable. Chuỗi nghiên cứu v2--v7 cho thấy cần kiểm tra nguồn mất gói và
hành động sử dụng dự báo: ghim tuyến theo chất lượng hiện tại cải thiện
PDR trong các kịch bản đã xét, trong khi các bộ điều khiển di chuyển đầu
tiên không tạo lợi ích ổn định. Kết quả âm v3/v5 và lỗi ARP được giữ trong
báo cáo, không loại bỏ khỏi quá trình nghiên cứu.

V8 dùng GNN hồi quy mới và giữ nguyên luật ghim tuyến, tăng PDR 0,66 điểm
phần trăm so với EWMA hiện tại (CI 95\% [0,38; 1,00]) và 0,38 điểm so với
MLP ([0,12; 0,67]) trên test mới; cả ba bước kiểm định đạt, toàn vẹn tuyến
144/144. Đây là bằng chứng vòng kín cho lợi ích của bộ dự báo đã triển khai
trong định tuyến, với phạm vi và chi phí telemetry chưa đầy đủ đã nêu.

V9 đã thực thi điều khiển vị trí relay, kiểm tra an toàn mô phỏng và toàn
vẹn tuyến 192/192. Di chuyển hữu ích trong so sánh mô tả với đứng yên ở
kịch bản tách xa nhanh, nhưng GNN chưa giảm chia cắt rõ so với trigger theo
chất lượng hiện tại: 0,20 giây, CI [$-1{,}05$; $1{,}45$]. Vì vậy không kết
luận dự báo đã cải thiện điều khiển vị trí hoặc ngăn được mọi chia cắt.
Các kết quả hỗ trợ một hệ thống định tuyến dự báo cùng baseline điều khiển
vị trí có giới hạn, và cung cấp bằng chứng để thiết kế nghiên cứu tiếp theo.
Chưa có điều khiển UAV thật, mô hình năng lượng hoặc chính sách học tăng cường.

'''+s[end:]
p.write_text(s,encoding='utf-8',newline='\n')

p=PAPER/'main.tex'
s=p.read_text(encoding='utf-8')
start=s.index(r'\begin{abstract}')+len(r'\begin{abstract}')
end=s.index(r'\end{abstract}',start)
s=s[:start]+'\n'+paper_abstract+'\n'+s[end:]
start=s.index(r'\section{Thảo luận}')+len(r'\section{Thảo luận}')
end=s.index(r'\section{Kết luận}',start)
limits=s[start:end]
discussion=r'''
Hai nghiên cứu cho các kết luận khác nhau về cùng một bộ dự báo. Với luật
ghim tuyến, GNN đem lại lợi ích thêm nhỏ nhưng CI dương; so sánh với MLP
ủng hộ lợi ích của cấu hình GNN đã thử, chưa cô lập riêng ảnh hưởng của
truyền thông điệp vì số tham số khác nhau. Việc ít đổi tuyến hơn là một
giải thích khả dĩ, chưa được can thiệp kiểm tra riêng.

Với điều khiển vị trí, endpoint chính không đạt. Kết quả GNN--MLP có CI
riêng lẻ dương chỉ mang tính mô tả vì cổng đầu tiên đã đóng. Trigger reactive
dùng ngưỡng chất lượng 0,70, thường đã hành động trước chia cắt đồ thị
ngưỡng 0,5; nó không chờ đến khi mất liên lạc mới di chuyển. Do đó dư địa
cho dự báo năm giây có thể nhỏ. Chuyển miền vận tốc và việc mô hình chỉ
được fit trên relay đứng yên cũng là các khả năng cần kiểm tra tiếp.
Không có phép thử riêng để chọn một giải thích làm nguyên nhân duy nhất.

Di chuyển giảm mạnh mất kết nối so với relay đứng yên ở kịch bản nhanh,
nhưng lợi ích đó chưa chứng minh phần đóng góp riêng của dự báo. Chi phí
hình học tương tự giữa trigger GNN và reactive, trong khi GNN cần tính
toán và telemetry bổ sung. CI chứa 0 không chứng minh tương đương; cũng
không đủ căn cứ để thêm một mô hình phức tạp chỉ từ chênh lệch điểm ước lượng.
Không thay ngưỡng hoặc huấn luyện lại sau khi thấy test. Hướng tiếp theo
là thu thập dữ liệu relay di chuyển, kiểm tra hiệu chuẩn và báo hiệu trên
kênh, rồi đăng ký một nghiên cứu mới nếu học chính sách bằng RL.
'''
s=s[:start]+discussion+'\n'+limits+s[end:]
start=s.index(r'\section{Kết luận}')+len(r'\section{Kết luận}')
end=s.index(r'\section*{Thông tin tác giả và dữ liệu}',start)
s=s[:start]+r'''
Trong hai nghiên cứu đã khóa trên chuỗi 5/7 UAV, GNN hồi quy năm giây
cải thiện PDR khi dùng cho lựa chọn tuyến, nhưng chưa thể hiện lợi ích
thêm rõ khi chỉ kích hoạt một luật bố trí relay chung. Kết luận về dự báo
cần gắn với hành động và baseline cụ thể; độ chính xác dự báo cao hơn
không thay thế thí nghiệm dịch vụ vòng kín. Hệ thống và bằng chứng hiện
có phù hợp cho đánh giá mô phỏng trong phạm vi công bố, chưa xác nhận
khả năng triển khai bay thật.
'''+s[end:]
p.write_text(s,encoding='utf-8',newline='\n')
print(json.dumps(dict(report_abstract_words=len(abstract.split()),paper_abstract_words=len(paper_abstract.split()),
                      new_formal_episodes=544,total_closed_loop_episodes=2524),indent=2))
