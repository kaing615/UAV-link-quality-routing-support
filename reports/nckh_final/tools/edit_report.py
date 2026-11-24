"""One-time, checked editorial corrections before adding prospective experiments."""
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "main.tex"
text = path.read_text(encoding="utf-8")


def replace(old, new):
    global text
    if old not in text:
        raise ValueError("Missing editorial anchor: " + old[:100])
    text = text.replace(old, new)


replace(r"1&\HoTenA&\MSSVA&\textit{[Nhóm điền nhiệm vụ.]}",
        r"1&\HoTenA&\MSSVA&Thực hiện chính: thiết kế nghiên cứu, mô hình dự báo, tích hợp mô phỏng, phân tích kết quả và bản thảo chính.")
replace(r"2&\HoTenB&\MSSVB&\textit{[Nhóm điền nhiệm vụ.]}",
        r"2&\HoTenB&\MSSVB&Hỗ trợ tổng hợp tài liệu, kiểm tra dữ liệu/kết quả, chuẩn bị hình/bảng và rà soát trình bày.")
replace(r"\caption{Phân công nhiệm vụ trong nhóm.}",
        r"\caption{Phân công nhiệm vụ đề xuất. Vai trò thực hiện chính của Nguyễn Đình Tâm đã được xác nhận; nhiệm vụ chi tiết cần nhóm đối chiếu trước khi nộp.}")
replace("gấp khoảng 3 lần OLSR-ETX, chủ yếu do probe đo liên kết", "gấp 3,12 lần OLSR-ETX-fast và 12,48 lần OLSR-ETX thường, chủ yếu do probe đo liên kết")
replace("chuỗi 4 hoặc 5 UAV (2 hoặc 3 relay)", "chuỗi 4--7 UAV (2--5 relay)")
replace("Không thử nghiệm trên phần cứng thật.", "Giai đoạn dự đoán ngoại tuyến dùng mạng 20 UAV. Bộ điều khiển tác động lên tuyến dữ liệu hoặc vận tốc relay trong simulator; chưa thử nghiệm UAV thật, chưa tích hợp autopilot và chưa đo năng lượng bay.")
replace("tổng cộng 2.284 episode có dữ liệu hợp lệ, không ô nào thiếu", "1.980 episode hợp lệ được liệt kê trong bảng tổng quan v2--v7, không cộng smoke và các lần tái hiện chẩn đoán")
replace("Lợi ích đó đến từ việc dùng ETX hay từ điều khiển tập trung, tức là ghim tuyến có tốt hơn OLSR-ETX phân tán hay không?", "Hệ thống ghim tuyến có tốt hơn bản OLSR-ETX phân tán trong cùng môi trường hay không? So sánh này chưa cô lập riêng tác động của tính tập trung.")
replace("Chẩn đoán định lượng nguyên nhân: lựa chọn tuyến min-hop của OLSR là nguồn mất gói chính, không phải vị trí relay.", "Chẩn đoán định lượng: phần lớn tổn thất trong các kịch bản đã xét gắn với tuyến trực tiếp do OLSR chọn theo số hop; từ đó đặt giả thuyết kiểm tra điều khiển tuyến.")
replace("lợi ích của ghim tuyến giữ nguyên", "lợi ích của ghim tuyến còn quan sát được")
replace("Phần lớn công trình đánh giá giao thức theo số hop hoặc vị trí; chất lượng liên kết đo được ít khi là tiêu chí chọn tuyến chính.", "Định tuyến theo chất lượng hoặc độ bền liên kết là một hướng đã có; đóng góp của đề tài cần được đối chiếu ở mức cơ chế và bằng chứng thực nghiệm cụ thể.")
replace("Kết quả giai đoạn 2 cho thấy biến đúng là tuyến dữ liệu;", "Trong các kịch bản đã xét, kết quả giai đoạn 2 tạo cơ sở tiếp tục kiểm tra điều khiển tuyến;")
replace("v5, v6 thêm arm OLSR-ETX", "v5--v7 thêm arm OLSR-ETX")
replace("v6 dùng 1 s", "v6--v7 dùng 1 s")
replace("Chuỗi n4 (2 relay) hoặc n5 (3 relay)", "Chuỗi n4/n5; v7 dùng n5/n7 (3/5 relay)")
replace(r"\hat\Delta=\tfrac12\left(\bar\delta_{n4}+\bar\delta_{n5}\right)", r"\hat\Delta=\frac{1}{|\mathcal G|}\sum_{g\in\mathcal G}\bar\delta_g")
replace("với $k$ là cặp hạt giống. CI 95", r"với $k$ là cặp hạt giống, $\mathcal G=\{n4,n5\}$ ở v2--v6 và $\mathcal G=\{n5,n7\}$ ở v7. Đơn vị lấy mẫu lại là kịch bản, không phải từng gói hoặc từng arm. Riêng endpoint chính v7 có 40 khối gradual, mỗi khối chứa 5 arm; 40 episode static còn lại chỉ dùng mô tả. CI 95")
replace("gần với hành vi của Linux (phân giải lại ở gói kế tiếp). Đây là thay đổi duy nhất so với v5.", "để giới hạn thời gian giữ trạng thái dead trong thí nghiệm. Đây là thay đổi cấu hình mô phỏng duy nhất so với v5; không xem đây là bằng chứng tái hiện chính xác toàn bộ hành vi ARP của Linux.")
replace("tức là chọn baseline mạnh nhất.", "tức là chọn cấu hình có PDR cao nhất trong ba ứng viên trên tập tuning riêng; chưa tối ưu mọi tham số OLSR-ETX.")
replace("gần như không mất lợi ích khi lệnh trễ 2 giây ($-0{,}24$ điểm \\%),", "arm trễ 2 giây vẫn hơn ETX-fast; so sánh trực tiếp hai arm ghim tuyến có CI chứa 0 nên chưa chứng minh tương đương hoặc không ảnh hưởng của độ trễ,")
replace("Lý do là điểm yếu của OLSR-ETX nằm ở các khoảng không có tuyến khi tuyến được tính lại, không nằm ở việc đo chất lượng chậm.", "Dữ liệu mất gói theo loại tuyến gợi ý vai trò của các khoảng không có tuyến. Tuy nhiên, chưa có can thiệp tách riêng từng cơ chế để kết luận đây là nguyên nhân duy nhất hoặc tốc độ đo không có vai trò.")
replace("Cái giá phải trả là báo hiệu nhiều hơn khoảng 3 lần, chủ yếu do probe.", "Chi phí phương pháp cao hơn 3,12 lần ETX-fast và 12,48 lần ETX thường, chủ yếu do probe; đây là số byte tầng IP cộng phần báo cáo ước lượng, chưa phải airtime hay năng lượng.")
replace("Bước 2 và 3 của kiểm định đạt nhờ phần ghim tuyến; không thể kết luận rằng di chuyển đem lại lợi ích thêm.", "Bước 2 và 3 kiểm tra hệ thống kết hợp so với các baseline đã định, không tách được phần đóng góp riêng của chuyển động; chưa thể kết luận di chuyển đem lại lợi ích thêm một cách ổn định.")
replace("và phần lớn triệt tiêu trong so sánh ghép cặp.", "nên ghép cặp giảm một phần nhiễu chung, nhưng không bảo đảm tác động lỗi triệt tiêu giữa các arm.")
replace("nên so sánh ghép cặp ít bị lệch, nhưng PDR tuyệt đối ở n5 thấp hơn thực tế.", "nên ghép cặp có thể giảm nhiễu chung; không thể suy ra PDR thực tế hoặc bảo đảm so sánh hoàn toàn không bị ảnh hưởng.")
replace("Đăng ký trước&Cố định giả thuyết, tham số, tiêu chí và phép phân tích trước khi chạy thí nghiệm.", "Đăng ký trước&Cố định giả thuyết, tham số, tiêu chí và phân tích trước từng vòng; ở đây lưu bằng commit/protocol và freeze, không hàm ý đăng ký trên một registry độc lập.")
replace("nên không cần hiệu chỉnh đa so sánh.", "để hạn chế chọn kết quả thuận lợi; các so sánh ngoài thứ tự này chỉ được diễn giải mô tả.")
replace("\n\\section{Khoảng trống}\n", r"""
\subsection{Các phương pháp gần với câu hỏi nghiên cứu}
Predictive-OLSR~\cite{rosati2013} dùng GPS và vận tốc tương đối để điều chỉnh ETX. OLSR-LCN~\cite{liu2025} kết hợp tuổi thọ liên kết, nhiễu và tải nút, đồng thời mở rộng HELLO/TC và cách chọn MPR/tuyến. Các hướng này cho thấy dự báo và chất lượng liên kết đã được dùng cho định tuyến UAV. Bảng~\ref{tab:related} chỉ đối chiếu cơ chế; chưa có thí nghiệm chung để xếp hạng hiệu năng giữa chúng và đề tài.
\begin{table}[H]
\centering\small
\begin{tabularx}{\linewidth}{@{}P{2.7cm}YY@{}}
\toprule\textbf{Phương pháp}&\textbf{Thông tin và cơ chế}&\textbf{Đối chiếu với đề tài}\\\midrule
OLSR-ETX~\cite{olsrd}&Quảng bá chất lượng liên kết; chọn tổng ETX nhỏ.&Baseline phân tán do nhóm cài đặt; chưa phải olsrd chạy trên UAV thật.\\
Predictive-OLSR~\cite{rosati2013}&GPS, vận tốc tương đối; điều chỉnh trọng số ETX.&Dự báo theo chuyển động; chưa so sánh trực tiếp trong đề tài.\\
OLSR-LCN~\cite{liu2025}&Tuổi thọ liên kết, nhiễu, tải; thay đổi MPR và tuyến.&Khác cơ chế và môi trường; chưa thể tuyên bố vượt phương pháp này.\\
Ghim tuyến v4--v7&Probe, EWMA, ETX, hysteresis; tập trung cài tuyến.&PDR vòng kín, kiểm tra lệnh/tuyến, seed riêng và đối chứng HELLO nhanh.\\\bottomrule
\end{tabularx}
\caption{Đối chiếu cơ chế với công trình liên quan.}\label{tab:related}
\end{table}

\section{Khoảng trống}
""")
replace("\n\\subsection{Chi phí báo hiệu}\n", r"""
\Fig{fig_v7_paired}{Chênh PDR của từng khối gradual v7 so với ETX-fast; mỗi điểm là một so sánh ghép cặp, không phải một gói tin độc lập.}{fig:v7paired}
Trong static, chênh PDR ghim tuyến so với ETX-fast là $+0{,}17$ điểm phần trăm, CI 95\% [$-0{,}22$; $0{,}69$], với 4 khối mỗi cấu hình. Dữ liệu này chưa cho thấy lợi ích rõ trong static. So sánh ghim tuyến với arm trễ 2 giây và các chỉ số luồng nền là mô tả; CI chứa 0 của so sánh độ trễ không phải kiểm định tương đương.

\subsection{Chi phí báo hiệu}
""")
replace("So với luồng chính (32 kB/s), báo hiệu của ghim tuyến bằng khoảng 17\\%, của OLSR-ETX tinh chỉnh khoảng 6\\%.", "Tổng trên là byte tầng IP cộng báo cáo ước lượng, tính trên các lần phát trong mạng; không quy đổi trực tiếp thành phần trăm dung lượng kênh hay năng lượng bằng cách chia cho tốc độ tải tại nguồn. Probe vẫn chạy ở mọi arm, nhưng chỉ được hạch toán vào chi phí phương pháp của các arm ghim tuyến. Phần báo cáo trung tâm chưa chiếm kênh trong mô phỏng, nên chưa đo được lợi ích sau khi mô phỏng đầy đủ mọi báo hiệu.")
replace(r"\Fig{fig_v7_cost}{Chi phí báo hiệu và PDR trung bình (gradual) của các arm, test v7.}{fig:v7cost}",
        r"\Fig{fig_v7_cost}{Chi phí phương pháp và PDR trung bình gradual, v7; phần báo cáo trung tâm là ước lượng.}{fig:v7cost}" + "\n" +
        r"\Fig{fig_v7_overhead}{Thành phần chi phí phương pháp v7: phần gạch chéo là ước lượng chưa chiếm kênh. Probe 5,018 kB/s vẫn được phát trong mọi arm, nhưng chỉ hạch toán cho phương pháp sử dụng probe.}{fig:v7overhead}")
replace(r"\bibitem{tiendo}", r"""\bibitem{rosati2013} S. Rosati, K. Kruzelecki, L. Traynard, B. Rimoldi. \textit{Speed-Aware Routing for UAV Ad-Hoc Networks}. arXiv:1307.6350, 2013. \url{https://arxiv.org/abs/1307.6350}.
\bibitem{liu2025} J. Liu, P. Gong, H. Yang, S. Li, X. Gao. Cross-Layer Optimized OLSR Protocol for FANETs in Interference-Intensive Environments. \textit{Drones}, 9(11):778, 2025. \url{https://doi.org/10.3390/drones9110778}.
\bibitem{tiendo}""")
path.write_text(text, encoding="utf-8")
print("Editorial corrections written to", path)
