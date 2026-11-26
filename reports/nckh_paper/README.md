# Bản thảo bài báo NCKH

Tác giả chính/liên hệ: Nguyễn Đình Tâm, 23521389@gm.uit.edu.vn.
Đồng tác giả: Võ Công Vinh. Người hướng dẫn: ThS. Đặng Lê Bảo Chương.

`main.pdf` là bản xuất cuối (13 trang). `main.tex`, `relatedwork.tex`,
`methods.tex`, `results.tex`, `posthoc.tex`, `results_v10.tex` và thư mục
`images/` tạo thành dự án LaTeX độc lập, biên dịch bằng XeLaTeX:

```text
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Bài báo dùng bố cục chung, chưa theo định dạng một tạp chí hoặc hội nghị
cụ thể. Khi chọn nơi nộp, cần áp mẫu và quy định tác giả của nơi đó.

Các số liệu được lấy từ kết quả v8/v9/v8x/v10 đã khóa, được kiểm toán trong
`../nckh_final/evidence/`. Từ gốc kho mã, có thể dựng lại bảng và chép các
hình đã kiểm toán bằng `python reports/nckh_paper/render_results.py`.
Không dùng script đó để chạy mô phỏng hoặc thay đổi phân tích. Script chỉ sinh
`results.tex` (v8/v9); `results_v10.tex` viết tay từ
`reports/uav_service_v10/results/analysis/test/analysis.json`.

## Luận đề của bài

Bài báo **không** tuyên bố đề xuất một bộ dự báo GNN tốt hơn. Câu hỏi của nó là
dự báo chất lượng liên kết học được có cải thiện *dịch vụ mạng* hay không, khi
được đặt vào vòng kín và so với baseline đã chỉnh tham số. Ba kết quả dương:

1. Thứ hạng theo sai số dự báo đi **ngược** thứ hạng theo dịch vụ: EWMA 0,1 có
   MSE xấu nhất (0,00165 so với 0,00078 của GNN) nhưng PDR cao nhất, hơn GNN
   0,31 điểm phần trăm [0,04; 0,61].
2. Toàn bộ mức tăng của GNN (+0,66 pp so với EWMA 0,5) đạt được bằng một EWMA
   một tham số được chỉnh hệ số, nên lợi ích gắn với ước lượng ổn định hơn,
   không phải giá trị của thông tin dự báo.
3. Giới hạn hình học: relay cách đều lý tưởng vẫn còn 26,09 s chia cắt mỗi
   lượt, nên trigger phản ứng đã đạt ~94% mức giảm khả thi — kết quả âm của v9
   có lời giải thích định lượng.

Ba kết quả đã đăng ký trước: v8 H1/H2/H3 đạt, v9 endpoint chính **không đạt**,
v10 H1 **không phát hiện khác biệt** (+0,08 pp [−0,11; 0,29]). Ba phân tích
trên ban đầu là **thăm dò sau test**; kết luận số 2 (làm mượt, không phải dự
báo) đã được v10 kiểm chứng bằng một phép kiểm định đăng ký trước
(`docs/service_v10_protocol.md`, chạy 24/11/2026, Mục `results_v10.tex`). Hai
kết luận còn lại vẫn là thăm dò. `docs/service_v11_protocol.md` đăng ký phần độ
bền theo seed huấn luyện cùng ablation cùng số tham số và **chưa chạy**.

Tất cả kết quả là mô phỏng; GNN học có giám sát, luật điều khiển cố định, không
dùng RL hoặc điều khiển UAV thật. So sánh với OLSR-ETX (+5,75 pp) là giới hạn
trên của một bộ điều khiển tập trung lý tưởng hóa một phần: telemetry và lệnh
chưa được truyền thành gói cạnh tranh kênh, chi phí báo hiệu 5,55 kB/s so với
1,78 kB/s.

## Còn lại trước khi nộp

- **Cần quyết định:** chọn nơi nộp. Trong nước (NICS/SOICT/kỷ yếu trường) thì
  bản hiện tại đã nộp được; quốc tế (IEEE Access, Drones, Ad Hoc Networks) cần
  dịch tiếng Anh và áp template.
- **Cần người hướng dẫn thông qua:** luận đề đã đổi từ "đề xuất GNN" sang
  "kiểm chứng giá trị thực của dự báo học được".
- Chạy v11 (v10 đã xong, α\* = 0,1 là baseline của v11).
