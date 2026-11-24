# Bản thảo bài báo NCKH

Tác giả chính/liên hệ: Nguyễn Đình Tâm, 23521389@gm.uit.edu.vn.
Đồng tác giả: Võ Công Vinh. Người hướng dẫn: ThS. Đặng Lê Bảo Chương.

`main.pdf` là bản xuất cuối. `main.tex`, `methods.tex`, `results.tex` và thư
mục `images/` tạo thành dự án LaTeX độc lập, biên dịch bằng XeLaTeX:

```text
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build main.tex
```

Bài báo dùng bố cục chung, chưa theo định dạng một tạp chí hoặc hội nghị
cụ thể. Khi chọn nơi nộp, cần áp mẫu và quy định tác giả của nơi đó.

Các số liệu được lấy từ kết quả v8/v9 đã khóa, được kiểm toán trong
`../nckh_final/evidence/`. Từ gốc kho mã, có thể dựng lại bảng và chép các
hình đã kiểm toán bằng `python reports/nckh_paper/render_results.py`.
Không dùng script đó để chạy mô phỏng hoặc thay đổi phân tích.

Kết quả chính: dự báo GNN cải thiện PDR định tuyến 0,66 điểm phần trăm so
với số đo hiện tại; chưa chứng minh giảm chia cắt hơn trigger phản ứng
trong điều khiển vị trí. Tất cả kết quả là mô phỏng; GNN học có giám sát,
luật điều khiển cố định, không dùng RL hoặc điều khiển UAV thật.
