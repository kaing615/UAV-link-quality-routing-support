# Script thuyết trình — 18 slide

Slide là hình minh họa; phần dưới là lời nói, giống ghi chú người thuyết trình trong Canva.
Sau 2 slide mở đầu và 2 slide kiến trúc là 10 bước đã làm (thêm 3 slide về cách xây GNN), mỗi bước nói theo thứ tự: **ĐÃ LÀM → VÌ SAO → CÁCH LÀM → KẾT QUẢ → TIẾP THEO**.

---

## Mở đầu

### 1. Trang bìa (~20 s)
Đề tài: dùng chất lượng liên kết để hỗ trợ định tuyến trong chuỗi UAV relay, và kiểm tra dự báo bằng GNN có giúp thêm không.
Em trình bày theo 10 bước đã làm. Mỗi bước nói: đã làm gì, vì sao, kết quả, bước tiếp theo.
Kết luận nói trước: chọn tuyến theo chất lượng liên kết giúp rõ rệt; GNN dự báo sát hơn nhưng không làm định tuyến tốt hơn EWMA đã chỉnh.

### 2. OLSR bỏ qua chất lượng liên kết (~40 s)
- **VẤN ĐỀ:** Chuỗi UAV relay, hai đầu bay tách xa nên tỉ lệ nhận gói trên từng liên kết giảm dần và dao động. OLSR chọn đường ít chặng nhất, không xét liên kết mạnh hay yếu. Khi liên kết đổi phải chờ HELLO 2 s, TC 5 s mới tính lại tuyến; trong khoảng đó gói bị rớt.
- **MÔI TRƯỜNG:** ns-3.45, Wi-Fi 802.11g, OLSR, fading Nakagami. Chuỗi 4–7 UAV, hai đầu tách xa dần (gradual) và đứng yên (static). Mỗi lượt 240 s. Probe mỗi 0,05 s, gộp theo 1 s ra tỉ lệ nhận từng liên kết.
- **Chỉ số chính:** PDR (tỉ lệ gói đến đích), đo bằng FlowMonitor.

### 3. Kiến trúc tổng quan hệ thống (~40 s) — [hàng trên rồi hàng dưới]
- **GIAI ĐOẠN 1 (ngoại tuyến):** ns-3 sinh 210 lượt mô phỏng mạng 20 UAV → tạo đặc trưng nút, cạnh và nhãn tương lai t+k → huấn luyện GNN Edge-SAGE cùng 3 đối chứng (XGBoost, Logistic Regression, persistence) → chấm Macro-F1 trên seed giữ lại.
- **GIAI ĐOẠN 2 (vòng kín):** ns-3 chạy chuỗi 4–7 UAV, mỗi giây gửi số đo probe sang Python → Python ước lượng chất lượng (EWMA hoặc GNN) → tính ETX, chọn và ghim tuyến → ns-3 áp tuyến, FlowMonitor đo PDR.
- **VÌ SAO TÁCH HAI GIAI ĐOẠN:** dự báo đúng ngoại tuyến chưa chắc làm mạng tốt hơn; chỉ đo được lợi ích thật khi quyết định tác động ngược lên mạng.
- **QUY TRÌNH CHUNG:** đăng ký giả thuyết trước; khóa mã nguồn, mô hình, seed bằng SHA-256; so cặp trên cùng kịch bản, bootstrap 10.000 lần; kiểm tra lệnh của bộ điều khiển trùng với tuyến ns-3 đã dùng.

### 4. Vòng kín ns-3 ↔ Python, mỗi giây (~40 s) — [trái → phải → xuống dưới]
- **ĐÃ LÀM:** Nối ns-3 với bộ điều khiển Python qua stdin/stdout, chạy đồng bộ từng giây.
- **CÁCH LÀM:** Mỗi giây ns-3 gửi một frame quan sát (vị trí, ma trận tỉ lệ nhận probe, tuyến OLSR hiện tại, bộ đếm gói) rồi chờ lệnh. Python ước lượng chất lượng (EWMA hoặc GNN), tính ETX = 1/(q_ab·q_ba), áp luật ghim tuyến và trả lệnh UAVACT. Lớp OverrideRouting (kế thừa Ipv4ListRouting) chuyển gói theo tuyến đã ghim; ngoài tuyến đó vẫn dùng OLSR. v2, v3 lệnh là vận tốc relay; từ v4 lệnh là tuyến.
- **KIỂM TRA:** ns-3 ghi log UAVROUTE mỗi lần đổi tuyến, Python ghi decisions.jsonl; hai chuỗi phải trùng khớp hoàn toàn.
- **VÌ SAO:** Sai số dự báo ngoại tuyến không cho biết PDR; phải đo khi quyết định tác động ngược lên mạng.

---

## 10 bước đã làm

### 5. Bước 1 – GNN dự đoán liên kết (~45 s)
- **ĐÃ LÀM:** Huấn luyện GNN Edge-SAGE dự đoán liên kết sẽ ổn định hay không ở 4 mốc dự báo, chạy ngoại tuyến.
- **VÌ SAO:** Chất lượng một liên kết phụ thuộc cả các UAV xung quanh, nên mô hình nhìn cả đồ thị mạng.
- **CÁCH LÀM:** 210 lượt mô phỏng (21 điều kiện × 10 seed), mạng 20 UAV. Thông điệp ghép nút lân cận với 7 đặc trưng cạnh (khoảng cách, RSSI, SNR, trễ, mất gói, thông lượng, vận tốc tương đối). So với XGBoost, Logistic Regression và persistence; kiểm tra trên seed giữ lại.
- **KẾT QUẢ:** Macro-F1 cao nhất ở cả hai nhãn và bốn mốc. Khoảng cách với XGBoost tăng theo mốc: QoS từ +0,65 điểm % (k=1) lên +6,10 điểm % (k=5), CI [5,60; 6,60].
- **HẠN CHẾ:** lỗi đoán liên kết hỏng thành ổn định của Edge-SAGE cao hơn XGBoost ở 7/8 tổ hợp.
- **TIẾP THEO:** Mới đo ngoại tuyến, chưa biết PDR thay đổi ra sao → đưa dự báo vào vòng kín ns-3.

### 6. Bước 1 – Cách xây GNN (Edge-SAGE) (~45 s) — [trái → phải]
- **ĐÃ LÀM:** Xây Edge-SAGE, biến thể GraphSAGE có đặc trưng cạnh.
- **CÁCH LÀM:**
  1. Đầu vào: đồ thị tại thời điểm t; nút là UAV (8 đặc trưng), cạnh là liên kết đang kết nối (7 đặc trưng: khoảng cách, RSSI, SNR, trễ, tỉ lệ mất gói, thông lượng, vận tốc tương đối).
  2. Hai lớp truyền thông điệp: mỗi thông điệp ghép biểu diễn nút lân cận với đặc trưng cạnh nối tới; ẩn 128 chiều, dropout 0,3.
  3. Bộ giải mã nhận [h_i ‖ h_j ‖ e_ij] và trả xác suất cạnh ổn định.
  4. Hai nhãn ở mốc k = 1, 2, 3, 5: QoS (còn kết nối, SNR ≥ 18 dB, mất gói ≤ 10%, trễ ≤ 10 ms) và survival (tồn tại liên tục từ t+1 đến t+k).
- **HUẤN LUYỆN:** binary cross-entropy có trọng số lớp, tối đa 100 epoch, dừng sớm; ngưỡng phân loại chọn trên tập xác thực.
- **ĐÁNH GIÁ:** held-seed; 210 lượt = 21 điều kiện × 10 seed; mỗi fold giữ 21 lượt của một seed để kiểm tra, 189 lượt còn lại để học; CI 95% từ 20.000 lần bootstrap theo seed.

### 7. Bước 1 – Vì sao chọn như vậy (~50 s) — [6 ô, trái → phải, trên → dưới]
- **GNN thay vì chỉ XGBoost:** chất lượng một liên kết phụ thuộc cả các UAV xung quanh, GNN nhìn được lân cận. XGBoost vẫn giữ làm đối chứng mạnh nhất.
- **GraphSAGE thay vì GCN:** GCN dùng tích chập phổ; GraphSAGE gom thông tin lân cận nên áp dụng được cho đồ thị chưa thấy khi huấn luyện, hợp với topology UAV đổi theo thời gian.
- **Đưa đặc trưng cạnh vào thông điệp thay vì GraphSAGE, GAT gốc:** hai mô hình gốc chỉ truyền đặc trưng nút, trong khi tín hiệu nằm ở RSSI, SNR, mất gói của cạnh. Giai đoạn đầu nhóm đã thử cả GraphSAGE và GAT; bỏ đặc trưng cạnh thì Macro-F1 giảm mạnh (báo cáo tiến độ 08/09, within-run: Edge-SAGE QoS k=1 từ 0,848 xuống 0,541). Edge-SAGE không phải kiến trúc hoàn toàn mới; gần với MPNN và edge-conditioned convolution.
- **Held-seed thay vì chia ngẫu nhiên:** các snapshot liền nhau tương quan mạnh; chia ngẫu nhiên làm điểm quá lạc quan.
- **Macro-F1 thay vì accuracy:** hai lớp ổn định / không ổn định lệch nhau; Macro-F1 lấy trung bình F1 hai lớp nên không bị lớp đông áp đảo.
- **4 mốc dự báo:** ở t+1 số đo hiện tại của chính cạnh đã mang phần lớn tín hiệu; giả thuyết là đồ thị có ích khi nhìn xa. Kết quả đúng: chênh với XGBoost tăng từ +0,65 (k=1) lên +6,10 điểm % (k=5).

> **NẾU THẦY HỎI:** Edge-SAGE vẫn còn hạn chế – lỗi đoán liên kết hỏng thành ổn định cao hơn XGBoost ở 7/8 tổ hợp.

### 8. Bước 2 – Dự báo + di chuyển relay, v2 (~40 s) — [trên rồi dưới]
- **ĐÃ LÀM:** Dự báo rủi ro suy giảm liên kết; khi rủi ro > 0,5 thì kéo relay về các điểm chia đều giữa nguồn và đích.
- **VÌ SAO:** Giữ liên kết bằng vị trí: kéo relay về trước khi liên kết đứt thì mạng không bị chia cắt.
- **CÁCH LÀM:** So với cùng luật nhưng không dự báo, 384 episode formal, ghép cặp theo kịch bản.
- **KẾT QUẢ:** Âm. PDR −0,99 điểm %, CI [−1,99; −0,02]; relay bay thêm 41,2 m.
- **TIẾP THEO:** Có thể do ngưỡng và thời điểm bay chưa tốt → v3 chọn ngưỡng trên tập tuning và chỉ cho bay khi có lợi.

### 9. Bước 3 – Ngưỡng + cổng giá trị, v3 (~35 s) — [1 → 2 → 3]
- **ĐÃ LÀM:** Chọn ngưỡng và tầm dự báo trên tập tuning; thêm mô hình dự đoán lợi ích của mỗi lần bay; chỉ cho bay khi lợi ích lớn hơn chi phí.
- **VÌ SAO:** v2 bay sai lúc, tốn quãng đường mà PDR không tăng.
- **CÁCH LÀM:** Tuning chọn tầm dự báo 20 giây; mô hình lợi ích học từ các đợt di chuyển có nhãn; kiểm ở validation trước khi chạy test.
- **KẾT QUẢ:** Dừng ở validation. Mô hình lợi ích dự đoán ngược: Spearman −0,192 trên 158 đợt; trên tập tuning mọi mức phạt đều cho tổng lợi ích thực tế âm. Không chạy test.
- **TIẾP THEO:** Hai vòng đều thất bại → phải tìm gói tin thật sự bị mất ở đâu.

### 10. Bước 4 – Chẩn đoán gói bị mất (~40 s) — [trái → phải]
- **ĐÃ LÀM:** Đọc lại 384 episode của v2 theo từng giây, phân loại theo tuyến OLSR đang dùng.
- **VÌ SAO:** v2, v3 đều thất bại; cần biết gói mất ở đâu trước khi làm tiếp.
- **KẾT QUẢ:** Khi tuyến ổn định hơn 5 giây, đi thẳng có PDR 0,900, qua relay 0,966. Phần lớn các giây đi thẳng đã có sẵn đường qua relay với liên kết yếu nhất ≥ 0,9. Cả 94 đợt di chuyển của v2 đều bắt đầu khi tuyến đã đi qua relay, nên di chuyển relay không chạm tới phần mất gói lớn nhất.
- **TIẾP THEO:** Bỏ hướng di chuyển relay, chuyển sang tự chọn tuyến theo chất lượng liên kết (v4).

### 11. Bước 5 – Ghim tuyến theo ETX, v4 (~45 s) — [đường đỏ rồi đường xanh]
- **ĐÃ LÀM:** Tự chọn tuyến theo chất lượng liên kết và ghim tuyến trong ns-3; OLSR vẫn chạy làm dự phòng.
- **VÌ SAO:** Gói mất vì OLSR đi thẳng qua liên kết yếu (đường đỏ) trong khi có đường qua relay tốt hơn (đường xanh).
- **CÁCH LÀM:** Mỗi giây ns-3 gửi tỉ lệ nhận probe từng liên kết sang bộ điều khiển Python qua stdin/stdout. Python tính ETX = 1/(q_ab·q_ba) và chọn đường có tổng ETX nhỏ nhất. Chỉ đổi tuyến khi đường mới tốt hơn rõ (margin) và giữ ít nhất 5 s (dwell). Đối chiếu log ns-3 để chắc gói đi đúng tuyến đã ghim.
- **KẾT QUẢ:** PDR +4,18 điểm % so với OLSR (HOLD), tăng ở 35/40 cặp kịch bản, 3/3 kiểm định đạt. Chỉ di chuyển relay để phục hồi thì +0,00 điểm %.
- **TIẾP THEO:** OLSR gốc là đối thủ yếu → so với OLSR-ETX (v5–v7).

### 12. Bước 6 – So với OLSR-ETX, v5–v7 (~45 s) — [đọc từ trên xuống]
- **ĐÃ LÀM:** So ghim tuyến với OLSR-ETX, bản OLSR có chọn tuyến theo chất lượng; tự cài OLSR-ETX cho ns-3.45.
- **VÌ SAO:** OLSR gốc là đối thủ yếu; phải chứng minh lợi ích không chỉ đến từ metric ETX.
- **CÁCH LÀM:** v5 so lần đầu; v6 sửa lỗi ARP của ns-3 rồi chạy lại; v7 kiểm tra độ bền: OLSR-ETX chỉnh HELLO 0,5 s, trễ điều khiển 2 s, chuỗi 7 UAV, có luồng nền.
- **KẾT QUẢ:** v5 +3,89 điểm %, CI [−1,11; 8,66] chứa 0 vì lỗi ARP làm nhiễu. v6 +4,57 [3,60; 5,55], thắng 39/40 kịch bản. v7 +5,06 [4,21; 5,91], thắng 39/40 kể cả khi trễ 2 s; chuỗi 7 UAV thắng 20/20.
- **CHI PHÍ:** ghim tuyến khoảng 5,55 kB/s so với 1,78 kB/s của OLSR-ETX HELLO 0,5 s, gấp khoảng 3 lần.
- **LƯU Ý:** điều khiển tập trung, telemetry chưa chiếm kênh nên đây là giới hạn trên.
- **TIẾP THEO:** Ghim tuyến đã tốt → thay EWMA bằng GNN dự báo trong vòng kín (v8).

### 13. Bước 7 – GNN dự báo trong vòng kín, v8 (~45 s) — [thanh xanh rồi thanh cam]
- **ĐÃ LÀM:** GNN hồi quy mới dự báo tỉ lệ nhận probe trung bình 5 giây tới của từng liên kết; đưa vào cùng luật ghim tuyến thay cho EWMA 0,5.
- **VÌ SAO:** Biết trước liên kết sắp yếu thì đổi tuyến sớm hơn.
- **CÁCH LÀM:** Huấn luyện trên 48 trace, chọn epoch trên 20 trace, chỉ dùng dữ liệu hiện tại và quá khứ; khóa mô hình bằng SHA-256 trước khi chạy. 80 cell validation, 192 cell test với seed mới; so cặp, bootstrap 10.000 lần.
- **KẾT QUẢ:** GNN hơn EWMA 0,5 +0,66 điểm %, CI [0,38; 1,00]; hơn MLP bỏ truyền thông điệp +0,38; hơn OLSR-ETX nhanh +5,75 (giới hạn trên vì điều khiển tập trung). 3/3 kiểm định đạt. MSE: GNN 0,00078, EWMA 0,00127. Số lệnh đổi tuyến: GNN 95, EWMA 359.
- **TIẾP THEO:** GNN đổi tuyến ít hơn hẳn → GNN hơn nhờ dự báo hay chỉ nhờ mượt hơn? Thử EWMA mượt hơn.

### 14. Bước 7 – Cách xây GNN hồi quy, v8 (~45 s) — [đọc theo hàng]
- **ĐÃ LÀM:** Xây GNN hồi quy mới cho vòng kín, khác mô hình phân loại giai đoạn 1.
- **VÌ SAO ĐỔI:** Bộ chọn tuyến tính ETX = 1/(q_ab·q_ba), cần một giá trị chất lượng q chứ không cần nhãn ổn định / không ổn định.
- **CÁCH LÀM:**
  - Nhãn: tỉ lệ nhận probe trung bình 5 giây tới, y = (1/5)·Σ q(t+h), h = 1..5.
  - 8 đặc trưng nút: tọa độ so với tâm đội hình, vận tốc, bậc hàng xóm, tỉ lệ nhận trung bình. 7 đặc trưng cạnh: khoảng cách, RSSI, tỉ lệ nhận hiện tại, EWMA 0,5, chênh tỉ lệ nhận so với giây trước, vận tốc tách xa, độ lớn vận tốc tương đối. Chỉ có cạnh truyền thông điệp khi tỉ lệ nhận hiện tại ≥ 0,1. Không đọc quỹ đạo tương lai.
  - Mô hình: 2 lớp tổng hợp trung bình kiểu GraphSAGE, 64 chiều, ReLU; bộ giải mã cặp nút + sigmoid, đối xứng hóa; 19.137 tham số.
  - Huấn luyện: MSE, Adam lr 0,001, batch 128, tối đa 60 epoch, patience 10; 48 trace HOLD của v7 để fit, 20 trace để chọn epoch; chọn epoch 58.
  - Triển khai: chuyển sang NumPy chạy trên CPU, sai số so với PyTorch tối đa 1,19×10⁻⁷; thời gian quyết định trung vị 4,92 ms.
  - Đối chứng MLP: bỏ hai nhánh truyền thông điệp, còn 13.505 tham số, để đo phần đóng góp của đồ thị.

> **NẾU THẦY HỎI:** MLP vẫn giữ bậc nút và ít tham số hơn, nên chưa tách hẳn tác động của truyền thông điệp khỏi dung lượng mô hình.

### 15. Bước 8 – Thử EWMA mượt hơn, v8x (~35 s) — [trái → phải]
- **ĐÃ LÀM:** Chạy lại EWMA với hệ số làm mượt 0,2 và 0,1 trên cùng dữ liệu test v8 (hệ số càng nhỏ càng mượt).
- **VÌ SAO:** Tách lợi ích của thông tin dự báo khỏi lợi ích của việc làm mượt.
- **KẾT QUẢ:** EWMA 0,2 có PDR ngang GNN. EWMA 0,1 hơn GNN +0,31 điểm %, CI [0,04; 0,61], dù MSE kém nhất (0,00165).
- **LƯU Ý:** Đây là phân tích sau khi xem kết quả test, không đăng ký trước, nên chỉ coi là gợi ý.
- **TIẾP THEO:** Cần một thí nghiệm đăng ký trước để so công bằng (v10). Song song thử dùng GNN cho bài toán di chuyển relay (v9).

### 16. Bước 9 – GNN kích hoạt relay, v9 (~40 s) — [đọc từ trên xuống]
- **ĐÃ LÀM:** Dùng GNN để kích hoạt relay sớm hơn; cùng định tuyến và cùng luật bố trí relay, chỉ khác điều kiện kích hoạt.
- **VÌ SAO:** Dự báo sớm thì kéo relay về sớm, giảm số giây mạng bị chia cắt.
- **CÁCH LÀM:** Luật phản ứng: liên kết yếu nhất trên đường tốt nhất dưới 0,7 trong 2 giây liên tiếp thì relay bay về các điểm chia đều, tối đa 8 m/s. GNN thêm điều kiện tương tự trên giá trị dự báo. 80 cell validation, 192 cell test.
- **KẾT QUẢ:** Không đạt. GNN chỉ bớt 0,20 s chia cắt, CI [−1,05; 1,45]. Relay đứng yên 123,43 s; luật phản ứng 31,45 s; GNN 31,25 s; relay đặt lý tưởng vẫn còn 26,09 s. Luật phản ứng đã đạt khoảng 94% mức giảm có thể, dự báo không còn chỗ cải thiện. Di chuyển tốn khoảng 637 m quãng đường relay mỗi lượt.
- **TIẾP THEO:** So công bằng GNN với EWMA khi cả hai cùng được chỉnh làm mượt (v10).

### 17. Bước 10 – GNN so với EWMA cùng chỉnh, v10 (~45 s) — [trái rồi phải]
- **ĐÃ LÀM:** Thí nghiệm đăng ký trước; cả GNN và EWMA cùng chọn hệ số làm mượt trên tập tuning riêng (đều chọn 0,1), rồi so trên 144 cell test mới.
- **VÌ SAO:** Ở v8 EWMA chưa được chỉnh, so sánh chưa công bằng.
- **KẾT QUẢ:** MSE GNN 0,00123, EWMA 0,1 là 0,00166 → GNN đoán sát hơn. PDR 96,07% so với 95,98%, chênh +0,08 điểm %, CI [−0,11; 0,29] chứa 0 → chưa có khác biệt. Không khẳng định là tương đương.
- **GIẢI THÍCH:** Bộ ghim tuyến cần đầu vào ít nhiễu để không đổi tuyến liên tục. GNN làm mượt như tác dụng phụ; lợi ích đến từ làm mượt, không từ thông tin tương lai.
- **TIẾP THEO:** Kết luận.

---

## Kết luận

### 18. Kết luận và việc tiếp theo (~40 s)
- Chọn tuyến theo chất lượng liên kết (ETX + ghim tuyến) là phần mang lại lợi ích: +4,57 đến +5,06 điểm % PDR so với OLSR-ETX, đổi lại tốn báo hiệu gấp khoảng 3 lần.
- GNN dự báo sát hơn (MSE thấp hơn) nhưng PDR ngang EWMA đã chỉnh (v10).
- Dự báo không giúp kích hoạt relay: luật phản ứng đã gần giới hạn hình học (v9).
- Bài học: MSE thấp hơn không suy ra PDR cao hơn, phải đánh giá trong vòng kín.

**Giới hạn:** một họ topology chuỗi relay; điều khiển tập trung, telemetry chưa chiếm kênh; vòng kín dùng một seed huấn luyện; chưa bay thật.

**Việc tiếp theo** (nếu thầy hỏi): GNN theo thời gian, topology phi chuỗi, mô phỏng telemetry chiếm kênh, thử nghiệm bay thật.

---

Nguồn số liệu: `reports/nckh_final/{main,methods_v8_v9,results_v10}.tex`, `reports/nckh_paper/{main,methods,results,posthoc}.tex`, `docs/uav_system_workflow.tex`, `docs/bao_cao_tien_do_2026-09-08.md` (ablation đặc trưng cạnh).
