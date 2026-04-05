# UAV Link Quality Routing Support

Dự án này tập trung vào bài toán **dự đoán chất lượng liên kết** và **hỗ trợ lựa chọn tuyến** trong mạng UAV động thông qua **Graph Neural Network (GNN)**. Ý tưởng cốt lõi là khai thác đồng thời **cấu trúc topology của toàn mạng**, **đặc trưng của nút** và **đặc trưng của liên kết** để dự đoán trước trạng thái liên kết, từ đó cung cấp tín hiệu hỗ trợ cho quá trình định tuyến trong môi trường UAV có topology thay đổi nhanh.

**Tên đề tài:** Ứng dụng Graph Neural Network trong dự đoán chất lượng liên kết và hỗ trợ định tuyến trong mạng UAV  
**Tên tiếng Anh:** Application of Graph Neural Networks for Link Quality Prediction and Routing Support in UAV Networks  
**Trường:** Trường Đại học Công nghệ Thông tin - Đại học Quốc gia Thành phố Hồ Chí Minh  
**Khoa:** Khoa Mạng máy tính và Truyền thông  
**Giảng viên hướng dẫn:** ThS. Đặng Lê Bảo Chương  
**Sinh viên thực hiện:** Nguyễn Đình Tâm - 23521389, Võ Công Vinh - 23521800  

---

## Giới thiệu

Trong mạng UAV ad hoc, các thiết bị bay hoạt động trong **không gian ba chiều**, liên tục di chuyển với vận tốc thay đổi, khiến **topology mạng biến động mạnh theo thời gian**. Điều này làm cho các liên kết vô tuyến dễ suy giảm, mất ổn định hoặc đứt kết nối, từ đó ảnh hưởng trực tiếp đến hiệu năng truyền dữ liệu đầu-cuối.

Thay vì chỉ dựa vào các thông tin tức thời như **số hop**, **vị trí** hoặc **khoảng cách**, đề tài hướng tới việc xây dựng một mô-đun học máy có khả năng học được **mối quan hệ không gian - thời gian của mạng UAV** thông qua biểu diễn đồ thị. Trên cơ sở đó, mô hình sẽ dự đoán chất lượng hoặc trạng thái của liên kết trong thời điểm kế tiếp, sau đó ánh xạ kết quả dự đoán thành tín hiệu hỗ trợ cho **route selection**.

Nói cách khác, đề tài **không nhằm thiết kế một giao thức định tuyến hoàn toàn mới**, mà tập trung vào việc phát triển một **mô-đun dự đoán chất lượng liên kết** có thể tích hợp vào quá trình lựa chọn tuyến nhằm tăng độ ổn định của đường truyền trong mạng UAV động.

---

## Tính năng / Định hướng chính (Core Objectives)

1. **Mô phỏng mạng UAV động trong không gian 3D:**  
   Xây dựng môi trường mô phỏng các UAV di chuyển trong không gian ba chiều với topology thay đổi theo thời gian.

2. **Thu thập snapshot topology theo time step:**  
   Ghi nhận trạng thái mạng tại từng thời điểm hoặc từng cửa sổ quan sát để phục vụ xây dựng dữ liệu học máy.

3. **Biểu diễn mạng dưới dạng đồ thị:**  
   Mô hình hóa mạng UAV thành graph, trong đó UAV là **node** và liên kết vô tuyến là **edge**.

4. **Xây dựng bộ đặc trưng cho node và edge:**  
   Khai thác các đặc trưng như vị trí, vận tốc, degree của node; khoảng cách, RSSI, SNR, delay, packet loss, relative speed của edge.

5. **Gán nhãn trạng thái liên kết:**  
   Xác định trạng thái liên kết như **ổn định / suy giảm** hoặc các mức chất lượng tương ứng để phục vụ huấn luyện mô hình.

6. **Huấn luyện mô hình GNN cho bài toán dự đoán liên kết:**  
   Sử dụng **GraphSAGE** làm mô hình chính để học biểu diễn đồ thị và dự đoán trạng thái liên kết ở mức cạnh.

7. **So sánh với các baseline và mô hình đối chiếu:**  
   Đối chiếu hiệu quả với các phương pháp như **Logistic Regression**, **Random Forest**, **MLP**, threshold-based methods và **GAT**.

8. **Hỗ trợ định tuyến dựa trên đầu ra dự đoán:**  
   Ánh xạ xác suất hoặc nhãn dự đoán thành **trọng số cạnh** hoặc **tiêu chí ưu tiên**, từ đó hỗ trợ lựa chọn tuyến ổn định hơn.

9. **Đánh giá toàn hệ thống ở hai mức:**  
   - **Mức mô hình học máy:** Accuracy, Precision, Recall, F1-score, ROC-AUC  
   - **Mức hiệu năng mạng:** Packet Delivery Ratio, End-to-End Delay, Route Stability, Throughput

---

## Ý tưởng cốt lõi của đề tài

Pipeline tổng thể của hệ thống gồm các bước sau:

1. **Mô phỏng mạng UAV động** trong không gian 3D  
2. **Thu thập snapshot topology** theo từng time step  
3. **Xây dựng graph dataset** từ topology và đặc trưng liên kết  
4. **Huấn luyện mô hình GNN** để dự đoán trạng thái liên kết tại thời điểm kế tiếp  
5. **Ánh xạ đầu ra dự đoán** thành trọng số cạnh hoặc tiêu chí ưu tiên  
6. **Hỗ trợ lựa chọn tuyến** ổn định hơn trong mạng UAV động  

Tóm lại, hệ thống đóng vai trò như một **routing support module**, trong đó mô hình GNN cung cấp thông tin dự đoán về chất lượng liên kết để hỗ trợ tầng định tuyến ra quyết định tốt hơn trong môi trường có tính động cao.

---

## Công nghệ sử dụng (Technology Stack)

| Thành phần | Công nghệ |
|-----------|-----------|
| **Ngôn ngữ chính** | Python |
| **Mô phỏng mạng UAV** | Python-based UAV simulator |
| **Xử lý dữ liệu** | NumPy, Pandas |
| **Trực quan hóa** | Matplotlib |
| **Biểu diễn và huấn luyện đồ thị** | PyTorch Geometric |
| **Baseline Machine Learning** | Logistic Regression, Random Forest, MLP |
| **Phương pháp heuristic** | Threshold-based methods |
| **Mô hình chính** | GraphSAGE |
| **Mô hình đối chiếu** | GAT |

---

## Đặc trưng dữ liệu đầu vào

Hệ thống khai thác đồng thời thông tin ở hai mức:

### 1. Đặc trưng nút (Node Features)
- Vị trí 3D của UAV
- Vận tốc / hướng di chuyển
- Degree của nút
- Thông tin lân cận cục bộ

### 2. Đặc trưng cạnh (Edge Features)
- Khoảng cách giữa hai UAV
- RSSI
- SNR
- Delay
- Packet loss
- Relative speed giữa hai UAV

Việc kết hợp cả **node features**, **edge features** và **graph topology** giúp mô hình không chỉ nhìn thấy trạng thái cục bộ của một liên kết, mà còn hiểu được **ngữ cảnh toàn cục của mạng**.

---

## Luồng hoạt động tổng thể (Overall Workflow)

1. **Giai đoạn mô phỏng:**  
   Hệ thống tạo ra các kịch bản UAV di chuyển trong không gian 3D và hình thành topology mạng tương ứng theo thời gian.

2. **Giai đoạn thu thập dữ liệu:**  
   Tại mỗi time step, hệ thống ghi nhận thông tin node, edge và trạng thái liên kết để xây dựng graph snapshot.

3. **Giai đoạn xây dựng dataset:**  
   Các snapshot được tiền xử lý, trích xuất đặc trưng và gán nhãn trạng thái liên kết để tạo thành bộ dữ liệu phục vụ huấn luyện.

4. **Giai đoạn huấn luyện mô hình:**  
   Mô hình GNN học biểu diễn đồ thị và dự đoán trạng thái hoặc chất lượng liên kết trong thời điểm kế tiếp.

5. **Giai đoạn hỗ trợ định tuyến:**  
   Kết quả dự đoán được ánh xạ thành điểm số ưu tiên hoặc trọng số cạnh, hỗ trợ chọn tuyến có độ ổn định cao hơn.

6. **Giai đoạn đánh giá:**  
   Hệ thống được đánh giá cả về hiệu quả dự đoán của mô hình và tác động thực tế đến hiệu năng mạng.

---

## Cấu trúc dự án đề xuất (Project Structure)

```text
uav-link-quality-routing
├── data/                     # Dữ liệu thô, dữ liệu đã xử lý, graph snapshots
├── simulation/               # Môi trường mô phỏng UAV và sinh topology
├── preprocessing/            # Tiền xử lý dữ liệu, trích xuất node/edge features
├── models/                   # Cài đặt GraphSAGE, GAT và các baseline
├── training/                 # Script huấn luyện, validation, testing
├── routing/                  # Logic ánh xạ kết quả dự đoán sang hỗ trợ định tuyến
├── evaluation/               # Đánh giá mô hình và hiệu năng mạng
├── utils/                    # Hàm tiện ích
├── configs/                  # File cấu hình tham số mô phỏng và huấn luyện
├── outputs/                  # Kết quả chạy thử, logs, figures, checkpoints
└── README.md
