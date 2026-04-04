# UAV Link Quality Routing Support

> Đồ án chuyên ngành về **ứng dụng Graph Neural Network (GNN)** trong **dự đoán chất lượng liên kết** và **hỗ trợ định tuyến** trong mạng UAV động.

---

## Thông tin đồ án

- **Tên đề tài:**
  **Ứng dụng Graph Neural Network trong dự đoán chất lượng liên kết và hỗ trợ định tuyến trong mạng UAV**
  _Application of Graph Neural Networks for Link Quality Prediction and Routing Support in UAV Networks_

- **Trường:**
  **Trường Đại học Công nghệ Thông tin**
  **Đại học Quốc gia Thành phố Hồ Chí Minh**

- **Khoa:**
  **Khoa Mạng máy tính và Truyền thông**

- **Giảng viên hướng dẫn:**
  **ThS. Đặng Lê Bảo Chương**

- **Sinh viên thực hiện:**
  - **Nguyễn Đình Tâm**
  - **Võ Công Vinh**

---

## Giới thiệu

Repository này phục vụ cho đồ án chuyên ngành nghiên cứu bài toán **dự đoán chất lượng liên kết** và **hỗ trợ định tuyến** trong mạng UAV ad hoc động.

Trong mạng UAV, các thiết bị bay hoạt động trong **không gian ba chiều**, topology thay đổi liên tục theo thời gian, khiến các liên kết vô tuyến dễ suy giảm và ảnh hưởng trực tiếp đến khả năng truyền dữ liệu đầu-cuối. Thay vì chỉ dựa vào thông tin tức thời như số hop, vị trí hoặc khoảng cách, đề tài hướng tới việc khai thác đồng thời:

- **Topology của toàn mạng**
- **Đặc trưng nút** như vị trí, vận tốc, degree
- **Đặc trưng cạnh** như khoảng cách, RSSI, SNR, delay, packet loss, relative speed

Trên cơ sở đó, hệ thống sẽ sử dụng **Graph Neural Network**, trong đó **GraphSAGE** là mô hình chính, để dự đoán trạng thái liên kết ở mức cạnh và dùng đầu ra dự đoán như một tín hiệu **hỗ trợ lựa chọn tuyến ổn định hơn**.

---

## Mục tiêu chính

- Xây dựng môi trường mô phỏng mạng UAV động trong không gian 3D
- Thu thập snapshot topology theo thời gian
- Biểu diễn mạng UAV dưới dạng đồ thị
- Xây dựng bộ đặc trưng cho nút và cạnh
- Gán nhãn trạng thái liên kết (ổn định / suy giảm)
- Huấn luyện mô hình GNN cho bài toán dự đoán chất lượng liên kết
- Tích hợp đầu ra dự đoán vào cơ chế hỗ trợ định tuyến
- Đánh giá hệ thống ở cả mức mô hình học máy và mức hiệu năng mạng

---

## Ý tưởng cốt lõi của đề tài

Pipeline tổng thể của đồ án gồm các bước:

1. **Mô phỏng mạng UAV động** trong không gian 3D
2. **Thu thập snapshot** theo từng time step
3. **Xây dựng graph dataset** từ topology và các đặc trưng liên kết
4. **Huấn luyện mô hình GNN** để dự đoán trạng thái liên kết tại thời điểm kế tiếp
5. **Ánh xạ đầu ra dự đoán** sang trọng số cạnh hoặc tiêu chí ưu tiên
6. **Hỗ trợ lựa chọn tuyến** ổn định hơn trong mạng UAV động

Nói ngắn gọn, đề tài không nhằm xây dựng một giao thức định tuyến hoàn chỉnh mới từ đầu, mà tập trung vào việc phát triển một **mô-đun dự đoán chất lượng liên kết** để hỗ trợ quá trình route selection.

---

## 🏗️ Công nghệ và hướng triển khai

- **Ngôn ngữ chính:** Python
- **Mô phỏng:** Python-based UAV simulator
- **Xử lý dữ liệu:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Biểu diễn đồ thị:** PyTorch Geometric
- **Baseline ML:** Logistic Regression, Random Forest, MLP, threshold-based methods
- **Mô hình chính:** GraphSAGE
- **Mô hình đối chiếu:** GAT

---
