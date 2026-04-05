# Tài liệu mô tả graph dataset

## 1. Mục tiêu

Mỗi snapshot mạng tại thời điểm `t` được biểu diễn thành một đồ thị `G_t = (V_t, E_t)`, trong đó:

- `V_t` là tập UAV đang hoạt động tại thời điểm `t`
- `E_t` là tập liên kết vô tuyến đang tồn tại tại thời điểm `t`

Snapshot được lưu từ `nodes.csv` và `edges.csv`, sau đó ánh xạ sang train/validation/test và sang biểu diễn đồ thị dùng cho PyTorch Geometric.

## 2. Dữ liệu đầu vào

### 2.1 Bảng nút `nodes_features.csv`

Mỗi dòng tương ứng với một UAV tại một thời điểm `t`.

Các cột:

- `time`: chỉ số snapshot
- `node_id`: định danh UAV trong snapshot
- `x, y, z`: vị trí không gian 3D
- `vx, vy, vz`: vận tốc theo trục
- `speed`: độ lớn vận tốc
- `degree`: số láng giềng kết nối trực tiếp
- `load`: tải của nút

### 2.2 Cách xác định `load`

Dữ liệu thô hiện tại của simulator chưa có cột `load` ở mức nút. Để vẫn hoàn tất bộ dataset đúng cấu trúc yêu cầu, pipeline này sinh `load` theo công thức proxy:

`load = degree / (num_nodes_in_snapshot - 1)`

Giá trị này phản ánh mức độ bận kết nối tương đối của UAV trong snapshot hiện tại. Đây là topology load proxy, không phải traffic load thật. Khi simulator có queue length, relay count hoặc traffic per node, cột `load` nên được thay bằng chỉ số đó.

## 3. Node features

Bộ node features dùng để build tensor `x`:

- `x`
- `y`
- `z`
- `vx`
- `vy`
- `vz`
- `degree`
- `load`

## 4. Edge features

Bộ edge features dùng để build tensor `edge_attr`:

- `distance`
- `rssi`
- `delay`
- `packet_loss`
- `relative_speed`

Trong dữ liệu thô của simulator còn có `snr`, `throughput`, `p_stable`, `weight`. Các cột này vẫn được giữ ở bảng trung gian để phân tích hoặc làm baseline, nhưng không đưa vào bộ edge features chuẩn của dataset v1.

## 5. Quy tắc gán nhãn liên kết

Pipeline chỉ lấy các cạnh đang tồn tại ở thời điểm `t`, tức là các dòng có `connected == 1`, làm tập cạnh giám sát của snapshot `G_t`.

Nhãn tại thời điểm `t` được xác định theo trạng thái của chính liên kết đó ở thời điểm `t+1`:

- `label = 1` (`stable`) nếu ở `t+1` liên kết vẫn tồn tại và đồng thời thỏa các ngưỡng chất lượng
- `label = 0` (`at_risk`) nếu liên kết biến mất hoặc chất lượng không còn đạt ngưỡng

Quy tắc cụ thể:

- `connected(t+1) == 1`
- `snr(t+1) >= tau_snr`
- `packet_loss(t+1) <= tau_loss`
- `delay(t+1) <= tau_delay`

Nếu một trong các điều kiện trên không đạt, nhãn được gán về lớp `0`.

Cấu hình mặc định:

- `tau_snr = 18.0`
- `tau_loss = 0.10`
- `tau_delay = 10.0`

## 6. Chia train / validation / test

Tập dữ liệu được chia theo thời gian, không xáo trộn ngẫu nhiên theo hàng.

Mặc định:

- `train = 70%` số snapshot có nhãn
- `val = 15%`
- `test = 15%`

Snapshot cuối cùng không được dùng vì không có `t+1` để xác định nhãn.

## 7. Biểu diễn graph output

Mỗi graph được lưu thành một record gồm:

- `x`: tensor node features, shape `[num_nodes, num_node_features]`
- `edge_index`: tensor adjacency hai chiều cho message passing, shape `[2, 2*num_edges]`
- `edge_attr`: tensor edge features hai chiều, shape `[2*num_edges, num_edge_features]`
- `edge_label_index`: tensor chỉ số các cạnh gốc dùng để học nhãn, shape `[2, num_edges]`
- `edge_label`: tensor nhãn 0/1 cho từng cạnh gốc, shape `[num_edges]`
- `time`, `split`, `node_ids`

## 8. Output cuối cùng

Pipeline sinh ra:

- `processed/nodes_features.csv`
- `processed/edges_features.csv`
- `processed/edges_labeled.csv`
- `splits/time_splits.csv`
- `graph/train.pt`
- `graph/val.pt`
- `graph/test.pt`
- `graph/dataset_summary.json`
