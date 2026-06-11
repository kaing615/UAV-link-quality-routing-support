# ns-3 Dataset Generator

Sinh dataset UAV link-quality bằng **ns-3** (stack 802.11 + OLSR thật) thay cho
simulator Python công thức hóa (`simulation/main.py`). Output giữ nguyên schema
(`nodes.csv`, `edges.csv`, `traffic_log.csv`, `scenario.json`) nên toàn bộ
pipeline phía sau (preprocessing → graph dataset → training) dùng lại không đổi.

## Khác biệt so với simulator Python

| Đại lượng | Python sim | ns-3 |
|---|---|---|
| RSSI | công thức log-distance | **sniff từ PHY** (kèm Nakagami fading) |
| Delay | công thức (base + propagation + penalty) | **đo thật** từ probe UDP một-hop |
| Packet loss | bảng tra theo SNR | **đo thật**: 1 − số probe nhận/kỳ vọng |
| Connected | cắt theo khoảng cách (`comm_range`) | tỷ lệ nhận probe ≥ 0.5 trong cửa sổ 1s |
| Topology/định tuyến | OLSR tự cài đặt | **ns3::olsr** chuẩn |
| Throughput, p_stable | công thức | cùng công thức nhưng áp lên giá trị đo |

`comm_range` vẫn được giữ làm tham số: nó quy đổi thành `RxSensitivity` của PHY
để mật độ kết nối tương đương kịch bản Python, nhưng trên đó còn có fading,
collision, contention thật. Fading Nakagami (m giảm dần theo khoảng cách) tạo
vùng suy giảm tự nhiên ở rìa vùng phủ — không có nó, mọi gói nhận được đều có
SNR trên ngưỡng nhãn và dataset trở thành degenerate (~98% stable).

## Build

```bash
brew install ns-3        # macOS; cung cấp ns-3 3.48
cmake -B simulation/ns3/build -S simulation/ns3 -DCMAKE_BUILD_TYPE=Release
cmake --build simulation/ns3/build
```

(Script `run_one_dataset_ns3.sh` tự build ở lần chạy đầu.)

## Chạy

```bash
# Một dataset (sim + preprocessing + standardize + imbalance)
./scripts/dataset/run_one_dataset_ns3.sh <RUN_NAME> [SEED] [MOBILITY]
SIM_NUM_UAVS=8 SIM_COMM_RANGE=250 ./scripts/dataset/run_one_dataset_ns3.sh ns3_dense_42 42 gauss-markov

# Batch ngẫu nhiên
./scripts/dataset/run_many_random_datasets_ns3.sh 10 ns3exp01
```

Chạy binary trực tiếp (xem đủ tham số với `--help`):

```bash
./simulation/ns3/build/uav-olsr-dataset \
  --numUavs=6 --timeSteps=145 --seed=42 --mobility=gauss-markov \
  --commRange=243 --outputDir=data/raw_snapshots/my_run
```

## Cách đo

- Mỗi node broadcast 20 probe UDP/giây (payload chứa timestamp + id nguồn)
- Mỗi giây chụp một snapshot: gom thống kê probe + RSSI sniff của cửa sổ 1s
- 10 giây warm-up trước snapshot đầu để OLSR hội tụ
- `traffic_log.csv`: walk bảng định tuyến OLSR từ source → dest mỗi giây
  (`olsr_mpr_nodes` luôn 0 — ns-3 không expose MPR set công khai)
