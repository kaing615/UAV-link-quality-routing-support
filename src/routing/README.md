# Routing Support Module (Nội dung 6+7)

Tích hợp đầu ra của mô hình dự đoán link quality vào lựa chọn tuyến, và đánh
giá tác động ở mức hiệu năng mạng bằng **replay** trên các test snapshot đã
ghi — không cần mô phỏng lại trong ns-3.

## Ý tưởng

Xác suất ổn định `ŷ` của từng link (từ GNN hoặc baseline) được ánh xạ thành
trọng số cạnh `w = 1 − ŷ + ε`, đưa vào Dijkstra để ưu tiên các link được dự
đoán là bền. Tùy chọn ngưỡng `p_th` loại hẳn các link có `ŷ < p_th` khỏi tập
ứng viên (thiết kế hệ thống §12, §13.5).

## Protocol đánh giá

Với mỗi snapshot test `t` và mỗi cặp `(src, dst)` liên thông, mỗi chiến lược
chọn một tuyến trên topology quan sát tại `t`, sau đó "tua" dữ liệu thật tại
`t+1..t+H` (mặc định H=5) để đo:

| Metric | Ý nghĩa |
|---|---|
| `route_lifetime` | số bước liên tiếp tuyến còn "sống" |
| `survival_at_1` | tuyến còn sống tại t+1 hay không |
| `realized_pdr_t1` | tích `(1 − packet_loss)` dọc tuyến tại t+1 (0 nếu đứt) |
| `route_changes` | số lần phải tính lại tuyến trong horizon |
| `e2e_delay_ms`, `hops`, `est_pdr` | thuộc tính tuyến tại thời điểm chọn |

Một link được coi là "sống" tại `t` khi connected **và** đạt cùng bộ ngưỡng
chất lượng dùng để gán nhãn (SNR > τ_snr, loss < τ_loss, delay < τ_delay) —
nhờ đó "tuyến sống" nhất quán với định nghĩa nhãn lớp 1.

## Các chiến lược so sánh

| Strategy | Mô tả |
|----------|-------|
| `hop` | Dijkstra trọng số 1 (shortest hop) — chặn trên của OLSR thật (topology ground-truth tức thời, không có HELLO/TC delay) |
| `delay` | Dijkstra theo delay đo tại t |
| `xgb` | `w = 1 − ŷ` từ XGBoost baseline |
| `gnn` | `w = 1 − ŷ` từ GNN (mặc định edge-sage) |
| `olsr` | **OLSR thật**: tuyến `ns3::olsr` đã chọn, đọc từ `route_path` trong `traffic_log.csv` (1 cặp src–dst mỗi run) |

## Kết quả chính (100 runs, ns3big dataset)

### Routing vs Shortest-Hop Baseline

| Metric | Shortest-Hop | XGBoost | GNN (Edge-SAGE) | Improvement |
|--------|-------------|---------|-----------------|-------------|
| Route Lifetime | 0.35 | **0.50** | **0.50** | **+43%** |
| Realized PDR @ t+1 | 0.10 | 0.15 | 0.14 | **+40%** |
| Mean Hops | 2.48 | 3.40 | 3.51 | +0.9 hop |
| E2E Delay (ms) | 2.13 | 2.92 | 3.01 | +0.8 ms |
| Route Changes | 3.44 | 3.22 | 3.23 | -6% |

**Insight**: Prediction-assisted routing tăng route lifetime 43%, PDR 40% với chi phí
~1 hop overhead (~1ms delay). GNN và XGBoost tương đương — giá trị nằm ở tín hiệu
dự đoán, không phân thắng kiến trúc.

### OLSR thật vs Prediction-Assisted

| Metric | OLSR (ns-3) | GNN-Assisted | Improvement |
|--------|-------------|--------------|-------------|
| Route Found Rate | **0.66** | 0.99 | +50% |
| Route Lifetime | 0.20 | **0.50** | +150% |
| Realized PDR @ t+1 | 0.06 | **0.14** | +133% |

**Key Finding**: OLSR thật có `route_found_rate=0.66` (thấp!) vì HELLO/TC delay (2s/5s)
làm topology OLSR "stale" so với ground-truth. Prediction-assisted routing vượt trội
cả OLSR thật lẫn shortest-hop baseline.

> **Lưu ý semantic**: `route_changes` của OLSR đếm số lần *giao thức tự đổi tuyến*,
> khác ngữ nghĩa với các chiến lược khác (đếm số lần *buộc phải* tính lại khi tuyến đứt).
> OLSR giữ nguyên tuyến đã chết nên con số thấp không có nghĩa ổn định hơn.

## Threshold Sweep (p_th)

Tăng `p_th` loại bỏ các link có `ŷ < p_th` khỏi tập ứng viên:

| p_th | Route Found Rate | Lifetime | PDR @ t+1 | Route Changes |
|------|-----------------|----------|-----------|---------------|
| 0.0 | 100% | 0.50 | 0.14 | 3.23 |
| 0.3 | 100% | 0.50 | 0.14 | 1.24 |
| 0.5 | 100% | 0.50 | 0.14 | 1.07 |
| 0.7 | 100% | 0.50 | 0.14 | 0.99 |
| 0.9 | 16.5% | 2.82 | 0.77 | 0.71 |

**Sweet spot**: `p_th ≈ 0.3` — giảm route changes 60% mà vẫn giữ route found rate 100%.

> **Survivorship bias**: 4 panel của `pth_tradeoff.png` phải đọc cùng nhau — panel
> Route Found Rate là cái giá của ba panel còn lại.

## Horizon Sweep (TODO)

**Research question**: Prediction-assisted routing có cải thiện hơn ở horizon xa hơn
(k=3,5,10) không? GNN có thể vượt tabular ở t+k>1 vì thấy được topology mới /
hàng xóm mới — nơi edge features cục bộ kém thông tin hơn.

**Planned experiment**:
```
H = 1, 3, 5, 10
Metrics: lifetime, pdr, changes theo H
Compare: hop vs xgb vs gnn
```

## Các module

- `predict_edges.py` — inference GNN từ `best_model.pt` + `test.pt`, xuất
  `ŷ` theo `(time, src, dst)` (training không lưu định danh cạnh trong predictions).
- `replay_eval.py` — chạy protocol trên một run; `--p-th` nhận danh sách
  (`0.3,0.5,0.7`) cho sweep; `--strict` tắt fallback khi filter làm mất đường;
  `--horizon` để thay đổi H (default 5).
- `aggregate_routing.py` — gộp `outputs/routing/*/summary.csv` và vẽ
  biểu đồ so sánh chiến lược 4 panel.
- `plot_pth_sweep.py` — gộp mọi `summary_pth*.csv` và vẽ đường cong
  trade-off an toàn tuyến ↔ duy trì liên thông theo `p_th`.

## Lệnh dùng

```bash
# Predict + Replay + Aggregate cho tất cả runs (p_th=0, H=5)
./scripts/routing/run_routing_for_runs.sh 'ns3big_*' edge-sage

# Sweep p_th với strict mode
python3 -m src.routing.replay_eval --run-name <RUN> \
    --p-th 0.0,0.1,0.2,0.3,0.5,0.7,0.9 --strict
python3 -m src.routing.plot_pth_sweep

# Sweep horizon (TODO)
./scripts/routing/sweep_horizon.sh 'ns3big_*' edge-sage 1,3,5,10
```

## Output

```
outputs/routing/<RUN>/edge_predictions_<MODEL>.csv
outputs/routing/<RUN>/{summary,details}.csv            # p_th = 0
outputs/routing/<RUN>/{summary,details}_pth<P>.csv     # các mức p_th khác
outputs/routing/<RUN>/summary_olsr_pair.csv            # OLSR same-pair comparison
outputs/aggregates/routing/summary_by_strategy.csv
outputs/aggregates/routing/routing_comparison.png
outputs/aggregates/routing/routing_comparison_olsr_pair.png
outputs/aggregates/routing/{pth_sweep.csv, pth_tradeoff.png}
outputs/aggregates/routing/{horizon_sweep.csv, horizon_tradeoff.png}  # TODO
```

## Cách đọc kết quả

1. **routing_comparison.png**: So sánh all-pairs, tất cả strategy trên cùng metric
2. **routing_comparison_olsr_pair.png**: Cùng cặp src-dst như OLSR ghi → so sánh công bằng
3. **pth_tradeoff.png**: Trade-off giữa an toàn tuyến vs liên thông theo p_th
4. **horizon_tradeoff.png** (TODO): Performance theo horizon H → xem prediction giá trị ra sao ở t+k>1
