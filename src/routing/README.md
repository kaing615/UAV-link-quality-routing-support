# Routing Support (Nội dung 6+7)

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

- `hop` — Dijkstra trọng số 1 (shortest hop). **Lưu ý**: đây chính là logic
  chọn tuyến của OLSR (link-state, metric hop-count) nhưng trên topology
  ground-truth tức thời — tức một *chặn trên* của hiệu năng OLSR thật (vốn
  chịu trễ hội tụ HELLO/TC). Vượt baseline này nghĩa là vượt OLSR.
- `delay` — Dijkstra theo delay đo tại t.
- `xgb` — `w = 1 − ŷ` từ XGBoost baseline.
- `gnn` — `w = 1 − ŷ` từ GNN (mặc định edge-sage).
- `olsr` — **OLSR thật**: tuyến `ns3::olsr` đã chọn, đọc từ `route_path`
  trong `traffic_log.csv` (mỗi run chỉ ghi một cặp src–dst). Vì vậy ngoài
  summary all-pairs còn có `summary_olsr_pair.csv`: mọi chiến lược lọc về
  đúng cặp đó (cùng tập session) để so sánh công bằng — chart
  `routing_comparison_olsr_pair.png`.

Hai lưu ý khi đọc kết quả `olsr`: (1) `route_changes` của nó đếm số lần
*giao thức tự đổi tuyến*, khác ngữ nghĩa với các chiến lược khác (đếm số lần
*buộc phải* tính lại khi tuyến đứt) — OLSR giữ nguyên tuyến đã chết nên con
số thấp không có nghĩa là ổn định hơn; (2) `e2e_delay_ms` của olsr chỉ tính
được khi mọi link trên tuyến còn tồn tại trong snapshot (tuyến cũ → NaN),
nên trung bình delay của nó bị lệch về các trường hợp thuận lợi.

## Các module

- `predict_edges.py` — inference GNN từ `best_model.pt` + `test.pt`, xuất
  `ŷ` theo `(time, src, dst)` (training không lưu định danh cạnh trong
  predictions).
- `replay_eval.py` — chạy protocol trên một run; `--p-th` nhận danh sách
  (`0.3,0.5,0.7`) cho sweep; `--strict` tắt fallback khi filter làm mất
  đường (khi đó `route_found_rate` đo đúng chi phí liên thông).
- `aggregate_routing.py` — gộp `outputs/routing/*/summary.csv` và vẽ
  biểu đồ so sánh chiến lược 4 panel.
- `plot_pth_sweep.py` — gộp mọi `summary_pth*.csv` và vẽ đường cong
  trade-off an toàn tuyến ↔ duy trì liên thông theo `p_th`.

## Lệnh dùng

```bash
# Toàn bộ pipeline cho mọi run (predict + replay + aggregate + plot)
./scripts/routing/run_routing_for_runs.sh 'ns3big_*' edge-sage

# Sweep p_th (strict mode) rồi vẽ trade-off
python3 -m src.routing.replay_eval --run-name <RUN> --p-th 0.3,0.5,0.7 --strict
python3 -m src.routing.plot_pth_sweep
```

## Output

```text
outputs/routing/<RUN>/edge_predictions_<MODEL>.csv
outputs/routing/<RUN>/{summary,details}.csv            # p_th = 0
outputs/routing/<RUN>/{summary,details}_pth<P>.csv     # các mức p_th khác
outputs/aggregates/routing/summary_by_strategy.csv
outputs/aggregates/routing/routing_comparison.png
outputs/aggregates/routing/{pth_sweep.csv, pth_tradeoff.png}
```

## Cách đọc kết quả p_th

Tăng `p_th` tạo ra hai hiệu ứng đồng thời: tuyến chỉ đi qua link tốt nên bền
hơn thật, **và** các cặp khó bị loại khỏi mẫu đo (`route_found_rate` giảm) nên
trung bình phần còn lại tự đẹp lên. Bốn panel của `pth_tradeoff.png` phải đọc
cùng nhau — panel Route Found Rate là cái giá của ba panel còn lại.
