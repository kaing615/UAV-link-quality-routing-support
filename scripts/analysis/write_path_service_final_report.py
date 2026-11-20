"""Write the final Vietnamese narrative from retained formal-v2 analyses.

No collection, fitting, parameter selection, or statistical recomputation.
Creates fresh report/annex/provenance outputs; refuses existing destinations.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'reports/uav_path_service_study_2026-11-11'
SPEC = ROOT / '.superpowers/sdd/2026-11-11-routing-service'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def number(value, digits=3):
    return 'NA' if value is None else f'{value:.{digits}f}'


def interval(value, factor=1, digits=3):
    if value['mean_difference'] is None:
        return 'NA'
    low, high = value['scenario_bootstrap_95pct_ci']
    return f"{value['mean_difference']*factor:+.{digits}f} [{low*factor:+.{digits}f}; {high*factor:+.{digits}f}]"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=STUDY)
    parser.add_argument('--sdd-report', type=Path, default=SPEC / 'task-3-report.md')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = read(STUDY / 'formal-v2-summary-v1/summary.json')
    evaluation = read(STUDY / 'formal-v2-evaluation-v1/evaluation.json')
    raw = STUDY / 'formal-v2-raw/formal-v2'
    manifest = read(raw / 'manifest.json')
    protocol = read(raw / 'protocol.json')
    provenance = read(raw / 'provenance.json')
    regimes = list(dict.fromkeys(x['regime'] for x in protocol['scenarios']))
    policies = protocol['policies']
    comparisons = evaluation['comparisons']
    means = {x['policy']: x for x in summary['summaries'] if x['regime'] == 'pooled'}
    mechanisms = defaultdict(Counter)
    coincidences = defaultdict(Counter)
    for x in summary['mechanisms']:
        mechanisms[x['policy']].update({k: v for k, v in x.items() if isinstance(v, (int, float))})
    for x in summary['coincidences']:
        coincidences[x['policy']].update({k: v for k, v in x.items() if isinstance(v, (int, float))})
    all_metrics = [
        ('actual_udp_pdr', 'PDR UDP ứng dụng', 100, 'điểm phần trăm'),
        ('total_relay_travel', 'Tổng quãng đường relay', 1, 'm'),
        ('source_dest_disconnected_fraction', 'Mất kết nối probe nguồn–đích', 100, 'điểm phần trăm'),
        ('disconnected_fraction', 'Đồ thị probe không liên thông toàn bộ', 100, 'điểm phần trăm'),
        ('olsr_unusable_fraction', 'OLSR không sử dụng được', 100, 'điểm phần trăm'),
        ('mean_delay_ms', 'Độ trễ có điều kiện', 1, 'ms'),
    ]
    annex = ['# Phụ lục: toàn bộ so sánh ghép cặp formal-v2', '',
             'Mỗi ô là trung bình (path-service trừ đối chứng) [CI percentile 95% mô tả]. '
             'Cả bốn chế độ và tám đối chứng đều được giữ. Không hiệu chỉnh nhiều so sánh.', '',
             'PDR và outage dùng điểm phần trăm. Quãng đường dùng mét. Với độ trễ, chỉ các cặp có gói nhận ở cả hai chính sách mới có giá trị; '
             'NA không được thay bằng 0. Cột gộp độ trễ có thể chỉ đại diện 3/4 chế độ.']
    for metric, title, factor, unit in all_metrics:
        annex += ['', f'## {title} ({unit})', '',
                  '| Đối chứng | n8/.55 | n8/.30 | n12/.55 | n12/.30 | Gộp |',
                  '|---|---:|---:|---:|---:|---:|']
        for ref, values in comparisons.items():
            cells = [interval(values['regimes'][r][metric], factor) for r in regimes]
            pool = values['pooled_equal_regime'][metric]
            cells += [interval(pool, factor)]
            annex.append('| ' + ref + ' | ' + ' | '.join(cells) + ' |')
        if metric == 'mean_delay_ms':
            annex += ['', '| Đối chứng | Cặp có độ trễ n8/.55 | n8/.30 | n12/.55 | n12/.30 | Chế độ có dữ liệu gộp |',
                      '|---|---:|---:|---:|---:|---:|']
            for ref, values in comparisons.items():
                cells = [str(values['regimes'][r][metric]['complete_pairs']) + '/20' for r in regimes]
                cells += [str(values['pooled_equal_regime'][metric]['represented_regimes']) + '/4']
                annex.append('| ' + ref + ' | ' + ' | '.join(cells) + ' |')
    annex += ['', '## Trung bình chính sách theo từng chế độ', '',
              '| Chế độ | Chính sách | PDR (%) | Quãng đường (m) | Outage nguồn–đích (%) | Outage toàn đồ thị (%) | OLSR không dùng được (%) | Không nhận/20 |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for x in summary['summaries']:
        if x['regime'] == 'pooled':
            continue
        annex.append(f"| {x['regime']} | {x['policy']} | {100*x['actual_udp_pdr']:.4f} | {x['total_relay_travel']:.3f} | "
                     f"{100*x['source_dest_disconnected_fraction']:.4f} | {100*x['disconnected_fraction']:.4f} | "
                     f"{100*x['olsr_unusable_fraction']:.4f} | {x['zero_received_episodes']} |")
    annex += ['', 'Số gốc và mẫu số từng chế độ: `formal-v2-summary-v1/{policy-regime-means,mechanism-counts,planner-coincidence-counts,first-action-context}.csv`. '
              'Từng kịch bản: `episodes.csv`, `paired-scenario-differences.csv`, `per-relay-motion.csv`. '
              'Toàn bộ 240 ô CI có cấu trúc: `all-paired-confidence-intervals.csv`.']
    put(output / 'PHU_LUC_KET_QUA.md', '\n'.join(annex) + '\n')

    report = ['# Kết quả điều khiển UAV theo đường đi và trạng thái dịch vụ', '',
              'Ngày hoàn thiện: 11/11/2026. Dữ liệu chính: formal-v2. Báo cáo này chốt phân tích ngoại tuyến; '
              'cổng kỹ thuật đã đạt. Rà soát độc lập Task 3 và tổng hợp Tasks 1–3 đã phê duyệt với các lưu ý không chặn. '
              'Bản sửa tài liệu này làm rõ mẫu số PDR lịch sử và nguồn gốc fixture; phạm vi kết luận khoa học giữ nguyên.', '',
              '## 1. Kết quả chính', '',
              'Trong 80 kịch bản ghép cặp, chính sách path-service nhận 33.076/600.000 gói ứng dụng, PDR 5,5127%. '
              'PDR trung bình của geometry là 6,8417%, predictive-simple 5,8460% và predictive-learned 5,3010%. '
              'Path-service di chuyển tổng cộng trung bình 1.956,189 m relay mỗi lượt, nhiều hơn geometry 762,342 m và predictive-learned 861,359 m.', '',
              'Chênh lệch PDR path-service so với predictive-learned là +0,2117 điểm phần trăm, CI mô tả 95% [−1,2967; +1,7617]. '
              'So với geometry, chênh lệch là −1,3290 [−2,5774; −0,1174] điểm phần trăm. '
              'Các kết quả này không hỗ trợ kết luận chính sách mới có ưu thế tổng thể. Việc đồ thị probe gần như liên thông cũng không bảo đảm UDP tới đích.', '',
              'Ablation bỏ thời hạn cho hành động gửi và mọi trường quan sát giống hệt chính sách đầy đủ ở 80/80 kịch bản. '
              'Trong dữ liệu đầy đủ, 101 lần bắt đầu can thiệp đều mang ngữ cảnh khôi phục; không có lần bắt đầu do dự báo phòng ngừa. '
              'Bởi vậy, kết quả hiện tại chưa chứng minh lợi ích thực nghiệm của điều kiện thời hạn.', '',
              '## 2. Thiết kế đã cố định', '',
              'Mô phỏng ns-3.45 dùng 80 cặp seed layout/radio mới, chia trước thành bốn chế độ: '
              'n8_spacing055, n8_spacing030, n12_spacing055 và n12_spacing030, mỗi chế độ 20 kịch bản. '
              'Chín chính sách dùng cùng cặp seed trong mỗi kịch bản, tổng 720 lượt. Mỗi lượt có 120 khoảng điều khiển 1 giây và 121 khung quan sát; '
              'hệ thống chạy hết thời gian, không dừng khi phục hồi hoặc hết ngân sách.', '',
              'Nguồn là UAV 0, đích là UAV 4 ở cả n8 và n12; các UAV còn lại là relay. Tốc độ lệnh relay tối đa 8 m/s, '
              'ngân sách cứng 400 m/relay, khoảng cách an toàn 5 m, tham số tầm liên lạc 240 m. '
              'Vùng mô phỏng 1.600 × 1.200 m, cao độ 50–150 m; warmup 10 giây. '
              'Chuyển động nhiệm vụ nguồn/đích là ngoại sinh, bắt đầu ở bước 10, tốc độ 3 m/s. '
              'Spacing .55/.30 là hệ số bố trí ban đầu đã ấn định, không phải nhóm chọn sau kết quả.', '',
              '| Chính sách | Vai trò |', '|---|---|',
              '| hold | Giữ relay đứng yên |',
              '| reactive | Phản ứng khi đồ thị probe mất liên thông |',
              '| geometry | Quy tắc hình học cũ |',
              '| predictive-simple | Đối chứng dự báo đơn giản cũ, ngưỡng .50 và trễ định tuyến danh nghĩa 1 s |',
              '| predictive-learned | Đối chứng dùng predictor đã đóng băng, ngưỡng .35 và trễ danh nghĩa 3 s |',
              '| path-service | Rủi ro theo đường đi, trạng thái dịch vụ, tập relay, khả thi ngân sách và cửa sổ hành động |',
              '| path-service-no-service | Bỏ thành phần dịch vụ/đường đi theo ablation đã chấp nhận, dùng đường cầu toàn cục |',
              '| path-service-no-deadline | Bỏ cửa sổ thời hạn khi có rủi ro dự báo; vẫn giữ khôi phục |',
              '| path-service-no-budget | Bỏ kiểm tra/điểm ngân sách của planner; giới hạn cứng simulator vẫn giữ |', '',
              'Phương pháp mới dùng ngưỡng rủi ro .35, rho=.50, trễ định tuyến danh nghĩa 3 s, '
              'mức dịch vụ .95 qua 3 cửa sổ và lookahead tối đa 20 s. Predictor được giữ nguyên; không huấn luyện PPO hoặc predictor mới. '
              'Số tìm kiếm tham số bằng 0. Protocol formal đóng băng trước pilot; pilot chỉ kiểm tra kỹ thuật. '
              'Audit seed đối chiếu 98 cặp lịch sử và xác nhận tách biệt.', '',
              '## 3. Định nghĩa chỉ số và cách ước lượng', '',
              'PDR chính bằng số gói UDP thực nhận ở ứng dụng đích chia số gói ứng dụng nguồn chào gửi trên toàn thời gian cấu hình, '
              'bao gồm giai đoạn trước khung điều khiển đầu và phần đuôi kết thúc. Mỗi lượt có 7.500 gói chào gửi. '
              'Bộ đếm chào gửi, IP phát nguồn, sink nhận và FlowMonitor được đối soát riêng. '
              'PDR lịch sử bằng số gói FlowMonitor nhận chia số gói chào gửi theo lịch, '
              'với mẫu số scheduled_offered = ceil(duration / interval). PDR chính dùng số sink ứng dụng nhận chia '
              'số lần chào gửi thực sự quan sát được tại ứng dụng nguồn. Trong formal-v2, các bộ đếm nhận và hai mẫu số '
              'đã đối soát trùng nhau: 7.500 gói chào gửi/lượt; sai khác nhỏ ở PDR lưu sẵn là do CSV làm tròn sáu chữ số thập phân. '
              'Số IP phát nguồn cũng bằng 7.500/lượt là kiểm tra kế toán bổ sung, không phải định nghĩa mẫu số PDR lịch sử.', '',
              'Outage nguồn–đích là tỷ phần 120 khoảng có nguồn/đích nằm khác thành phần probe. '
              'Outage toàn đồ thị là tỷ phần đồ thị probe có hơn một thành phần. '
              'OLSR không dùng được là chỉ số kiểm tra khả dụng OLSR đã có của simulator; '
              'traversal mới theo từng next hop được báo riêng. Cả ba là quan sát/chẩn đoán, không đồng nhất với giao UDP thực.', '',
              'Độ trễ là trung bình độ trễ các gói nhận trong từng lượt, rồi lấy trung bình các lượt có dữ liệu. '
              'Lượt không nhận gói có giá trị null. Phân tích ghép cặp độ trễ chỉ giữ kịch bản có độ trễ ở cả hai chính sách. '
              'Do đó các nhóm gói/lượt được điều kiện hóa khác nhau; không thể dùng trung bình độ trễ để xếp hạng chất lượng dịch vụ chung.', '',
              'Mỗi chênh lệch lấy path-service trừ đối chứng trên cùng cặp seed. Bootstrap 10.000 lần, seed 91011, '
              'lấy lại mẫu toàn bộ kịch bản trong từng chế độ; kết quả gộp cho bốn chế độ trọng số bằng nhau. '
              'CI là percentile 95%, mang tính mô tả, không hiệu chỉnh nhiều so sánh. '
              'Không bootstrap gói tin hoặc cửa sổ thời gian. Khi chỉ số độ trễ thiếu cả một chế độ, '
              'đầu ra gộp phản ánh các chế độ còn dữ liệu và báo rõ số chế độ, không được diễn giải là gộp đủ bốn chế độ.', '',
              'Formal-v2 không có lượt lỗi nên cả 80 cặp của tám đối chứng đều đầy đủ cho PDR, chuyển động và outage. '
              'Không gán PDR bằng 0 cho lỗi kỹ thuật. Cận nhạy cảm theo lưới thử vẫn lưu trong evaluation.json; '
              'ở lần chạy này cận dưới/trên trùng chênh lệch quan sát vì không có lỗi.', '',
              '## 4. Kết quả gộp đủ bốn chế độ', '',
              '| Chính sách | PDR ứng dụng (%) | Quãng đường relay/lượt (m) | Outage nguồn–đích (%) | Outage toàn đồ thị (%) | OLSR không dùng được (%) | Lượt không nhận/80 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for policy in policies:
        x = means[policy]
        report.append(f"| {policy} | {100*x['actual_udp_pdr']:.4f} | {x['total_relay_travel']:.3f} | "
                      f"{100*x['source_dest_disconnected_fraction']:.4f} | {100*x['disconnected_fraction']:.4f} | "
                      f"{100*x['olsr_unusable_fraction']:.4f} | {x['zero_received_episodes']} |")
    report += ['', '### 4.1. FlowMonitor lịch sử được giữ riêng', '',
               '| Chính sách | PDR ứng dụng chính (%) | PDR FlowMonitor đã lưu, trung bình (%) |',
               '|---|---:|---:|']
    for policy in policies:
        values = [x['original_flowmonitor_pdr'] for x in manifest if x['policy'] == policy]
        report.append(f"| {policy} | {100*means[policy]['actual_udp_pdr']:.7f} | {100*sum(values)/len(values):.7f} |")
    report += ['', 'Cả hai cột dùng 80 lượt. Cột lịch sử là FlowMonitor nhận / scheduled_offered; '
               'cột chính là sink ứng dụng nhận / observed_application_offers. Không trộn hai định nghĩa trong các chênh lệch chính.', '',
               '### 4.2. Chênh lệch ghép cặp', '',
               '| Đối chứng | ΔPDR (điểm %) [CI 95%] | Δquãng đường (m) [CI 95%] | Δoutage nguồn–đích (điểm %) [CI 95%] |',
               '|---|---:|---:|---:|']
    for ref, value in comparisons.items():
        v = value['pooled_equal_regime']
        report.append(f"| {ref} | {interval(v['actual_udp_pdr'],100,4)} | {interval(v['total_relay_travel'])} | {interval(v['source_dest_disconnected_fraction'],100,4)} |")
    report += ['', 'So với hold, path-service có PDR cao hơn trong bộ mô phỏng này, nhưng hold có quãng đường bằng 0. '
               'So với no-service, PDR chỉ khác +0,0100 điểm phần trăm [−1,6350; +1,6814], trong khi tổng quãng đường tăng '
               '952,044 m [903,460; 1.006,481]. Bỏ kiểm tra ngân sách planner cho PDR 6,6358%; '
               'chênh lệch đầy đủ trừ no-budget là −1,1232 điểm phần trăm [−2,4217; +0,1420]. '
               'Dữ liệu chưa cho thấy lợi ích PDR của thành phần dịch vụ hoặc kiểm tra ngân sách, dù ngân sách vẫn là yêu cầu kỹ thuật.', '',
               'Trung bình độ trễ có điều kiện của full là 887,550 ms (49/80 lượt), learned 742,128 ms (47/80), geometry 553,901 ms (51/80). '
               'Ở n12/.55, full và learned không có cặp nào cùng có độ trễ hợp lệ; gộp so sánh độ trễ này chỉ đại diện 3/4 chế độ. '
               'Phụ lục giữ đầy đủ CI, số cặp độ trễ và số chế độ có dữ liệu.', '',
               '![Chênh lệch PDR và chuyển động](formal-v2-summary-v1/paired-pdr-motion.png)', '',
               'Hình 1. Path-service trừ từng đối chứng; thanh là CI bootstrap kịch bản 95% mô tả. '
               'Bản PDF để xuất: `formal-v2-summary-v1/paired-pdr-motion.pdf`.', '',
               '## 5. Bốn chế độ đã ấn định', '',
               '| Chính sách | n8/.55 PDR (%) | n8/.30 | n12/.55 | n12/.30 |', '|---|---:|---:|---:|---:|']
    for policy in policies:
        values = [next(x for x in summary['summaries'] if x['regime'] == r and x['policy'] == policy) for r in regimes]
        report.append('| ' + policy + ' | ' + ' | '.join(f"{100*x['actual_udp_pdr']:.4f}" for x in values) + ' |')
    report += ['', 'Mỗi ô có 20 kịch bản. n12/.55 hầu như không giao được gói dù phần lớn chính sách giảm mạnh outage probe. '
               'Đây là khoảng cách giữa kết nối đo được và dịch vụ ứng dụng; kết quả hiện có không xác định một nguyên nhân duy nhất. '
               'Toàn bộ chênh lệch ghép cặp và CI PDR, quãng đường, ba loại outage và độ trễ của từng chế độ nằm trong '
               '[PHU_LUC_KET_QUA.md](PHU_LUC_KET_QUA.md). Không loại chế độ khó hoặc chọn nhóm thuận lợi.', '',
               '![PDR từng chế độ và khoảng cách probe–dịch vụ](formal-v2-summary-v1/regime-pdr-service-gap.png)', '',
               'Hình 2. Cả 36 trung bình chính sách × chế độ. Biểu đồ bên phải mô tả đồng biến giữa các trung bình; '
               'không ước lượng quan hệ nhân quả. Bản PDF: `formal-v2-summary-v1/regime-pdr-service-gap.pdf`.', '',
               '## 6. Cơ chế hoạt động và kiểm tra an toàn', '',
               'Mỗi chính sách mới có 9.600 quyết định. Bốn chính sách mới tạo tổng 38.400 bản ghi quyết định; '
               'các số đếm dưới đây dùng tất cả chế độ, không chọn kịch bản.', '',
               '| Chính sách | Có plan/9.600 | Không plan | Plan tập con | Có lệnh di chuyển | Bắt đầu khôi phục | Bắt đầu dự báo | Khoảng gửi bị scale ngân sách |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
    for p, x in mechanisms.items():
        report.append(f"| {p} | {x['chosen_plan_frames']} | {x['no_plan_frames']} | {x['strict_subset_plan_frames']} | "
                      f"{x['commanded_motion_frames']} | {x['restoration_starts']} | {x['forecast_starts']} | {x['budget_scaled_sent_frames']} |")
    report += ['', 'Plan tập con dùng mẫu số có plan: full 4.761/9.357; no-service 0/7.671; '
               'no-deadline 4.761/9.357; no-budget 4.476/9.377. Có plan không có nghĩa có lệnh di chuyển, '
               'vì kích hoạt và lọc an toàn vẫn áp dụng. Với full, 243 khung không plan gồm 198 khung không thay đổi hình học có thể hành động '
               'và 45 khung không có plan hoàn chỉnh đủ ngân sách. Các số này là số khung, không phải số lượt thất bại.', '',
               'Full loại 657.581 ứng viên vì ngân sách hoàn chỉnh và 1.185.267 vì thời gian còn lại; '
               '15.257 vì kiểm tra khoảng cách quét toàn relay. Một khung xét nhiều ứng viên nên không chia các số này cho 9.600 như xác suất thất bại. '
               'Không có lệnh nào bị bộ lọc an toàn một bước hoặc simulator từ chối ở bốn chính sách mới; '
               'tổng rejected_actions của cả 720 lượt cũng bằng 0.', '',
               'Audit toàn bộ formal-v2 xác nhận tốc độ/ngân sách cứng. Quãng đường relay lớn nhất 399,970743657 m; '
               'bước relay lớn nhất 8,00000000944614 m, trong dung sai kiểm tra 1e−5 m. '
               'Khoảng cách nhỏ nhất trên các khung lấy mẫu là 5,181762232 m. '
               'Đây là kiểm tra khung mẫu cùng cơ chế kiểm tra quét trong simulator; con số cực tiểu mẫu không phải chứng minh cực tiểu liên tục của mọi quỹ đạo.', '',
               '### 6.1. Đường dự kiến và đường quan sát', '',
               'Full chọn plan ở 9.357 khung, nhưng chuỗi dự kiến khác danh sách next hop quan sát ở 9.327/9.357 khung. '
               'Định nghĩa mismatch bao gồm đường quan sát thiếu hoặc traversal chưa tới đích; nó không chỉ đo hai đường hoàn chỉnh khác nhau. '
               'Chỉ 1.685 khung có plan đồng thời traversal tới đích thành công. '
               'Trên toàn bộ khung có plan, số hop dự kiến trung bình 7,720; trên 1.685 traversal thành công, hop quan sát trung bình 3,877. '
               'Hai trung bình dùng tập khung khác nhau, không phải chênh lệch hop ghép cặp.', '',
               'Nguồn đường để tính rủi ro của full: 9.430/9.600 khung dùng đường probe min-sum tại H20, '
               '69 dùng đường OLSR quan sát và 101 không có đường khả dụng. Planner không cài tuyến OLSR. '
               'Các relay không được chọn vẫn bật radio và có thể tham gia đường thực.', '',
               '### 6.2. So với planner hình học cũ trên cùng khung', '',
               'Phân tích ngoại tuyến chạy target_plan cũ trên chính khung trước hành động của chính sách mới. '
               'Sai số tuyệt đối cho target là 1e−6 m, cho vector hành động là 1e−8; rtol=0. '
               'Không so target từ hai quỹ đạo chính sách khác nhau.', '',
               '| Chính sách | Target trùng/có plan | Hành động danh nghĩa trùng/có plan | Đề xuất sau bảo vệ trùng/tất cả khung | Lệnh gửi trùng/tất cả khung | Lệnh gửi trùng/khung có lệnh |',
               '|---|---:|---:|---:|---:|---:|']
    for p, c in coincidences.items():
        report.append(f"| {p} | {c['target_coincidences']}/{c['chosen_plan_frames']} | {c['nominal_action_coincidences']}/{c['chosen_plan_frames']} | "
                      f"{c['protected_proposal_coincidences']}/{c['all_decision_frames']} | {c['sent_vs_old_geometry_action_coincidences']}/{c['all_decision_frames']} | "
                      f"{c['active_sent_coincidences']}/{c['commanded_motion_frames']} |")
    report += ['', 'Tỷ lệ target trùng của full là 2,586%; lệnh gửi trùng là 2,104% trên mọi quyết định hoặc 2,177% trên khung có lệnh. '
               'Sự khác nhau với planner cũ xác nhận cơ chế không trùng hoàn toàn; tỷ lệ này không chứng minh lợi ích giao gói.', '',
               '### 6.3. Khôi phục, dự báo và thời điểm hành động', '',
               'Full bắt đầu lệnh đầu ở bước 1 trong cả 80 lượt; tại đó nguồn–đích vẫn cùng thành phần probe. '
               'Tuy nhiên trạng thái dịch vụ/đường đi đã kích hoạt khôi phục, nên không gọi các hành động này là dự báo phòng ngừa. '
               'Trong 9.600 quyết định, full ở trạng thái khôi phục 9.518 khung; delivery_degraded 9.338, '
               'route_missing 7.887 và service_unknown 240. Tỷ lệ đến/chào gửi gần đây vượt 1 ở 10 khung, '
               'phù hợp khả năng có gói đến muộn; tỷ lệ này không phải PDR theo cohort.', '',
               'Full có 3.206 khung rủi ro vượt ngưỡng và 2.260 khung vượt ngưỡng ngoài cửa sổ hành động. '
               'Cả 2.260 khung đều đồng thời thuộc khôi phục; không có khung ngoài cửa sổ mà không khôi phục. '
               'Điều kiện khôi phục đã đủ để gửi lệnh, giải thích vì sao ablation no-deadline không đổi hành động hoặc quan sát.', '',
               'Trong 22/80 lượt full có lần mất kết nối probe nguồn–đích xuất hiện sau lệnh đầu, khoảng cách thời gian quan sát là 18–118 s '
               '(trung bình 95,864 s). 58 lượt còn lại không có sự kiện sau lệnh đầu trong chân trời quan sát. '
               'Các con số là thời điểm trên quỹ đạo đã can thiệp; không phải thời gian dự báo đúng hay bằng chứng đã tránh đứt mạng. '
               'No-service có 78 lệnh đầu mang ngữ cảnh dự báo và 2 khôi phục; toàn quá trình có 1.814 lần bắt đầu dự báo và 22 khôi phục, '
               'nhưng đối chứng này thay thành phần dịch vụ/đường đi theo thiết kế, nên không cô lập riêng hiệu chuẩn predictor.', '',
               '## 7. Lịch sử chạy và tính toàn vẹn dữ liệu', '',
               'Task 1 và Task 2 đã có các vòng sửa/rà soát được giữ trong SDD. Task 3 qua preflight kỹ thuật trước khi thu dữ liệu. '
               'Kiểm tra observer bật/tắt giữ nguyên các trường cũ và các CSV khoa học cũ trên fixture đã khớp; '
               '2.912 mảng đặc trưng và luồng hành động/chẩn đoán của năm baseline được so bằng nhau. '
               'Đây là bằng chứng không can thiệp trên fixture, không phải chứng minh mọi quỹ đạo luôn giống nhau.', '',
               'Formal-v1 dừng do race khi đóng stdout: luồng đọc có thể ghi vào file đã đóng và lỗi không được truyền đầy đủ. '
               'Giữ nguyên 302 lượt hoàn tất, 4 lượt gián đoạn và 414 lượt chưa chạy trong hồ sơ thất bại. '
               'Có khả năng log thô bị cắt; chưa có bằng chứng xác lập sai lệch khoa học của các endpoint đã hoàn tất. '
               'Không dùng bất kỳ kết quả formal-v1 nào để ước lượng hiệu quả.', '',
               'Runtimefix1 chuyển quyền đọc/drain/đóng pipe và log về cùng luồng, truyền lỗi đọc về luồng chính, '
               'đã được rà soát độc lập. Pilot-v2 gồm 18 lượt chỉ kiểm tra kỹ thuật. '
               'Formal-v2 chạy lại toàn bộ 720 lượt trong thư mục mới, giữ phương pháp, mô hình, protocol, seed và baseline. '
               'Không lựa chọn cấu hình dựa trên PDR.', '',
               'Cổng FORMAL_V2_ENGINEERING_GATE.json xác nhận 720/720 ok, 0 lỗi, '
               '12.240 hash artifact khớp; tất cả 121 khung trùng stdout và đủ 120 quyết định, '
               'bộ đếm/cuối phiên khớp, simulator/runner không có stderr. '
               'Status cuối complete=true, không còn tiến trình quan sát được và lock chung đã mất. '
               'Mã thoát trực tiếp của job nền không được thu lại; không suy diễn nó là 0. '
               'Giới hạn này tách biệt với bằng chứng status và audit raw đầy đủ.', '',
               'Bản protocol sao chép đổi CRLF sang LF. Analysisfix1 xác minh hash byte gốc hoặc dựng lại chính xác chỉ bằng chuyển LF/CRLF; '
               'không chấp nhận tái tuần tự JSON, đổi khoảng trắng hoặc đổi thứ tự khóa. '
               'Hash frozen và hash bản chạy được lưu riêng. Sửa này chỉ tác động evaluator ngoại tuyến, không đổi runner hoặc dữ liệu.', '',
               'Ba lệnh phân tích endpoint, coincidence và summary đều lưu command.json, stdout.log, stderr.log, returncode=0. '
               'Tất cả metric endpoint được tính lại từ raw và so chính xác với manifest trước khi bootstrap.', '',
               '## 8. Giả định và giới hạn', '',
               'Predictor học nhãn biến mất cạnh trong chân trời dự báo. Tổng rủi ro trên đường đi là điểm heuristic; '
               'không có giả định độc lập đã kiểm chứng, không phải xác suất mất đường đã hiệu chuẩn và không phải xác suất phản thực sau hành động. '
               'Bố trí compact, n12 và quỹ đạo do chính sách sinh ra khác miền hiệu chuẩn cũ; không có bảo đảm hiệu chuẩn tự chuyển sang các miền này.', '',
               'Planner xét một giả thuyết hình học/động học với target tĩnh và dự phóng đầu mút bị chặn lookahead. '
               'Thời gian danh nghĩa T=d/V không phải thời gian tới đích thực. '
               'Simulator nhân vector chuẩn hóa với min(V, ngân sách còn lại), nên khi còn 4 m ngân sách, target cách 4 m, '
               'vector lệnh có norm .5 ở V=8 m/s chỉ tạo 2 m bước thực. '
               'Scale ngân sách xuất hiện ở 125/9.277 khung có lệnh của full và 703/9.297 của no-budget. '
               'Kiểm tra an toàn, biên vùng và thay đổi đầu mút cũng có thể làm plan không hoàn thành như giả định.', '',
               'Bộ điều khiển có quyền truy cập tập trung vị trí, vận tốc và quãng đường của mọi UAV trực tiếp từ trạng thái simulator. '
               'Quan sát probe, routing và dịch vụ cũng đi qua pipe đồng bộ của tiến trình. '
               'Simulator dừng tiến triển khi chờ lệnh. Mô hình này chưa tính trễ telemetry, thời gian tính quyết định, trễ chấp hành, '
               'airtime/mất gói của thông điệp điều khiển hoặc chi phí năng lượng thu thập/điều khiển. '
               'Trễ định tuyến danh nghĩa 3 s trong planner không thay các chi phí triển khai đó.', '',
               'Formal không có đo thời gian từng quyết định. Thời gian wall-clock khi chạy đồng thời bốn lượt, nhất là n12, '
               'không thể dùng làm độ trễ planner hoặc xác nhận thời gian thực. '
               'Fixture n8 cục bộ trước đây cho trung bình 8,7 ms và cực đại 12,7 ms; một lần gọi planner thuần n12 là 0,140 s. '
               'Đây là fixture kỹ thuật hạn chế, không phải phân phối độ trễ triển khai hoặc kiểm chứng thời gian thực. Không bổ sung thí nghiệm timing sau kết quả.', '',
               'Chuyển relay không trực tiếp sửa hoặc cài đường OLSR. Các sai khác OLSR/ARP/chuyển tiếp và dịch vụ có thể còn ngay khi probe liên thông. '
               'Số đếm hiện có mô tả những sai khác ấy; không đủ để quy mọi thất bại cho một giao thức hoặc một lỗi riêng.', '',
               'Mỗi chế độ chỉ có 20 kịch bản, bốn chế độ cố định và chỉ dùng mô phỏng. '
               'Nhiều so sánh tương quan với nhau, CI chưa hiệu chỉnh multiplicity. '
               'Chưa có dữ liệu thực địa, mô hình điều khiển phân tán, kiểm định hiệu chuẩn mới hoặc xác minh nhân quả việc ngăn phân vùng. '
               'Không tuyên bố tính mới hay sẵn sàng công bố từ bộ kết quả này.', '',
               '## 9. Nguồn gốc, tái lập và tài liệu kèm theo', '',
               'Thư mục raw chính: `formal-v2-raw/formal-v2`; hồ sơ thực thi: `formal-v2-raw/execution`. '
               'Source/runtime/package và lệnh thực thi thực tế nằm trong provenance.json và execution/inventory-v2.json. '
               'Danh sách đầy đủ hash nguồn, thư viện runtime, binary, model, config và raw metadata được giữ; '
               'FINAL_ANALYSIS_PROVENANCE.json thêm hash script/đầu ra phân tích hiện tại và phiên bản môi trường ngoại tuyến. '
               'Nguồn evaluator ghi lúc thu dữ liệu khác evaluator đã sửa provenance khi phân tích; hai hash được giữ, không ghi đè lịch sử.', '',
               '| Đối tượng | SHA-256 |', '|---|---|']
    hash_items = {
        'method frozen': protocol['method_sha256'],
        'formal protocol frozen (CRLF)': provenance['protocol_sha256'],
        'formal protocol run copy (LF)': digest(raw / 'protocol.json'),
        'predictor frozen': provenance['model_sha256'],
        'binary thực thi': provenance['binary_sha256'],
        'source-v2 package': '418e7ebbf83242ef4a5044a0f308a069beb271dce9a6a4f699a743ecc1f4342d',
        'raw manifest': digest(raw / 'manifest.json'),
        'raw provenance': digest(raw / 'provenance.json'),
    }
    for key, value in hash_items.items():
        report.append(f'| {key} | `{value}` |')
    report += ['', 'Năm archive vận chuyển được xác minh riêng trong `formal-v2-archives.json`; hash từng shard:', '',
               '| Archive | SHA-256 |', '|---|---|']
    for item in read(STUDY / 'formal-v2-archives.json')['archives']:
        report.append(f"| {item['archive']} | `{item['sha256']}` |")
    report += ['', 'Các lệnh sau chỉ đọc raw và tạo thư mục đầu ra mới; thay hậu tố `reproduce` nếu đã tồn tại:', '',
               '```powershell',
               'rtk proxy py scripts/analysis/evaluate_path_service.py --run reports/uav_path_service_study_2026-11-11/formal-v2-raw/formal-v2 --output reports/uav_path_service_study_2026-11-11/formal-v2-evaluation-reproduce',
               'rtk proxy py scripts/analysis/summarize_path_service_pilot_engineering.py --run reports/uav_path_service_study_2026-11-11/formal-v2-raw/formal-v2 --output reports/uav_path_service_study_2026-11-11/formal-v2-coincidence-reproduce',
               'rtk proxy py scripts/analysis/summarize_path_service_results.py --run reports/uav_path_service_study_2026-11-11/formal-v2-raw/formal-v2 --evaluation reports/uav_path_service_study_2026-11-11/formal-v2-evaluation-reproduce/evaluation.json --coincidence reports/uav_path_service_study_2026-11-11/formal-v2-coincidence-reproduce/formal-coincidence.json --output reports/uav_path_service_study_2026-11-11/formal-v2-summary-reproduce',
               '```', '',
               'Đầu ra đã lưu của lần phân tích chính là `formal-v2-evaluation-v1`, `formal-v2-coincidence-v1` và `formal-v2-summary-v1`; '
               'lệnh/returncode tương ứng ở `formal-v2-*-command-v1`. '
               'Bảng gốc có 720 lượt, 640 dòng chênh lệch kịch bản (80 × 8 đối chứng), '
               'dữ liệu chuyển động từng relay và tất cả CI, cho phép kiểm tra từng kết luận.', '',
               '## 10. Phạm vi đã hoàn tất và công việc còn lại', '',
               'Đã triển khai observer dịch vụ, controller và planner mới, ablation, runner/evaluator và protocol đóng băng. '
               'Đã kiểm tra kỹ thuật qua preflight, sửa race và sửa kiểm tra provenance; formal-v2 hoàn tất 720 lượt và phân tích ngoại tuyến đã đối soát raw. '
               'Bằng chứng thực nghiệm hiện tại là kết quả thấp hoặc không khác biệt về giao gói kèm chi phí chuyển động cao; '
               'không chuyển trạng thái đạt kỹ thuật thành kết luận hiệu quả.', '',
               'Rà soát độc lập Task 3 và Tasks 1–3 đã phê duyệt trong phạm vi nghiên cứu mô phỏng âm/null này; '
               'xem task-3-final-review.md và final-whole-review.md trong SDD. '
               'Để phát triển thành nghiên cứu công bố cần thiết kế mới cho kiểm định hiệu chuẩn/miền chuyển, '
               'cơ chế routing–dịch vụ, mô hình telemetry/chi phí điều khiển và đo latency trước khi thu một bộ xác nhận độc lập mới. '
               'Các bước đó không thuộc phân tích đã chốt và không được thực hiện để sửa kết quả formal-v2.', '',
               'Nguồn gốc fixture Task 2 đã được làm rõ: test route-present/PDR0/retry3 tối giản từ '
               'static_n2_s0.55_r256_seed71001 có route thành công 31/31 và 0/1.875 gói nhận. '
               'Ca static_n8_s0.55_r256_seed71001 là ví dụ suy giảm riêng, có 8/1.875 gói nhận, PDR .004267 '
               'và 22/31 missing_entry; nó không phải fixture PDR bằng 0. Test route0/PDR0 là trường hợp biên minh họa '
               'của hợp đồng phân loại, không phải sao nguyên một dòng của ca suy giảm đó. '
               'Các số trong fixture không đổi. Định dạng mã dày vẫn là vấn đề bảo trì. '
               'Thiếu số đếm trùng planner cũ đã được xử lý bằng phân tích cùng khung nêu trên.']
    put(output / 'BAO_CAO_KET_QUA.md', '\n'.join(report) + '\n')

    # This file is an audit note, not a second independent scientific review.
    put(output / 'FINAL_WRITING_AUDIT.md', '''# Rà soát cách viết báo cáo cuối

## Bản thảo

Bản thảo được ghép từ bảng formal-v2 đã tính và các định nghĩa trong mã hiện hành. Giữ nguyên số âm, null, mẫu số và giới hạn bằng chứng.

## What makes the below so obviously AI generated?

- Một số đoạn ban đầu lặp cấu trúc liệt kê và câu kết tổng quát; không cần cho báo cáo khoa học này.
- Việc dùng từ “thành công” chung có thể trộn cổng kỹ thuật với hiệu quả UDP.
- Các câu về hành động sớm dễ khiến người đọc suy ra phòng ngừa nhân quả nếu thiếu ngữ cảnh khôi phục.

## Now make it not obviously AI generated.

Bản cuối BAO_CAO_KET_QUA.md dùng kết luận định lượng ngay đầu, ghi đúng ngữ cảnh khôi phục, giải thích mẫu số từng bảng và tách việc hoàn tất kỹ thuật khỏi bằng chứng hiệu quả. Bỏ ngôn ngữ quảng bá, lời trao đổi chatbot, nhấn mạnh tầm quan trọng chung chung và kết luận lạc quan không có dữ liệu. Không thêm ý kiến cá nhân hoặc thay mức chắc chắn của kết quả. Phụ lục giữ toàn bộ đối chứng và chế độ để phần kể chuyện không che kết quả âm.
''')
    sdd = '''# Task 3: báo cáo hoàn tất thực thi và phân tích

Ngày: 2026-11-11. Trạng thái: đã hoàn tất phần triển khai, formal-v2 và báo cáo ngoại tuyến. Các hồ sơ task-3-final-review.md và final-whole-review.md ghi phê duyệt độc lập với các lưu ý không chặn. Bản này bổ sung sửa tài liệu sau review, không tự mở rộng phạm vi phê duyệt khoa học.

## Phạm vi và thiết kế

Theo task-3-design-accepted.md và task-3-frozen: 18 pilot kỹ thuật, 720 formal = 4 chế độ × 20 kịch bản × 9 vai trò. 120 khoảng/121 khung. Baseline simple .50/1s, learned .35/3s; rho .50, service .95×3. Không tuning, retrain, PPO mới, thay protocol/model/seed/baseline, sửa simulator, SSH, hoặc thu thêm dữ liệu trong giai đoạn báo cáo cuối.

## Kết quả và bằng chứng

FORMAL_V2_ENGINEERING_GATE.json: PASS; 720/720 ok, 0 lỗi, 12.240 artifact hash xác minh, đủ chân trời, stdout exact, counter/tail khớp, stderr rỗng, hard speed/budget và sampled separation đạt. Mã thoát trực tiếp job nền không được thu lại; chỉ status cuối, tiến trình/lock vắng mặt và audit đầy đủ được xác nhận.

Ba lệnh ngoại tuyến đã ghi command.json/stdout/stderr và returncode=0: formal-v2-evaluation-command-v1, formal-v2-coincidence-command-v1, formal-v2-summary-command-v1. Endpoint tính lại từ raw, xác minh hash và metric manifest. Summary có 720 lượt, 640 dòng cặp và 38.400 decision mới. Không chọn nhóm thuận lợi; CI bootstrap 10.000 whole paired scenario, seed91011, phân tầng/equal regime, mô tả 95%, không multiplicity claim. Độ trễ có null và có so sánh chỉ đại diện 3/4 chế độ.

Full PDR 5,5127%, motion1956,189m/lượt. Full−learned PDR+0,2117 điểm% [−1,2967;+1,7617], motion+861,359m [813,745;912,873]. Full−geometry PDR−1,3290 [−2,5774;−0,1174], motion+762,342m [714,185;815,036]. Full−no-service PDR+0,0100 [−1,6350;+1,6814] với motion+952,044m. Không tuyên bố ưu thế. Full/no-deadline sent action và tất cả trường frame exact80/80. Full bắt đầu101 lần khôi phục,0 dự báo;9357plan/9600,4761subset/9357,9327mismatch/9357 (gồm route thiếu),243no-plan,125budget-scaled sent/9277motion. Safety rejects0. Planner coincidence cùng frame, atol target1e−6m/action1e−8,rtol0;242target/9357,202sent/9600 hoặc202/9277active.

## Lịch sử lỗi và thay đổi

Formal-v1 giữ nguyên302ok+4interrupted+414unstarted; không dùng outcome. Race stdout có khả năng cắt raw log, scientific corruption chưa được xác lập. Runtimefix1 đã được độc lập Approved; pilot-v2 kỹ thuật và formal-v2 chạy lại toàn lưới. Analysisfix1 đã được độc lập Approved: kiểm tra exact frozen hash bằng byte gốc hoặc LF/CRLF-only reconstruction; không semantic JSON equivalence. Chỉ evaluator ngoại tuyến đổi, runner bất biến. Report giữ hai protocol hashes và runtime/offline evaluator hashes riêng.

Giai đoạn cuối thêm summarize_path_service_results.py, write_path_service_final_report.py, báo cáo tiếng Việt, phụ lục mọi CI, writing audit, provenance, bảng/figure và cập nhật generator review. Giữ before-image generator trong task-3-final-before. Các file mới không có baseline. Không commit hoặc thay danh tính Git.

## Tài liệu chính

- reports/uav_path_service_study_2026-11-11/BAO_CAO_KET_QUA.md: báo cáo tiếng Việt tự đủ nội dung, cơ chế/âm/null/giới hạn/provenance/tái lập.
- PHU_LUC_KET_QUA.md cùng thư mục: đủ8đối chứng×4chếđộ+gộp×6endpoint và mẫu số delay.
- formal-v2-summary-v1: bảng từng lượt/cặp/relay, cơ chế/trùng planner và PNG/PDF khoa học.
- FINAL_ANALYSIS_PROVENANCE.json: hash raw/source/model/runtime pointers, offline scripts/outputs.
- formal-v2-raw/formal-v2 và execution: raw nguyên vẹn/provenance/status/command; formal-v2-archives.json kiểm tra5archive.

## Giới hạn bắt buộc giữ khi rà soát

Rủi ro cạnh biến mất và tổng rủi ro đường chỉ heuristic; chưa xác suất đường/action counterfactual đã hiệu chuẩn. Compact/n12/policy trajectory domain shift. Plan không cài OLSR, relay không chọn vẫn radio-active; dịch chuyển không bảo đảm sửa OLSR/ARP/service. T=d/V danh nghĩa khác motion thực gần ngân sách (4m remaining,target4m,norm.5→2m), còn biên/an toàn/đầu mút. Centralized simulator positions/velocities/travel và probe/routing/service qua synchronous pipe; mô phỏng dừng chờ lệnh; chưa mô hình telemetry/compute/actuation delay, control airtime/loss/energy. Formal không timing từng quyết định; wall time chạy4lượt không xác nhận latency/thời gian thực. Fixture n8mean8.7ms/max12.7ms và một n12call.140s chỉ kỹ thuật. Hành động trước sự kiện không chứng minh ngăn phân vùng; full thực tế khôi phục.20scenario/regime,simulation only,multiplicity và conditional delay giới hạn kết luận. Không novelty/publication-ready claim.

## Review và tồn đọng

Review độc lập Task3 và tổng hợp Task1–3 đã phê duyệt với Minors theo hai hồ sơ review cuối. Hai điểm tài liệu được làm rõ: PDR lịch sử dùng FlowMonitor nhận/scheduled offers=ceil(duration/interval), còn PDR chính dùng sink nhận/observed application offers; Tx nguồn là accounting check. Fixture route-present/PDR0/retry3 lấy từ n2/.55/r256/71001, còn n8/.55/r256/71001 có .004267 là ca suy giảm khác; route0/PDR0 là trường hợp biên minh họa. Không đổi số fixture hoặc kết quả. Dense formatting vẫn là maintainability; coincidence omission đã giải quyết offline. Bất kỳ phát triển nghiên cứu mới cần protocol và dữ liệu xác nhận mới, không chỉnh kết quả formal-v2.
'''
    put(args.sdd_report, sdd)
    paths = [Path(__file__), ROOT / 'scripts/analysis/evaluate_path_service.py',
             ROOT / 'scripts/analysis/summarize_path_service_pilot_engineering.py',
             ROOT / 'scripts/analysis/summarize_path_service_results.py',
             output / 'BAO_CAO_KET_QUA.md', output / 'PHU_LUC_KET_QUA.md', output / 'FINAL_WRITING_AUDIT.md', args.sdd_report.resolve()]
    for folder in ('formal-v2-evaluation-v1', 'formal-v2-coincidence-v1', 'formal-v2-summary-v1',
                   'formal-v2-evaluation-command-v1', 'formal-v2-coincidence-command-v1', 'formal-v2-summary-command-v1'):
        paths += sorted(p for p in (STUDY / folder).rglob('*') if p.is_file())
    import numpy
    import matplotlib
    index = dict(schema='path-service-final-analysis-provenance-v1', phase='offline reporting only',
                 raw_inputs=evaluation['input_hashes'], collection_source_hashes=provenance['source_hashes'],
                 collection_runtime_hashes=provenance['runtime_hashes'], model_sha256=provenance['model_sha256'],
                 binary_sha256=provenance['binary_sha256'], protocol_representation=evaluation['protocol_provenance'],
                 offline_versions=dict(python=platform.python_version(), numpy=numpy.__version__, matplotlib=matplotlib.__version__),
                 artifact_hashes={p.relative_to(ROOT).as_posix(): digest(p) for p in paths})
    put(output / 'FINAL_ANALYSIS_PROVENANCE.json', json.dumps(index, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(report=str(output / 'BAO_CAO_KET_QUA.md'), annex=str(output / 'PHU_LUC_KET_QUA.md'), indexed_artifacts=len(paths))))


if __name__ == '__main__':
    main()
