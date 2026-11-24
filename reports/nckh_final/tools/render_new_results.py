"""Create LaTeX result tables from audited, locked v8/v9 artifacts."""
import json
from pathlib import Path

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def number(value, digits=2):
    return f'{value:.{digits}f}'.replace('.', ',')


def interval(value, factor=1):
    return '[' + '; '.join(number(x * factor) for x in value) + ']'


def table(headers, rows, caption, label, widths):
    return '\n'.join([r'\begin{table}[htbp]', r'\centering\small',
        r'\begin{tabularx}{\linewidth}{@{}' + widths + r'@{}}', r'\toprule',
        '&'.join(r'\textbf{' + x + '}' for x in headers) + r'\\\midrule',
        *['&'.join(row) + r'\\' for row in rows], r'\bottomrule', r'\end{tabularx}',
        r'\caption{' + caption + r'}\label{' + label + '}', r'\end{table}', ''])


def render(version):
    audit = read(REPORT / f'evidence/{version}-audit.json')
    phase = audit['phase']
    analysis = read(ROOT / f'reports/uav_service_{version}/results/analysis/{phase}/analysis.json')
    title = 'Dự báo GNN trong lựa chọn tuyến' if version == 'v8' else 'Dự báo kích hoạt điều khiển vị trí'
    text = [r'\section{Vòng ' + version + ': ' + title + r'}\label{sec:r' + version[1:] + '}']
    nval = audit['counts']['validation']
    if phase == 'test':
        text.append(f'Validation hợp lệ {nval}/{nval} cell và đạt go; test hợp lệ '
                    f'{audit["counts"]["test"]}/{audit["counts"]["test"]} cell. Các kết quả dưới đây dùng 40 khối gradual '
                    'và tám khối static của test mới, không dùng kết quả chọn epoch.')
        technical = audit['technical']
        assert all(value for key, value in technical.items() if key != 'startup_blackouts')
        assert not technical.get('startup_blackouts')
        text.append('Các kiểm tra kỹ thuật đều đạt. Toàn vẹn tuyến đúng ở '
                    f'{audit["route_integrity_count"]}/{audit["route_integrity_count"]} episode có kiểm tra tuyến.')
        text.append(r'Bảng~\ref{tab:'+version+r'sequence} trình bày các so sánh theo thứ tự; '
                    r'Hình~\ref{fig:'+version+'paired} cho thấy chênh lệch theo từng seed.')
        rows = []
        for index, step in enumerate(analysis['fixed_sequence'], 1):
            c = step['contrast']
            factor = 100 if c['metric'] == 'actual_udp_pdr' else 1
            unit = r'điểm \%' if factor == 100 else 'giây'
            state = ('Đạt' if step['success'] else 'Không đạt') if step['interpretable'] else 'Chỉ mô tả'
            label = f'H{index}: ' + r'\code{' + c['arm'] + '} $-$ ' + r'\code{' + c['comparator'] + '}'
            rows.append([label, number(c['point'] * factor) + ' ' + interval(c['ci95'], factor),
                         number(c['regimes']['n5']['point'] * factor) + ' / ' + number(c['regimes']['n7']['point'] * factor), state])
        text.append(table(['So sánh theo thứ tự',r'Chênh lệch [CI 95\%]','n5 / n7','Diễn giải'], rows,
            f'Test {version}: ' + ('PDR tính bằng điểm phần trăm.' if version == 'v8' else 'H1/H3 tính bằng giây, H2 bằng điểm phần trăm.') +
            ' Bước sau chỉ được khẳng định nếu mọi bước trước đạt.',f'tab:{version}sequence','Yp{3.5cm}p{2.3cm}p{2cm}'))
        first = analysis['fixed_sequence'][0]
        if first['success']:
            text.append('Endpoint chính đạt theo quy tắc đã định trước; các bước sau được diễn giải theo cổng thứ tự trong bảng.')
        else:
            text.append('Endpoint chính không đạt quy tắc cận dưới CI dương. Do đó các so sánh sau chỉ mô tả, '
                        'kể cả nếu CI riêng lẻ không chứa 0; không đổi kết luận chính sang baseline thuận lợi hơn.')
        text.append(r'\Fig{fig_' + version + r'_paired}{Chênh lệch endpoint chính theo từng khối seed test; đường ngang là không chênh lệch.}{fig:' + version + 'paired}')
    else:
        text.append(f'Validation hợp lệ {nval}/{nval} cell nhưng cổng go không đạt; không chạy pha test. '
                    'Các số liệu dưới đây chỉ mô tả validation, không phải bằng chứng xác nhận hiệu quả.')
        rows = [[r'\code{' + name + '}', 'Đạt' if item['passed'] else 'Không đạt']
                for name, item in audit['validation_go']['criteria'].items()]
        text.append(r'Bảng~\ref{tab:'+version+'go} ghi kết quả từng cổng validation.')
        text.append(table(['Tiêu chí','Kết quả'],rows,'Cổng validation đã định trước, '+version+'.',f'tab:{version}go','Yp{3cm}'))
        challenge = audit['validation_go']['criteria'].get('partition_challenge')
        if challenge:
            text.append(f'Có {challenge["qualifying_baseline_episodes"]}/16 baseline gradual mất kết nối ít nhất năm giây; '
                        f'cổng yêu cầu ít nhất {challenge["required"]}/16. Không điều chỉnh lại tốc độ hoặc ngưỡng sau kết quả này.')
    means, sd = audit['means']['gradual'], audit['episode_sd']['gradual']
    text.append(r'Bảng~\ref{tab:'+version+'means} mô tả mức dịch vụ và chi phí di chuyển theo arm.')
    rows=[]
    for arm, values in means.items():
        rows.append([r'\code{' + arm + '}',
            number(100*values['actual_udp_pdr']) + r' $\pm$ ' + number(100*sd[arm]['actual_udp_pdr']),
            number(values['disconnected_seconds']),number(values['total_relay_travel_m'])])
    text.append(table(['Arm',r'PDR (\%, TB $\pm$ SD)','Mất kết nối (s)','Quãng đường (m)'],rows,
        f'Mô tả gradual {phase} {version}: PDR trung bình và độ lệch chuẩn theo episode; các cột còn lại là trung bình. '
        'Quãng đường cộng trên toàn bộ relay; không phải năng lượng.',f'tab:{version}means','Yp{3.7cm}p{2.5cm}p{2.5cm}'))
    if version == 'v8' and phase == 'test':
        mse = analysis['forecast_evaluation']['balanced_gradual_mse']
        text.append(r'Bảng~\ref{tab:v8mse} đánh giá riêng sai số dự báo trên cùng các quan sát.')
        rows=[[name.upper(),number(value,7)] for name,value in mse.items()]
        text.append(table(['Dự báo','MSE'],rows,
            'MSE năm giây tương lai trên cùng trace lq-route gradual test v8, cân bằng n5/n7. Đây là đánh giá dự báo riêng, không thay thế PDR vòng kín.',
            'tab:v8mse','Yp{4cm}'))
    text.append(r'Bảng~\ref{tab:'+version+'timing} ghi thời gian tính quyết định; '
                r'Bảng~\ref{tab:'+version+'secondary} ghi các so sánh mô tả bổ sung.')
    rows=[]
    for arm, value in audit['timings'].items():
        rows.append([r'\code{' + arm + '}',number(value['median_ms'],3),number(value['p95_ms'],3)])
    text.append(table(['Arm','Trung vị (ms)',r'Phân vị 95\% (ms)'],rows,
        f'Thời gian toàn bộ quyết định {phase} {version} trên CPU server, bốn worker đồng thời. '
        'Bao gồm dự báo nếu có và luật hành động, chưa gồm truyền telemetry/lệnh.',f'tab:{version}timing','Yp{3cm}p{4cm}'))
    rows=[]
    pair=('gnn-route','lq-route') if version=='v8' else ('gnn-motion','reactive-motion')
    for context, contrasts in (('gradual',analysis['descriptive']),('static',analysis['static_descriptive'])):
        for c in contrasts:
            if (c['arm'],c['comparator'])!=pair:
                continue
            if context=='gradual' and c['metric'] not in ('background_pdr','no_olsr_route_seconds'):
                continue
            if context=='static' and c['metric'] not in ('actual_udp_pdr','disconnected_seconds'):
                continue
            factor=100 if c['metric'].endswith('pdr') else 1
            metric={'background_pdr':r'PDR luồng nền (điểm \%)','actual_udp_pdr':r'PDR (điểm \%)',
                    'no_olsr_route_seconds':'Không có tuyến OLSR (s)','disconnected_seconds':'Mất kết nối (s)'}[c['metric']]
            rows.append([context,metric,number(c['point']*factor),interval(c['ci95'],factor)])
    text.append(table(['Bối cảnh','Chỉ số','Chênh lệch',r'CI 95\%'],rows,
        f'So sánh mô tả {phase} {version}: '+pair[0]+' trừ '+pair[1]+'. Không nằm trong thứ tự kiểm định khẳng định.',
        f'tab:{version}secondary','p{1.8cm}Yp{2cm}p{3.8cm}'))
    changes=audit['route_changes']
    text.append('Tổng số lệnh đổi tuyến trên mọi episode: '+
                '; '.join(r'\code{'+arm+'}: '+str(v['commanded_changes']) for arm,v in changes.items() if v['episodes'])+'.')
    if version=='v9':
        motion=audit['motion_supplement']
        text.append('Kiểm tra chuyển động gradual: '+
            '; '.join(r'\code{'+arm+'}: '+str(v['moving_episodes'])+'/'+str(v['episodes'])+
                      ' episode có chuyển động, quãng đường relay lớn nhất '+number(v['max_relay_travel_m'])+' m'
                      for arm,v in motion.items())+'.')
        text.append('Số quyết định bị chặn bởi kiểm tra an toàn / số frame simulator từ chối chuyển động: '+
            '; '.join(r'\code{'+arm+'}: '+str(v['safety_blocked_decisions'])+' / '+str(v['simulator_rejections'])
                      for arm,v in motion.items())+'. '+
            'Số episode có relay chạm ngân sách 400 m: '+
            '; '.join(r'\code{'+arm+'}: '+str(v['episodes_with_exhausted_relay']) for arm,v in motion.items())+'.')
        text.append('Chỉ số thời điểm can thiệp là mô tả: trong các episode vừa có chuyển động vừa có mất kết nối, '+
            '; '.join(r'\code{'+arm+'} bắt đầu di chuyển trước outage ở '+str(v['motion_before_outage'])+'/'+
                      str(v['both_motion_and_outage'])+' episode' for arm,v in motion.items() if v['both_motion_and_outage'])+
            '. Không diễn giải tỷ lệ này như hiệu quả trên toàn tập hoặc bằng chứng nhân quả riêng cho trigger.')
    target = REPORT / f'results_{version}.tex'
    target.write_text('\n\n'.join(text) + '\n',encoding='utf-8')
    print(target)


if __name__ == '__main__':
    import sys
    for version in sys.argv[1:] or ['v8','v9']:
        render(version)
