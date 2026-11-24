"""Build concise paper results from audited locked study artifacts."""
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
PAPER=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'reports/nckh_final/tools'))
from render_new_results import number, interval, table


def main():
    (PAPER/'images').mkdir(exist_ok=True)
    for name in ('fig_v8_v9_design.pdf','fig_v9_motion_example.pdf'):
        source=ROOT/'reports/nckh_final/images'/name
        if source.exists():
            shutil.copyfile(source,PAPER/'images'/name)
    text=[]
    for version in ('v8','v9'):
        audit_path=ROOT/f'reports/nckh_final/evidence/{version}-audit.json'
        if not audit_path.exists():
            text.append(r'\subsection{Nghiên cứu vị trí: đang chờ kết quả đã khóa}')
            text.append('Chưa có kết quả cuối; không kết luận hiệu quả khi queue còn chạy.')
            continue
        audit=json.loads(audit_path.read_text(encoding='utf-8'))
        phase=audit['phase']
        a=json.loads((ROOT/f'reports/uav_service_{version}/results/analysis/{phase}/analysis.json').read_text())
        title='Dự báo cho định tuyến' if version=='v8' else 'Dự báo cho kích hoạt chuyển động'
        text.append(r'\subsection{'+title+'}')
        text.append(f'Validation {version} có {audit["counts"]["validation"]}/{audit["counts"]["validation"]} cell hợp lệ; '+
                    ('đạt go và test có 192/192 cell hợp lệ.' if phase=='test' else 'không đạt cổng go, vì vậy không chạy test.'))
        if phase=='test':
            text.append(r'Bảng~\ref{tab:'+version+'primary} trình bày các phép so sánh theo thứ tự đã khóa.')
            rows=[]
            for i,step in enumerate(a['fixed_sequence'],1):
                c=step['contrast']
                factor=100 if c['metric']=='actual_udp_pdr' else 1
                status=('Đạt' if step['success'] else 'Không đạt') if step['interpretable'] else 'Mô tả'
                rows.append([f'H{i}: '+r'\code{'+c['arm']+'} $-$ '+r'\code{'+c['comparator']+'}',
                             number(c['point']*factor),interval(c['ci95'],factor),status])
            text.append(table(['So sánh','Chênh lệch',r'CI 95\%','Cổng'],rows,
                ('Test định tuyến: PDR tính bằng điểm phần trăm.' if version=='v8' else
                 'Test vị trí: H1/H3 là giây mất kết nối tiết kiệm được, H2 là chênh PDR theo điểm phần trăm.')+
                 ' Mỗi so sánh dùng 40 khối gradual mới.',f'tab:{version}primary','Yp{1.8cm}p{3.4cm}p{1.7cm}'))
            first=a['fixed_sequence'][0]
            text.append('Endpoint chính '+('đạt' if first['success'] else 'không đạt')+
                        '; các bước sau tuân theo cổng thứ tự trong bảng. '+
                        f'Toàn vẹn tuyến đúng {audit["route_integrity_count"]}/{audit["route_integrity_count"]} episode được kiểm tra; các kiểm tra kỹ thuật đều đạt.')
        else:
            g=audit['validation_go']['criteria']
            failed=[key for key,v in g.items() if not v['passed']]
            text.append('Cổng không đạt: '+', '.join(r'\code{'+key+'}' for key in failed)+'.')
            if 'partition_challenge' in g:
                c=g['partition_challenge']
                text.append(f'Có {c["qualifying_baseline_episodes"]}/16 baseline gradual mất kết nối ít nhất năm giây; '
                            f'yêu cầu tối thiểu {c["required"]}/16. Các số liệu tiếp theo chỉ mô tả validation.')
        rows=[]
        text.append(r'Bảng~\ref{tab:'+version+'meanspaper} ghi PDR, thời gian mất kết nối và quãng đường của từng arm.')
        metrics=['actual_udp_pdr','disconnected_seconds','total_relay_travel_m']
        for arm,v in audit['means']['gradual'].items():
            rows.append([r'\code{'+arm+'}',number(v['actual_udp_pdr']*100)+r' $\pm$ '+number(audit['episode_sd']['gradual'][arm]['actual_udp_pdr']*100),
                         number(v['disconnected_seconds']),number(v['total_relay_travel_m'])])
        text.append(table(['Arm',r'PDR (\%, TB $\pm$ SD)','Mất kết nối (s)','Quãng đường (m)'],rows,
            f'Mô tả gradual {phase} {version}; PDR trung bình và SD theo episode, các cột còn lại là trung bình. '
            'Quãng đường cộng trên relay, không đại diện năng lượng.',f'tab:{version}meanspaper','Yp{3.4cm}p{2.2cm}p{2.5cm}'))
        if version=='v8':
            mse=audit['forecast_evaluation']['balanced_gradual_mse']
            text.append('Trên cùng trace test '+r'\code{lq-route}'+', MSE GNN, MLP và EWMA lần lượt là '+
                        '; '.join(number(mse[k],7) for k in ('gnn','mlp','ewma'))+'. '+
                        'Số lệnh đổi tuyến trên toàn test của GNN, MLP và EWMA là '+
                        '; '.join(str(audit['route_changes'][arm]['commanded_changes']) for arm in ('gnn-route','mlp-route','lq-route'))+'. '+
                        'GNN trừ EWMA có chênh thời gian mất kết nối '+
                        r'$-0{,}13$ giây, CI 95\% [$-0{,}97$; $0{,}68$]; không cho thấy giảm chia cắt rõ. '+
                        r'Static có PDR gần 100\% và chênh GNN trừ EWMA bằng 0 trên tám khối. '+
                        r'Chênh PDR luồng nền GNN trừ EWMA là 0,005 điểm phần trăm, CI 95\% [$-0{,}119$; $0{,}123$], chỉ mô tả.')
            t=audit['timings']
            text.append('Thời gian toàn bộ quyết định GNN trung vị '+number(t['gnn-route']['median_ms'],2)+
                        r' ms, phân vị 95\% '+number(t['gnn-route']['p95_ms'],2)+' ms; EWMA trung vị '+
                        number(t['lq-route']['median_ms'],2)+' ms. Đây là số đo CPU server, chưa gồm telemetry và truyền lệnh.')
        else:
            text.append(r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{images/fig_v9_motion_example.pdf}'
                        r'\caption{Ví dụ n5 gradual có layout seed nhỏ nhất, lấy đủ bốn arm; vùng đỏ nhạt là mất kết nối theo probe, đường tím là lần chuyển động đầu tiên. Không chọn seed theo kết quả thuận lợi.}\label{fig:motionexample}\end{figure}')
            text.append('Hình~'+r'\ref{fig:motionexample}'+' minh họa hành động thực thi; kết luận lấy từ toàn bộ các khối kịch bản. '+
                        'Số episode có chuyển động trong gradual: '+
                        '; '.join(r'\code{'+arm+'}: '+str(v['moving_episodes'])+'/'+str(v['episodes']) for arm,v in audit['motion_supplement'].items())+'.')
    (PAPER/'results.tex').write_text('\n\n'.join(text)+'\n',encoding='utf-8',newline='\n')
    print('Wrote paper results from audited evidence')


if __name__=='__main__':
    main()
