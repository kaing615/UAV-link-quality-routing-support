"""Derive new-study reproducibility tables from exported manifests and queue evidence."""
import hashlib
import json
from pathlib import Path
from render_new_results import table, number

REPORT=Path(__file__).resolve().parents[1]
ROOT=REPORT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    rows=[]
    hashes=[]
    summary={}
    for version in ('v8','v9'):
        base=ROOT/f'reports/uav_service_{version}/results'
        audit_path=REPORT/f'evidence/{version}-audit.json'
        if not audit_path.exists():
            continue
        audit=read(audit_path)
        selection=read(base/'selection.json')
        assert selection['models']==read(ROOT/'reports/uav_service_v8/model/selection.json')['models']
        summary[version]={}
        for phase,n in audit['counts'].items():
            queue=read(base/f'evidence/queue-{phase}/current.json')['cells']
            assert len(queue)==n
            assert all(s['direct_wrapper_exit']==0 for s in queue.values())
            start=min(s['started_unix'] for s in queue.values())
            end=max(s['finished_unix'] for s in queue.values())
            summary[version][phase]=dict(cells=n,seconds=end-start)
            rows.append([version,phase,str(n),number((end-start)/60)])
        for name,digest in audit['hashes'].items():
            if name.endswith(('freeze.json','analysis.json')):
                hashes.append([r'\path{'+f'uav_service_{version}/results/{name}'+'}',r'\code{'+digest[:16]+'}'])
        hashes.append([r'\path{'+f'uav_service_{version}/results/selection.json'+'}',
                       r'\code{'+hashlib.sha256((base/'selection.json').read_bytes()).hexdigest()[:16]+'}'])
    text=[r'\section{Hồ sơ bổ sung v8 và v9}',
          'Mỗi queue được khởi chạy một lần; bảng thời gian không cộng smoke, fit, validate hoặc analyze. '
          r'Bảng~\ref{tab:newqueues} được tính từ thời điểm bắt đầu/kết thúc wrapper của các cell; '
          r'Bảng~\ref{tab:newhashes} liên kết kết quả với package đã khóa.']
    text.append(table(['Nghiên cứu','Pha','Cell hợp lệ','Queue (phút)'],rows,
                      'Số cell và thời gian queue của các nghiên cứu bổ sung.','tab:newqueues','p{2.2cm}Yp{3cm}p{3cm}'))
    text.append(table(['Tệp trong reports/','SHA-256 (16 ký tự đầu)'],hashes,
                      'Định danh kết quả mới. Hash đầy đủ và danh sách tệp nằm trong export-manifest.json của từng nghiên cứu.',
                      'tab:newhashes','Yp{4.5cm}'))
    text.append(r'V8 dùng layout seed 3400001 trở đi, radio seed 3500001 trở đi; v9 dùng '
                '3600001 và 3700001 trở đi. Mỗi freeze chứa toàn bộ seed plan và source snapshot. '
                r'Mô hình hồi quy nằm trong \path{reports/uav_service_v8/model/selection.json}; '
                'selection v9 giữ nguyên trọng số, thêm metadata điều khiển vị trí. '
                r'Hướng dẫn tái lập và kiểm toán: \path{reports/nckh_final/REPRODUCIBILITY.md}.')
    (REPORT/'provenance_v8_v9.tex').write_text('\n\n'.join(text)+'\n',encoding='utf-8',newline='\n')
    (REPORT/'evidence/new-queues.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
