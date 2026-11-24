"""Package the final PDFs, editable sources and compact scientific evidence."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/nckh_submission'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for version in ('v8','v9'):
        audit=json.loads((ROOT/f'reports/nckh_final/evidence/{version}-audit.json').read_text())
        assert audit['phase']=='test' and audit['counts']==dict(validation=80,test=192)
    files=[]
    for folder in ('nckh_final','nckh_paper'):
        base=ROOT/'reports'/folder
        for path in base.rglob('*'):
            if not path.is_file() or any(x in path.parts for x in ('build','__pycache__')):
                continue
            if path.name in ('edit_report.py','update_scope.py','finalize_text.py'):
                continue
            if path.suffix in ('.tex','.py','.md','.pdf','.png','.jpeg','.jpg','.json','.csv'):
                files.append(path)
        assert (base/'main.pdf').is_file()
    for version in ('v8','v9'):
        files += [ROOT/f'docs/service_{version}_{kind}.md' for kind in ('protocol','execution')]
        files += [ROOT/f'reports/uav_service_{version}/{name}' for name in ('results-export.tar.gz','seed-plan.json')]
        files += list((ROOT/f'src/control/service_{version}').glob('*.py'))
        files += [ROOT/f'orchestration/service_{version}_queue.py',ROOT/f'tests/test_service_{version}.py']
    files += [ROOT/'reports/uav_service_v8/model/selection.json',ROOT/'reports/uav_service_v8/development/manifest.json',
              ROOT/'reports/uav_service_v9/selection.json']
    files += [ROOT/'scripts/train'/name for name in ('export_service_v8_data.py','train_service_v8_gnn.py')]
    files += [ROOT/'scripts/analysis'/name for name in ('prepare_service_v8.py','prepare_service_v9.py',
              'service_v8_runner.sh','smoke_service_v8.py','smoke_service_v9.py','monitor_service_queue.py',
              'export_service_results.py','deploy_service_v9.sh')]
    files=sorted(set(files))
    assert all(path.is_file() for path in files)
    OUT.mkdir(exist_ok=True)
    readme=OUT/'README.md'
    readme.write_text('''# Hồ sơ NCKH UAV — 23/11/2026

- Báo cáo đầy đủ: reports/nckh_final/main.pdf (51 trang).
- Bài báo: reports/nckh_paper/main.pdf (7 trang).
- Nguồn LaTeX và hình nằm cạnh mỗi PDF; dùng XeLaTeX để biên dịch.
- Hướng dẫn tái lập: reports/nckh_final/REPRODUCIBILITY.md.
- Kết quả v8/v9 xuất nguyên trạng nằm trong các file results-export.tar.gz.
- manifest.json ghi SHA-256 của từng tệp trong gói.

Kết quả: GNN tăng PDR định tuyến 0,66 điểm phần trăm so với chất lượng hiện
tại (CI 95% [0,38; 1,00]). Điều khiển vị trí đã được thực thi trong ns-3,
nhưng chưa chứng minh trigger GNN giảm chia cắt hơn trigger phản ứng:
0,20 giây, CI [-1,05; 1,45]. GNN học có giám sát; luật điều khiển cố định.

Nguyễn Đình Tâm là người thực hiện chính, email 23521389@gm.uit.edu.vn.
Phân công chi tiết trong báo cáo là phương án đề xuất cần nhóm đối chiếu
với đóng góp thực tế. Tài liệu đang theo bố cục chung; khi có mẫu của trường
hoặc nơi gửi bài, cần áp định dạng và biểu mẫu tương ứng.
''',encoding='utf-8',newline='\n')
    manifest=OUT/'manifest.json'
    manifest.write_text(json.dumps(dict(files={**{p.relative_to(ROOT).as_posix():sha(p) for p in files},'README.md':sha(readme)}),
                                   ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    target=OUT/'NCKH_UAV_2026-11-23.zip'
    with ZipFile(target,'w',compression=ZIP_DEFLATED) as z:
        for path in files:
            z.write(path,path.relative_to(ROOT).as_posix())
        z.write(manifest,'manifest.json')
        z.write(readme,'README.md')
    print(json.dumps(dict(archive=str(target.resolve()),files=len(files),bytes=target.stat().st_size,
                          sha256=sha(target)),indent=2))


if __name__=='__main__':
    main()
