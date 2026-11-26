"""Package the final PDFs, editable sources and compact scientific evidence."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/nckh_submission'
DATE='2026-11-26'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for version in ('v8','v9'):
        audit=json.loads((ROOT/f'reports/nckh_final/evidence/{version}-audit.json').read_text())
        assert audit['phase']=='test' and audit['counts']==dict(validation=80,test=192)
    v10=json.loads((ROOT/'reports/uav_service_v10/results/analysis/test/analysis.json').read_text())
    assert v10['ok_cells']==v10['planned_cells']==144 and v10['route_integrity_ok']
    files=[]
    for folder in ('nckh_final','nckh_paper','nckh_paper_en'):
        base=ROOT/'reports'/folder
        for path in base.rglob('*'):
            if not path.is_file() or any(x in path.parts for x in ('build','__pycache__')):
                continue
            if path.name in ('edit_report.py','update_scope.py','finalize_text.py'):
                continue
            if path.suffix in ('.tex','.py','.md','.pdf','.png','.jpeg','.jpg','.json','.csv'):
                files.append(path)
        assert (base/'main.pdf').is_file()
    for version in ('v8','v9','v10'):
        files += [ROOT/f'docs/service_{version}_{kind}.md' for kind in ('protocol','execution')]
        files += list((ROOT/f'src/control/service_{version}').glob('*.py'))
        files += [ROOT/f'orchestration/service_{version}_queue.py',ROOT/f'tests/test_service_{version}.py']
    for version in ('v8','v9'):
        files += [ROOT/f'reports/uav_service_{version}/{name}' for name in ('results-export.tar.gz','seed-plan.json')]
    files += [ROOT/'reports/uav_service_v8/model/selection.json',ROOT/'reports/uav_service_v8/development/manifest.json',
              ROOT/'reports/uav_service_v9/selection.json']
    # v8x: post-hoc smoothing check; v10: pre-registered GNN vs tuned EWMA
    files += list((ROOT/'src/control/service_v8x').glob('*.py'))
    files += [ROOT/'reports/uav_service_v8x'/name for name in ('seed-plan.json','selection.json')]
    files += [ROOT/'reports/uav_service_v10'/name for name in ('seed-plan.json','tuning-seed-plan.json','tuning-model-selection.json')]
    files += [p for p in (ROOT/'reports/uav_service_v10/results').rglob('*.json')]
    files += [ROOT/'scripts/train'/name for name in ('export_service_v8_data.py','train_service_v8_gnn.py')]
    files += [ROOT/'scripts/analysis'/name for name in ('prepare_service_v8.py','prepare_service_v9.py',
              'service_v8_runner.sh','service_v10_runner.sh','smoke_service_v8.py','smoke_service_v9.py',
              'monitor_service_queue.py','export_service_results.py','deploy_service_v9.sh')]
    files += [ROOT/'reports/nckh_slides_canva'/name for name in ('NCKH_UAV_gioi_thieu_don_gian.pptx','script_thuyet_trinh.md')]
    files=sorted(set(files))
    missing=[str(p) for p in files if not p.is_file()]
    assert not missing, missing
    OUT.mkdir(exist_ok=True)
    readme=OUT/'README.md'
    readme.write_text(f'''# Hồ sơ NCKH UAV — {DATE[8:]}/{DATE[5:7]}/{DATE[:4]}

- Báo cáo tổng kết: reports/nckh_final/main.pdf.
- Bài báo tiếng Việt: reports/nckh_paper/main.pdf; bản tiếng Anh: reports/nckh_paper_en/main.pdf.
- Slide báo cáo GVHD: reports/nckh_slides_canva/NCKH_UAV_gioi_thieu_don_gian.pptx;
  lời thuyết trình: reports/nckh_slides_canva/script_thuyet_trinh.md.
- Nguồn LaTeX và hình nằm cạnh mỗi PDF; dùng XeLaTeX để biên dịch.
- Hướng dẫn tái lập: reports/nckh_final/REPRODUCIBILITY.md.
- Kết quả v8/v9 xuất nguyên trạng trong các file results-export.tar.gz; kết quả v10
  (phân tích, gói đã khóa, hàng đợi) trong reports/uav_service_v10/results.
- manifest.json ghi SHA-256 của từng tệp trong gói.

Kết quả chính:
- Ghim tuyến theo ETX hơn OLSR-ETX đã tinh chỉnh 5,06 điểm phần trăm PDR
  (v7, CI 95% [4,21; 5,91]), đổi lại chi phí báo hiệu cao hơn.
- GNN tăng PDR 0,66 điểm phần trăm so với EWMA 0,5 (v8, CI 95% [0,38; 1,00]);
  thăm dò sau test (v8x) cho thấy EWMA làm mượt mạnh hơn đạt mức tương đương.
- v10 đăng ký trước, GNN và EWMA cùng được chỉnh hệ số làm mượt: chênh +0,08
  điểm phần trăm, CI 95% [-0,11; 0,29], không phát hiện khác biệt.
- v9: trigger GNN chỉ giảm 0,20 giây chia cắt so với trigger phản ứng,
  CI 95% [-1,05; 1,45]; endpoint chính không đạt.
GNN học có giám sát; luật điều khiển cố định.

Nguyễn Đình Tâm là người thực hiện chính, email 23521389@gm.uit.edu.vn.

Việc còn mở trước khi nộp:
- Tên đề tài trên bìa báo cáo khác tên đã đăng ký trong thuyết minh; cần GVHD quyết định.
- Bảng phân công trong báo cáo cần nhóm xác nhận theo đóng góp thực tế.
- Bài báo chưa áp mẫu của nơi gửi.
''',encoding='utf-8',newline='\n')
    manifest=OUT/'manifest.json'
    manifest.write_text(json.dumps(dict(files={**{p.relative_to(ROOT).as_posix():sha(p) for p in files},'README.md':sha(readme)}),
                                   ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    target=OUT/f'NCKH_UAV_{DATE}.zip'
    with ZipFile(target,'w',compression=ZIP_DEFLATED) as z:
        for path in files:
            z.write(path,path.relative_to(ROOT).as_posix())
        z.write(manifest,'manifest.json')
        z.write(readme,'README.md')
    print(json.dumps(dict(archive=str(target.resolve()),files=len(files),bytes=target.stat().st_size,
                          sha256=sha(target)),indent=2))


if __name__=='__main__':
    main()
