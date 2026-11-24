"""Package approved PDFs, the editable deck and LaTeX sources for the presentation."""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT.parent
OUT = REPORTS / 'nckh_presentation_package'
OUT.mkdir(exist_ok=True)
SLIDES = len(json.loads((ROOT / 'slides.json').read_text(encoding='utf-8')))
# Slides are exported from the editable deck since 24/11/2026; the 26-slide AI-image deck is archived in final/.
for stale in ('04_Slide_bao_cao.pptx', '05_Slide_bao_cao_editable.pptx'):
    (OUT / stale).unlink(missing_ok=True)
files = [
    (ROOT / 'final' / 'NCKH_UAV_slides.pdf', '01_Slide_bao_cao.pdf'),
    (REPORTS / 'nckh_explained' / 'build' / 'main.pdf', '02_Giai_thich_tu_dau.pdf'),
    (REPORTS / 'nckh_speaker_notes' / 'build' / 'main.pdf', '03_Loi_thuyet_trinh.pdf'),
    (ROOT / 'final' / 'NCKH_UAV_bao_cao_editable.pptx', '04_Slide_bao_cao_editable.pptx'),
]
manifest = []
for src, name in files:
    dest = OUT / name
    shutil.copy2(src, dest)
    entry = {'file': name, 'bytes': dest.stat().st_size, 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
    if name.endswith('.pdf'):
        with pymupdf.open(dest) as pdf:
            entry['pages'] = len(pdf)
            if name.startswith('01'):
                assert len(pdf) == SLIDES, (len(pdf), SLIDES)
                assert all(abs(p.rect.width / p.rect.height - 16 / 9) < 1e-6 for p in pdf)
    manifest.append(entry)
for name in ('nckh_explained', 'nckh_speaker_notes'):
    dest = OUT / 'latex' / name
    dest.mkdir(parents=True, exist_ok=True)
    for src in (REPORTS / name).glob('*.tex'):
        shutil.copy2(src, dest / src.name)
    shutil.copy2(REPORTS / name / 'build' / 'main.pdf', REPORTS / name / 'main.pdf')
readme = f'''# Bộ báo cáo NCKH mạng UAV

Cập nhật ngày 24/11/2026: thêm kết quả kiểm định v10 và các phân tích sau test (làm mượt EWMA, giới hạn hình học). Bản ngày 23/11 (26 slide ảnh) được lưu trong `reports/nckh_slides/final/archive-2026-11-23-26slides`.

## Đọc theo thứ tự nào?

1. Đọc **02_Giai_thich_tu_dau.pdf** nếu chưa nắm đề tài: 8 chương, từ khái niệm mạng đến quy trình, công cụ, kết quả và câu hỏi bảo vệ.
2. Mở **01_Slide_bao_cao.pdf** để xem {SLIDES} slide trình chiếu, phù hợp buổi báo cáo khoảng 20–25 phút.
3. Dùng **03_Loi_thuyet_trinh.pdf** để tập nói theo từng slide. Mỗi phần có lời gợi ý và nguồn đối chiếu.
4. **04_Slide_bao_cao_editable.pptx** là bản PowerPoint chỉnh sửa được: chữ, số liệu và sơ đồ đều là đối tượng riêng, font Arial; PDF số 01 được xuất từ chính tệp này. Notes chứa lời thuyết trình.

Thư mục `latex/` chứa nguồn của hai tài liệu 02 và 03, biên dịch bằng XeLaTeX (cần Times New Roman và Arial). Mã dựng slide ở `reports/nckh_slides/editable` và `reports/nckh_slides/.build/build_editable.mjs`.

## Nội dung cần giữ khi trình bày

- V8: GNN tăng trung bình 0,66 điểm phần trăm PDR so với bộ chọn tuyến dùng EWMA 0,5 chưa chỉnh; CI 95% [0,38; 1,00].
- V10 (đăng ký trước): khi cả EWMA và GNN cùng chọn hệ số làm mượt trên tập tuning riêng (α = β = 0,1), GNN chỉ hơn 0,08 điểm phần trăm, CI 95% [−0,11; 0,29]: không phát hiện khác biệt. Lợi ích định tuyến đến từ ước lượng chất lượng ổn định hơn.
- V9: phản ứng trừ GNN bằng +0,20 giây chia cắt; CI 95% [−1,05; 1,45], chưa chứng minh lợi ích thêm. Relay đặt lý tưởng vẫn chia cắt khoảng 26 giây, nên luật phản ứng đã đạt khoảng 94% mức giảm có thể có.
- Đây là kết quả mô phỏng; GNN học có giám sát, luật điều khiển cố định, chưa dùng RL hay thử bay thật.

## Kiểm tra bản giao

PDF slide có {SLIDES} trang 16:9, xuất bằng PowerPoint từ bản editable; hai slide mới (23, 24) và các nhãn sửa ở slide 27, 28 đã được xem trực quan. Hai tài liệu LaTeX biên dịch không có cảnh báo tràn khung hoặc tham chiếu chưa giải quyết. `manifest.json` lưu số trang và SHA-256 của các tệp.
'''
(OUT / 'README.md').write_text(readme, encoding='utf-8')
(OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
archive = REPORTS / 'NCKH_slides_va_giai_thich.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            z.write(path, arcname=str(Path(OUT.name) / path.relative_to(OUT)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print(json.dumps({'files': manifest, 'archive': str(archive), 'archive_bytes': archive.stat().st_size}, ensure_ascii=False, indent=2))
