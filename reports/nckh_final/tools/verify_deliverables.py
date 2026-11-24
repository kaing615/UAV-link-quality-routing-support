"""Check final LaTeX/PDF outputs and copy verified PDFs to stable paths."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import pymupdf

ROOT=Path(__file__).resolve().parents[3]
summary={}
for name,minimum in (('nckh_final',45),('nckh_paper',6)):
    root=ROOT/'reports'/name
    pdf=root/'build/main.pdf'
    log=(root/'build/main.log').read_text(errors='replace')
    assert not re.search(r'Overfull|undefined references|Missing character|LaTeX Error',log,re.I)
    assert pdf.stat().st_mtime>=max(p.stat().st_mtime for p in root.glob('*.tex'))
    doc=pymupdf.open(pdf)
    assert len(doc)>=minimum
    text='\n'.join(page.get_text() for page in doc)
    assert '\ufffd' not in text and '23521389' in text
    for phrase in ('0,66','0,20','1,45','637','192'):
        assert phrase in text,(name,phrase)
    assert 'chưa' in text
    assert not any(marker in text for marker in ('đang chờ kết quả','Chưa phải bản nộp','Kết luận được hoàn thiện'))
    shutil.copyfile(pdf,root/'main.pdf')
    summary[name]=dict(pages=len(doc),bytes=pdf.stat().st_size,
        sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
        compiler='XeLaTeX + latexmk',missing_glyphs=False,unresolved_references=False,overfull_boxes=False)
for version in ('v8','v9'):
    audit=json.loads((ROOT/f'reports/nckh_final/evidence/{version}-audit.json').read_text(encoding='utf-8'))
    assert audit['counts']==dict(validation=80,test=192)
summary['studies']=dict(new_formal_cells=544,all_valid=True,automatic_cell_retries=0,
    v8_primary_passed=True,v9_primary_passed=False)
summary['editorial_notes']=[
    'Visual review covered title/abstract, methods, result tables/figures, conclusion and provenance pages.',
    'Chktex style warnings for half-open intervals/function parentheses do not indicate compilation failures.',
    'Detailed member task assignment in the full report is explicitly proposed and should be checked by the group before submission.',
    'No venue template or institutional submission form has been supplied; the article uses a general format.'
]
target=ROOT/'reports/nckh_final/evidence/final-qa.json'
target.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
