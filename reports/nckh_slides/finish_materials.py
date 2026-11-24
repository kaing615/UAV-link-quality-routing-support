"""Create speaker notes and inspection sheets without modifying slide pixels."""
import json
import hashlib
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SLIDES = json.loads((ROOT / 'slides.json').read_text(encoding='utf-8'))

def tex(value):
    escapes = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(escapes.get(char, char) for char in value)

notes = []
# Slide number = position; slides added later (with 'key') have no AI source image.
for pos, slide in enumerate(SLIDES, 1):
    notes += [r'\needspace{8\baselineskip}',
              r'\section*{Slide ' + str(pos) + '. ' + tex(slide['title']) + '}',
              r'\addcontentsline{toc}{section}{Slide ' + str(pos) + ': ' + tex(slide['title']) + '}',
              slide['note'] and tex(slide['note']),
              r'{\small\textcolor{gray}{Nguồn đối chiếu: ' + tex(slide['source']) + '.}}', '']
(ROOT.parent / 'nckh_speaker_notes' / 'notes.tex').write_text('\n\n'.join(notes), encoding='utf-8')

SLIDES = [s for s in SLIDES if 'key' not in s]
qa = ROOT / 'qa'
qa.mkdir(exist_ok=True)
manifest = []
for s in SLIDES:
    p = ROOT / 'slides' / f"slide-{s['n']:02}.png"
    with Image.open(p) as im:
        manifest.append({'slide': s['n'], 'file': p.name, 'width': im.width, 'height': im.height,
                         'relative_aspect_error': abs(im.width / im.height / (16 / 9) - 1),
                         'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
for start in range(0, len(SLIDES), 4):
    sheet = Image.new('RGB', (1672, 2 * (471 + 28)), '#dddddd')
    draw = ImageDraw.Draw(sheet)
    for k, s in enumerate(SLIDES[start:start+4]):
        with Image.open(ROOT / 'slides' / f"slide-{s['n']:02}.png") as im:
            preview = im.convert('RGB')
            preview.thumbnail((836,471))
            x, y = (k % 2) * 836, (k // 2) * 499
            draw.text((x+10, y+7), f"SLIDE {s['n']:02}", fill='black')
            sheet.paste(preview, (x,y+28))
    sheet.save(qa / f'group-{start//4+1:02}.jpg', quality=95)
(qa / 'source-images.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps({'notes':len(SLIDES),'image_sizes':sorted({(x['width'],x['height']) for x in manifest}),
                  'max_relative_aspect_error':max(x['relative_aspect_error'] for x in manifest)}, indent=2))
