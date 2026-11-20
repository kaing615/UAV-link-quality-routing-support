"""Export retained Vietnamese report and complete annex to a fresh PDF.

Document formatting only. Reads Markdown and existing figures; does not run
the simulator, evaluator, bootstrap, predictor, or any scientific computation.
"""
import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import re

from reportlab import Version
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, NextPageTemplate, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return text.translate(str.maketrans({'–': '-', '—': '-', '−': '-', '‑': '-'}))


def inline(text):
    text = normalized(text)
    # Keep readable link labels; source paths remain elsewhere in the report.
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = escape(text)
    text = re.sub(r'`([^`]+)`', r'<font color="#294d63">\1</font>', text)
    return text


def styles():
    base = getSampleStyleSheet()
    common = dict(fontName='Arial', textColor=colors.HexColor('#172b39'))
    return {
        'body': ParagraphStyle('BodyVN', **common, fontSize=9.7, leading=14,
                               spaceAfter=7, splitLongWords=True, allowWidows=0, allowOrphans=0),
        'h1': ParagraphStyle('TitleVN', **common, fontSize=18, leading=23,
                             spaceBefore=2, spaceAfter=15, keepWithNext=True),
        'h2': ParagraphStyle('SectionVN', **common, fontSize=13, leading=17,
                             spaceBefore=12, spaceAfter=7, keepWithNext=True),
        'h3': ParagraphStyle('SubsectionVN', **common, fontSize=10.5, leading=14,
                             spaceBefore=9, spaceAfter=6, keepWithNext=True),
        'cell': ParagraphStyle('CellVN', **common, fontSize=7.1, leading=9.3,
                               splitLongWords=True, wordWrap='CJK', alignment=TA_LEFT),
        'header': ParagraphStyle('HeaderVN', fontName='ArialBold', fontSize=7.0,
                                 leading=9.4, textColor=colors.white, wordWrap='CJK'),
        'code': ParagraphStyle('CodeVN', **common, fontSize=7.0, leading=10.0,
                               spaceAfter=8, wordWrap='CJK', borderPadding=7,
                               backColor=colors.HexColor('#f1f4f6')),
    }


def column_widths(count, width, annex):
    if count == 2:
        return [width * .25, width * .75]
    if annex and count == 8:
        return [105, 145] + [(width - 250) / 6] * 6
    first = 138 if count <= 4 else (148 if annex else 116)
    return [first] + [(width - first) / (count - 1)] * (count - 1)


def table_block(lines, width, style, annex):
    rows = [[x.strip() for x in line.strip().strip('|').split('|')]
            for line in lines if not re.fullmatch(r'[| :\-]+', line)]
    count = len(rows[0])
    if any(len(row) != count for row in rows):
        raise ValueError('Unequal Markdown table columns')
    content = [[Paragraph(inline(cell), style['header'] if index == 0 else style['cell'])
                for cell in row] for index, row in enumerate(rows)]
    table = Table(content, colWidths=column_widths(count, width, annex), repeatRows=1,
                  hAlign='LEFT', splitByRow=True)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#234a62')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f6')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LINEBELOW', (0, 0), (-1, 0), .4, colors.HexColor('#234a62')),
        ('LINEBELOW', (0, -1), (-1, -1), .4, colors.HexColor('#b8c6ce')),
    ]))
    return table


def markdown_blocks(path, width, style, annex=False):
    blocks = []
    lines = path.read_text(encoding='utf-8').splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line.startswith('```'):
            code = []
            index += 1
            while index < len(lines) and not lines[index].startswith('```'):
                code.append(escape(normalized(lines[index])))
                index += 1
            blocks.append(Paragraph('<br/>'.join(code), style['code']))
        elif line.startswith('|'):
            table = []
            while index < len(lines) and lines[index].startswith('|'):
                table.append(lines[index])
                index += 1
            blocks.extend([table_block(table, width, style, annex), Spacer(1, 9)])
            continue
        elif line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            blocks.append(Paragraph(inline(line.lstrip('#').strip()), style[f'h{min(level, 3)}']))
        elif line.startswith('!['):
            match = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            if not match:
                raise ValueError('Unsupported figure reference')
            figure = (path.parent / match[2]).resolve()
            image = Image(str(figure))
            image.drawHeight *= width / image.drawWidth
            image.drawWidth = width
            blocks.extend([Spacer(1, 7), image, Spacer(1, 8)])
        else:
            paragraph = [line]
            while (index + 1 < len(lines) and lines[index + 1].strip()
                   and not lines[index + 1].startswith(('#', '|', '```', '!['))):
                index += 1
                paragraph.append(lines[index])
            blocks.append(Paragraph(inline(' '.join(paragraph)), style['body']))
        index += 1
    return blocks


def header_footer(canvas, doc):
    width, height = canvas._pagesize
    canvas.saveState()
    canvas.setFont('Arial', 7.4)
    canvas.setFillColor(colors.HexColor('#5b6b75'))
    canvas.drawString(16 * mm, height - 12 * mm, 'UAV | Báo cáo kết quả formal-v2')
    canvas.drawRightString(width - 16 * mm, height - 12 * mm, '11/11/2026 | Bản sửa tài liệu 1')
    canvas.setStrokeColor(colors.HexColor('#c5d0d7'))
    canvas.setLineWidth(.4)
    canvas.line(16 * mm, height - 15 * mm, width - 16 * mm, height - 15 * mm)
    canvas.drawString(16 * mm, 10 * mm, 'Dữ liệu và kết quả số giữ nguyên; CI mang tính mô tả.')
    canvas.drawRightString(width - 16 * mm, 10 * mm, str(doc.page))
    canvas.restoreState()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--annex', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.manifest.exists():
        raise ValueError('Fresh PDF and export-manifest paths required')
    fonts = [Path('C:/Windows/Fonts/arial.ttf'), Path('C:/Windows/Fonts/arialbd.ttf')]
    pdfmetrics.registerFont(TTFont('Arial', str(fonts[0])))
    pdfmetrics.registerFont(TTFont('ArialBold', str(fonts[1])))
    pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='ArialBold', italic='Arial', boldItalic='ArialBold')
    doc = BaseDocTemplate(str(args.output), pagesize=A4, leftMargin=16 * mm,
                          rightMargin=16 * mm, topMargin=20 * mm, bottomMargin=17 * mm,
                          title='Kết quả điều khiển UAV theo đường đi và trạng thái dịch vụ',
                          author='UAV link-quality routing support', pageCompression=1)
    margin = 16 * mm
    portrait_width = A4[0] - 2 * margin
    landscape_size = landscape(A4)
    annex_width = landscape_size[0] - 2 * margin
    doc.addPageTemplates([
        PageTemplate(id='main', pagesize=A4, onPage=header_footer,
                     frames=[Frame(margin, 17 * mm, portrait_width, A4[1] - 37 * mm,
                                   leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]),
        PageTemplate(id='annex', pagesize=landscape_size, onPage=header_footer,
                     frames=[Frame(margin, 17 * mm, annex_width, landscape_size[1] - 37 * mm,
                                   leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]),
    ])
    style = styles()
    story = markdown_blocks(args.report, portrait_width, style)
    story += [NextPageTemplate('annex'), PageBreak()]
    story += markdown_blocks(args.annex, annex_width, style, annex=True)
    doc.build(story)
    figures = [(args.report.parent / match).resolve() for match in
               re.findall(r'!\[[^\]]*\]\(([^)]+)\)', args.report.read_text(encoding='utf-8'))]
    index = dict(schema='path-service-submission-pdf-export-v1',
                 scope='Document formatting from captured report/annex/figures only; no scientific rerun',
                 reportlab_version=Version,
                 input_hashes={str(p.resolve()): sha256(p) for p in [args.report, args.annex, *figures, *fonts]},
                 output=str(args.output.resolve()), output_sha256=sha256(args.output),
                 exporter_sha256=sha256(Path(__file__)), formats=['A4 portrait main report', 'A4 landscape complete annex'])
    with args.manifest.open('x', encoding='utf-8') as stream:
        json.dump(index, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(pdf=str(args.output), bytes=args.output.stat().st_size,
                         sha256=index['output_sha256']), ensure_ascii=False))


if __name__ == '__main__':
    main()
