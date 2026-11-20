"""Standalone Vietnamese PDF/source renderer; imports no simulation modules.

Both inputs are immutable directories with manifest.json listing every data
file's SHA256. Engineering data lives in engineering.json, schema
service-v2-engineering-v1; its named sections are rendered without inventing
acceptance, results or missing values. report.json pins all input/output bytes.
"""
from pathlib import Path
from html import escape
import hashlib
import importlib.util
import csv
import json
import os
import subprocess
import sys
from uuid import uuid4


# Standalone worker helpers deliberately have no package imports: importing
# src.control would import the old Gymnasium environment before this worker.
def strict_json(path):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result: raise ValueError('Duplicate JSON field: '+key)
            result[key]=value
        return result
    def reject(value): raise ValueError('Nonfinite JSON: '+value)
    value=json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=pairs,parse_constant=reject)
    json.dumps(value,allow_nan=False)
    return value


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_path(root,name):
    if not isinstance(name,str) or not name or '\\' in name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Unsafe report-relative path')
    target=(Path(root)/name).resolve()
    if not target.is_relative_to(Path(root).resolve()): raise ValueError('Report path escape')
    return target


def reserve(path):
    p=Path(path)
    if p.exists() or p.is_symlink(): raise FileExistsError(str(p))
    p.mkdir(parents=True,exist_ok=False)
    return p


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream: json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2)


def _verified_input(root, filename, schema):
    root=Path(root).resolve(); manifest=strict_json(root/'manifest.json')
    files=manifest.get('files')
    if not isinstance(files,dict) or filename not in files: raise ValueError('Immutable input manifest required')
    if {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}!=set(files)|{'manifest.json'}:
        raise ValueError('Unlisted/missing report input')
    for name,digest in files.items():
        if sha256(safe_path(root,name))!=digest: raise ValueError('Report input hash mismatch')
    value=strict_json(root/filename)
    if value.get('schema')!=schema: raise ValueError('Report input schema')
    return value,sha256(root/'manifest.json')


def _display(value):
    if value is None: return 'không xác định'
    if isinstance(value,float): return f'{value:.6g}'
    if isinstance(value,(list,dict)): return json.dumps(value,ensure_ascii=False,allow_nan=False)
    return str(value)


def build_report(analysis_root, engineering_root, output):
    analysis,analysis_hash=_verified_input(analysis_root,'analysis.json','service-v2-analysis-v1')
    engineering,engineering_hash=_verified_input(engineering_root,'engineering.json','service-v2-engineering-v1')
    if Path(output).exists() or Path(output).is_symlink(): raise FileExistsError(str(output))
    if importlib.util.find_spec('reportlab') is None:
        bundled=Path('C:/Users/kaing/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
        if not bundled.is_file(): raise ValueError('ReportLab unavailable; use a document runtime with ReportLab for reporting.py worker')
        command=[str(bundled),str(Path(__file__).resolve()),'--analysis',str(Path(analysis_root).resolve()),
            '--engineering',str(Path(engineering_root).resolve()),'--output',str(Path(output).resolve())]
        process=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
        if process.returncode: raise ValueError(f'Report worker exit {process.returncode}: {process.stderr}')
        return strict_json(Path(output)/'report.json')
    out=reserve(output)
    (out/'tables').mkdir(); (out/'figures').mkdir()
    # Lazy imports allow the bundled document runtime without Gymnasium.
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
    from reportlab.graphics.shapes import Drawing, Line, Circle, String
    candidates=[Path('C:/Windows/Fonts/arial.ttf'),
        Path('C:/Users/kaing/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/share/fonts/DejaVuSans.ttf'),
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')]
    font=next((p for p in candidates if p.is_file()),None)
    if font is None: raise ValueError('Vietnamese-capable font unavailable')
    pdfmetrics.registerFont(TTFont('ServiceV2',str(font)))
    styles=getSampleStyleSheet()
    for name in styles.byName:
        if hasattr(styles[name],'fontName'): styles[name].fontName='ServiceV2'
        if hasattr(styles[name],'leading'): styles[name].leading=max(styles[name].leading,styles[name].fontSize*1.4)
    styles.add(ParagraphStyle(name='SmallV2',fontName='ServiceV2',fontSize=7,leading=10,wordWrap='CJK'))
    story=[]; source=[]
    synthetic=bool(analysis.get('synthetic_only') or engineering.get('synthetic_only'))
    def paragraph(text,style='BodyText'):
        story.append(Paragraph(escape(text).replace('\n','<br/>'),styles[style])); story.append(Spacer(1,5)); source.append(text+'\n')
    def heading(text):
        paragraph(text,'Heading2'); source[-1]='## '+text+'\n'
    def table(headers,rows,widths=None):
        csv_path=out/'tables'/f'table-{len(list((out/"tables").glob("*.csv")))+1:03d}.csv'
        with csv_path.open('x',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream); writer.writerow(headers); writer.writerows([[_display(v) for v in row] for row in rows])
        values=[[Paragraph(escape(_display(v)),styles['SmallV2']) for v in row] for row in [headers,*rows]]
        if not rows: paragraph('Không có quan sát trong mục này.'); return
        t=Table(values,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7edf4')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#bcc5cf')),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.append(t); story.append(Spacer(1,10))
        source.append('| '+' | '.join(map(str,headers))+' |\n| '+' | '.join('---' for _ in headers)+' |')
        source.extend('| '+' | '.join(_display(v).replace('|','/') for v in row)+' |' for row in rows); source.append('')
    if synthetic: paragraph('SYNTHETIC TEST DATA - NOT RESEARCH RESULTS','Heading1')
    paragraph('Dự báo suy giảm dịch vụ và điều khiển chuyển tiếp UAV','Title')
    paragraph('Báo cáo phiên bản service-prevention-v2','Heading1')
    paragraph(f"Pha: {analysis['phase']}. Số ô có dữ liệu hợp lệ: {analysis['ok_cells']}/{analysis['planned_cells']}. Lưới đầy đủ: {_display(analysis['complete'])}.")
    paragraph('Đơn vị phân tích là cặp hạt giống bố trí và vô tuyến. Các gói tin và cửa sổ thời gian không phải các lần lặp độc lập. Kết quả âm, không có sự kiện, không được kích hoạt và lỗi kỹ thuật đều được giữ lại.')
    heading('1. Phương pháp và phạm vi')
    paragraph('Hai cấu hình n4/.15 và n5/.10 dùng commRange 240 m, 240 khoảng điều khiển một giây và 241 khung đo. PDR ứng dụng bằng số gói đến bộ nhận chia cho số gói được ứng dụng phát chào tại cùng ranh giới kết thúc. Không thêm thời gian xả gói. Bằng chứng Stage A/B ở 243 m chỉ là bối cảnh lịch sử.')
    paragraph('Mô hình dự báo suy giảm dịch vụ ở chân trời 10 và 20 giây trên thời điểm đã được xác nhận khỏe và sẵn sàng. Xác suất này không biểu diễn lợi ích của một hành động. Khoảng cho phép định tuyến 3 giây là giả thuyết lập kế hoạch. Cầu điều khiển đồng bộ dừng tiến trình mô phỏng khi tính quyết định; thời gian xử lý không phải trễ truyền của mặt phẳng điều khiển.')
    paragraph('Phạm vi phiên bản này là mô hình logistic dự báo suy giảm dịch vụ 10/20 giây và điều khiển chuyển động chuyển tiếp. Phiên bản này không triển khai GNN hoặc RL và không hoàn thành thay cho các mục tiêu GNN/RL của đề xuất ban đầu.')
    if synthetic:
        paragraph('DỮ LIỆU KIỂM THỬ TỔNG HỢP - KHÔNG PHẢI KẾT QUẢ NGHIÊN CỨU','Heading1')
    heading('2. PDR và quãng đường di chuyển')
    summary=[]
    for context,regimes in analysis.get('descriptive',{}).items():
        for regime,roles in regimes.items():
            for role,r in roles.items():
                summary.append([context,regime,role,f"{r['observed']}/{r['planned']}",r['mean_application_pdr'],r['mean_total_relay_meters']])
    table(['Ngữ cảnh','Chuỗi','Vai trò','Hợp lệ/kế hoạch','PDR trung bình','Tổng mét'],summary,[48,35,169,68,80,80])
    heading('3. Hiệu ứng ghép cặp')
    paragraph('So sánh chính là service-v2 trừ service-v2-no-forecast. Trung bình gộp cân bằng hai cấu hình. Bootstrap 10000 lần, RNG 20261112, khoảng phần trăm 95%; giữ nguyên khối hạt giống và toàn bộ vai trò khi lấy mẫu. Các so sánh khác có ý nghĩa mô tả. Cờ chênh PDR ít nhất .01 là mô tả thực hành, không phải ngưỡng kiểm định.')
    contrasts=[]
    for role,contrast in analysis.get('contrasts',{}).items():
        for regime,r in [*contrast['regimes'].items(),('gộp',contrast['pooled'])]:
            contrasts.append([role,regime,r['pdr_difference'],r['pdr_ci95'],r['motion_difference_m'],r['missing_pdr_sensitivity']])
    table(['Đối chứng','Chuỗi','Chênh PDR','CI 95%','Chênh mét','Biên PDR thiếu'],contrasts,[150,35,65,95,65,70])
    # Exportable vector scatter: each point is one episode, no metric alteration.
    valid=[r for r in analysis.get('rows',[]) if r['status']=='ok']
    if valid:
        drawing=Drawing(490,230); x0,y0=45,32; width,height=420,170
        xmax=max(1,max(r['total_relay_travel_m'] for r in valid))
        drawing.add(Line(x0,y0,x0+width,y0)); drawing.add(Line(x0,y0,x0,y0+height))
        palette=[colors.HexColor(v) for v in ('#1e40af','#b91c1c','#047857','#7e22ce','#c2410c','#0e7490','#4d7c0f','#7c2d12')]
        for r in valid:
            role=r['policy']; c=palette[list(POLICY_ORDER).index(role)] if role in POLICY_ORDER else colors.black
            drawing.add(Circle(x0+r['total_relay_travel_m']/xmax*width,y0+r['actual_udp_pdr']*height,2.1,fillColor=c,strokeColor=None))
        drawing.add(String(x0,12,'Total relay travel (m)',fontName='ServiceV2',fontSize=9))
        drawing.add(String(3,210,'PDR',fontName='ServiceV2',fontSize=9)); drawing.add(String(x0+width-30,12,f'{xmax:.0f}',fontName='ServiceV2',fontSize=8))
        for i,role in enumerate(POLICY_ORDER):
            if role in {r['policy'] for r in valid}:
                drawing.add(Circle(45+(i%2)*240,223-(i//2)*10,2.5,fillColor=palette[i],strokeColor=None))
                drawing.add(String(53+(i%2)*240,220-(i//2)*10,role,fontName='ServiceV2',fontSize=6))
        story.append(drawing)
        from reportlab.graphics import renderSVG
        renderSVG.drawToFile(drawing,str(out/'figures/pdr-motion.svg'))
        source.append('![PDR và quãng đường](figures/pdr-motion.svg)\n')
    heading('4. Sự kiện, khởi động và phục hồi')
    paragraph('Thời điểm không có sự kiện được kiểm duyệt phải, không thay bằng 0 và không loại bỏ. Thời lượng không sự kiện bị giới hạn ở 240 giây, tính từ cửa sổ đầy đủ đầu tiên. Phân nhóm HOLD sẵn sàng trước hoặc tại bước 20 chỉ là mô tả; suy luận chính vẫn dùng tất cả hạt giống.')
    event_summary=[]; startup_summary=[]
    for context,regimes in analysis.get('descriptive',{}).items():
        for regime,roles in regimes.items():
            for role,r in roles.items():
                for event,metric in r['events'].items():
                    event_summary.append([context,regime,role,event,f"{metric['numerator']}/{metric['observed_denominator']} (kế hoạch {metric['planned_denominator']})",metric['restricted_event_free_mean_seconds']])
                def fraction(value):
                    return f"{value['numerator']}/{value['denominator']}"+(f" (kế hoạch {value['planned_denominator']})" if 'planned_denominator' in value else '')
                startup_summary.append([context,regime,role,fraction(r['armed_by_step20']),fraction(r['never_armed']),fraction(r['startup_restoration'])])
    table(['Ngữ cảnh','Chuỗi','Vai trò','Sự kiện','Số ca/mẫu','Giây không sự kiện TB'],event_summary,[45,30,145,65,105,90])
    table(['Ngữ cảnh','Chuỗi','Vai trò','Sẵn sàng bước20','Không sẵn sàng','Phục hồi khởi động'],startup_summary,[45,30,145,100,80,80])
    endpoints=[]
    for r in analysis.get('rows',[]):
        if r['status']!='ok': continue
        e=r['endpoints']; event=e['events']['degraded']
        endpoints.append([r['cell_id'],e['first_armed_step'],event['onset_step'],event['confirmation_step'],event['restricted_event_free_seconds'],e['actual_moving_seconds'],e['forecast_starts']])
    table(['Ô','Sẵn sàng','Bắt đầu giảm','Xác nhận','Giây không sự kiện','Giây di chuyển','Bắt đầu dự báo'],endpoints,[150,45,60,55,65,55,50])
    heading('5. Kiểm định dự báo trên HOLD')
    forecast=analysis.get('forecasts')
    if forecast is None: paragraph('Pha này chưa có mô hình mới; không có điểm dự báo.')
    else:
        score_rows=[]
        for h,r in forecast['horizons'].items():
            score_rows.append([h,r['rows'],r['censored_rows'],r['prevalence'],r['brier'],r['log_loss'],r['pr_auc']])
        table(['H (s)','Hàng đủ','Kiểm duyệt','Tỷ lệ dương','Brier','Log loss','PR-AUC'],score_rows)
        table(['Chỉ số cảnh báo','Giá trị'],[[k,forecast.get(k)] for k in ('false_warning_count','evaluable_warning_count','censored_warning_count','false_warnings_per100_eligible_minutes','false_warning_fraction','event_sensitivity','eligible_event_denominator','ordinary_events_outside_risk_set','redundant_positive_warnings')],[270,210])
        for h,r in forecast['horizons'].items():
            paragraph(f'Độ tin cậy H={h} giây','Heading3')
            table(['Cận dưới','Cận trên','Số hàng','Xác suất TB','Tỷ lệ sự kiện'],[[b['lower'],b['upper'],b['count'],b['mean_probability'],b['event_fraction']] for b in r['reliability']])
        paragraph('Chuỗi dự báo liên tiếp từ .50 trở lên được gộp thành một cảnh báo. Nhãn đúng/sai dùng chân trời ngắn nhất vượt ngưỡng tại lúc bắt đầu, với đầy đủ dữ liệu tương lai. Cảnh báo dương lặp lại được giữ riêng. Cơ hội từ điều khiển bóng trên HOLD không phải chuyển động đã thực hiện. Trên quỹ đạo được can thiệp, không có sự kiện sau đó không đủ để gọi một cảnh báo là sai.')
    heading('6. Kiểm tra kỹ thuật và nguồn gốc')
    paragraph('Các dữ liệu kỹ thuật sau được chuyển nguyên ý từ gói engineering đã đóng băng. Việc renderer chạy thành công không tự xác nhận cổng chấp thuận pilot hoặc formal.')
    def render_object(value,prefix=''):
        if isinstance(value,dict):
            for key,child in value.items(): render_object(child,prefix+'.'+key if prefix else key)
        elif isinstance(value,list) and value and all(isinstance(x,dict) for x in value):
            for i,child in enumerate(value): render_object(child,f'{prefix}[{i}]')
        else: paragraph(prefix+': '+_display(value),'SmallV2')
    render_object(engineering)
    paragraph('Mã băm manifest phân tích: '+analysis_hash,'SmallV2')
    paragraph('Mã băm manifest engineering: '+engineering_hash,'SmallV2')
    timing=analysis.get('timing',{})
    render_object({k:v for k,v in timing.items() if k!='subprocess_wall_seconds'},'timing')
    wall=timing.get('subprocess_wall_seconds',[])
    with (out/'tables/subprocess-wall-seconds.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream); writer.writerow(['cell_id','attempt','wall_seconds'])
        writer.writerows([r['cell_id'],r['attempt'],r['seconds']] for r in wall)
    paragraph(f"Thời gian tiến trình của {len(wall)} lần thử được giữ đầy đủ trong tables/subprocess-wall-seconds.csv.",'SmallV2')
    heading('7. Giới hạn diễn giải')
    for text in analysis.get('inference_limits',[]): paragraph(text)
    paragraph('Hạt giống chung giúp ghép cặp kịch bản; chuyển động khác nhau làm đổi lịch sự kiện và cách dùng số ngẫu nhiên. Không diễn giải thành đối chứng cùng gói tin. Quãng đường không cho phép suy ra năng lượng, pin hay thời gian bay. Các lý do loại gói ở PHY/MAC/IP/ARP có thể chồng lấp. Kết quả không bảo đảm ưu thế và không xác lập tính mới.')
    heading('8. Toàn bộ ô theo kế hoạch')
    table(['Ô','Trạng thái','Số lần thử','PDR','Nhận/chào','Chênh sink-FM','Mét'],[[r['cell_id'],r['status'],len(r.get('attempt_records',[])),r.get('actual_udp_pdr'),f"{r.get('application_received_packets')}/{r.get('application_offered_packets')}",r.get('sink_minus_flowmonitor_received'),r.get('total_relay_travel_m')] for r in analysis.get('rows',[])],[150,65,45,45,80,55,40])
    paragraph('Các điểm ghép cặp, khung đo cơ chế, chuỗi rủi ro, kiểm duyệt và giá trị null đầy đủ nằm trong analysis.json được sao chép kèm theo. PDF không thay đổi số liệu đầu vào.')
    def page_number(canvas,doc):
        canvas.saveState(); canvas.setFont('ServiceV2',8); canvas.drawRightString(A4[0]-35,20,str(doc.page))
        if synthetic: canvas.setFont('ServiceV2',7); canvas.drawString(32,20,'SYNTHETIC TEST DATA - NOT RESEARCH RESULTS')
        canvas.restoreState()
    doc=SimpleDocTemplate(str(out/'report.vi.pdf'),pagesize=A4,rightMargin=32,leftMargin=32,topMargin=35,bottomMargin=35,
        title='Service prevention v2',author='UAV service study')
    doc.build(story,onFirstPage=page_number,onLaterPages=page_number)
    (out/'report.vi.md').write_text('\n'.join(source),encoding='utf-8')
    (out/'analysis.json').write_bytes((Path(analysis_root)/'analysis.json').read_bytes())
    (out/'engineering.json').write_bytes((Path(engineering_root)/'engineering.json').read_bytes())
    (out/'analysis-manifest.json').write_bytes((Path(analysis_root)/'manifest.json').read_bytes())
    (out/'engineering-manifest.json').write_bytes((Path(engineering_root)/'manifest.json').read_bytes())
    source_inputs=strict_json(Path(analysis_root)/'inputs.json') if (Path(analysis_root)/'inputs.json').is_file() else {}
    reanalysis=[sys.executable,'-m','src.control.service_v2.cli','analyze','--protocol',source_inputs.get('original_protocol_path','ORIGINAL_FROZEN_PROTOCOL'),
        '--run-root',source_inputs.get('original_run_root','ORIGINAL_RAW_RUN_ROOT'),'--output','FRESH_EXCLUSIVE_ANALYSIS_ROOT']
    write_json(out/'reproduce.json',dict(schema='service-v2-reproduce-v1',synthetic_only=synthetic,
        original_roots=dict(analysis=str(Path(analysis_root).resolve()),engineering=str(Path(engineering_root).resolve()),**source_inputs),
        relocation=dict(analysis_files='relative to downloaded analysis root',engineering_files='relative to downloaded engineering root',
            report_files='relative to this report root',absolute_data_paths='retained provenance; never rewrite frozen bytes'),
        read_only_reanalysis_argv=reanalysis,automatic_simulation=False,
        report_only_worker_argv=[sys.executable,str(Path(__file__).resolve()),'--analysis','DOWNLOADED_ANALYSIS_ROOT','--engineering','DOWNLOADED_ENGINEERING_ROOT','--output','FRESH_REPORT_ROOT']))
    readme='''# Tái kiểm tra gói báo cáo\n\n'''
    if synthetic: readme+='**SYNTHETIC TEST DATA - NOT RESEARCH RESULTS.**\n\n'
    readme+='''Các tệp report.vi.md và report.vi.pdf được sinh từ dữ liệu đã phân tích, không thu thập hay điều chỉnh mô hình. tables/ chứa bảng CSV UTF-8; figures/ chứa hình vector. analysis.json và engineering.json giữ nguyên byte đầu vào.\n\n## Kiểm tra chỉ đọc\n\nĐọc artifact-inventory.json và tính SHA256 từng đường dẫn tương đối từ thư mục gói này; từ chối thiếu tệp hoặc khác byte. Không sửa đường dẫn tuyệt đối bên trong dữ liệu. Các đường dẫn đó ghi nguồn gốc trên máy chạy ban đầu. Mỗi manifest đầu vào dùng đường dẫn tương đối từ đúng thư mục artifact đó.\n\n## Chuyển máy và tái phân tích\n\nSao chép nguyên gốc các thư mục frozen, raw, analysis và engineering sang đích mới. Ghi ánh xạ ngoài gói bất biến: original_root -> downloaded_root. Việc dựng PDF chỉ cần các thư mục analysis/engineering đã tải về và manifest tương đối của chúng; không mở đường dẫn raw tuyệt đối. Truyền hai thư mục tải về cho lệnh report.\n\nLệnh tái phân tích trong reproduce.json là thao tác chỉ đọc raw trên máy nghiên cứu gốc, dùng runtime và thư mục nguồn đã đóng băng. Thay duy nhất đích phân tích bằng thư mục mới chưa tồn tại. Nếu cần tái phân tích trên máy khác, giữ ánh xạ ngoài gói và dùng đường dẫn protocol/run-root cụ thể; không sửa các tệp protocol hoặc collection. Việc xác minh từng byte tải về dựa trên manifest tương đối không cần chạy mô phỏng.\n\nKhông có bước tự động khởi chạy mô phỏng, refit, tải thư viện hoặc ghi đè kết quả. Lệnh phân tích/renderer thất bại giữ nguyên thư mục dở dang. Các lần thử lại dùng đích mới. Đọc reproduce.json để xem runtime thực và nguồn gốc.\n'''
    (out/'README.vi.md').write_text(readme,encoding='utf-8')
    result=dict(schema='service-v2-report-v1',analysis_manifest_sha256=analysis_hash,engineering_manifest_sha256=engineering_hash,
        renderer_runtime=dict(executable=sys.executable,python=sys.version,source_sha256=sha256(__file__),argv=[str(Path(sys.executable).resolve()),str(Path(__file__).resolve()),
            '--analysis',str(Path(analysis_root).resolve()),'--engineering',str(Path(engineering_root).resolve()),'--output',str(out.resolve())]),
        font_path=str(font),font_sha256=sha256(font),synthetic_only=synthetic,
        files={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file()})
    write_json(out/'report.json',result)
    write_json(out/'artifact-inventory.json',dict(schema='service-v2-artifact-inventory-v1',files={p.relative_to(out).as_posix():sha256(p) for p in out.rglob('*') if p.is_file()}))
    return result


POLICY_ORDER=('hold','reactive','geometry','path-service-v1-reference','service-v2','service-v2-no-forecast',
              'service-v2-no-online-service-state','service-v2-no-motion-gate')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Report-only worker; no simulation dependencies')
    for argument in ('analysis','engineering','output'): parser.add_argument('--'+argument,type=Path,required=True)
    args=parser.parse_args()
    try: build_report(args.analysis,args.engineering,args.output)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr); raise SystemExit(1)
