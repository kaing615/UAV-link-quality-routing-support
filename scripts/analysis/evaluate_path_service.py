"""Validate raw artifacts and evaluate prespecified whole paired scenarios.

No simulator, fitting, subgroup search or packet/window bootstrap. Output must
be fresh. Primary endpoint uses final actual application sink receives/offers.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.control.prevention.prediction import strict_json,sha256
from scripts.analysis.run_path_service_study import write
from src.control.path_service.study import reconcile_episode
from src.control.prevention.config import PreventionConfig
from src.control.prevention.evaluation import episode_metrics

METRICS=('actual_udp_pdr','total_relay_travel','source_dest_disconnected_fraction','disconnected_fraction','olsr_unusable_fraction','mean_delay_ms')


def verify_protocol_representation(path,source_sha256):
    """Bind copied bytes to original hash, allowing only LF/CRLF conversion.

    Do not parse/reserialize JSON or use semantic equality as a fallback. Each
    accepted representation must reconstruct the exact frozen byte hash.
    """
    raw=Path(path).read_bytes()
    copied_sha256=hashlib.sha256(raw).hexdigest()
    if copied_sha256==source_sha256:
        mode='exact_bytes'
    else:
        lf=raw.replace(b'\r\n',b'\n')
        if b'\r' in lf: raise ValueError('Run protocol/provenance hash mismatch: unsupported lone CR')
        candidates=(lf,lf.replace(b'\n',b'\r\n'))
        if not any(hashlib.sha256(candidate).hexdigest()==source_sha256 for candidate in candidates):
            raise ValueError('Run protocol/provenance hash mismatch: not an LF/CRLF-only representation change')
        mode='verified_LF_CRLF_only'
    return dict(original_frozen_protocol_sha256=source_sha256,copied_run_protocol_sha256=copied_sha256,
                validation=mode,semantics='Exact source byte hash or exact reconstruction changing only LF/CRLF bytes; no JSON reserialization equivalence accepted')


def evaluate(records,protocol):
    expected=Counter((p,s['layout_seed'],s['radio_seed']) for s in protocol['scenarios'] for p in protocol['policies'])
    actual=Counter((r['policy'],r['layout_seed'],r['radio_seed']) for r in records)
    if actual!=expected: raise ValueError('Missing/unexpected/duplicate manifest cells')
    lookup={(r['policy'],r['layout_seed'],r['radio_seed']):r for r in records}
    regimes=defaultdict(list)
    for s in protocol['scenarios']: regimes[s['regime']].append((s['layout_seed'],s['radio_seed']))
    settings=protocol['evaluation']; rng=np.random.default_rng(settings['rng_seed']); resamples=settings['bootstrap_resamples']
    comparisons={}; differences=[]; sensitivity={}
    for reference in [p for p in protocol['policies'] if p!='path-service']:
        per_regime={}; pooled={k:[] for k in METRICS}; pooled_boot={k:[] for k in METRICS}; bounds=[]
        for regime,scenarios in regimes.items():
            pairs=[(s,lookup[('path-service',*s)],lookup[(reference,*s)]) for s in scenarios]
            per_regime[regime]={}
            lo,hi=0.,0.
            for s,new,old in pairs:
                valid=new['status']=='ok' and old['status']=='ok'
                differences.append(dict(reference=reference,regime=regime,layout_seed=s[0],radio_seed=s[1],complete_pair=valid,
                                        differences={k:new.get(k)-old.get(k) if valid and new.get(k) is not None and old.get(k) is not None else None for k in METRICS}))
                if valid: low=high=new['actual_udp_pdr']-old['actual_udp_pdr']
                elif new['status']=='ok': low,high=new['actual_udp_pdr']-1.,new['actual_udp_pdr']
                elif old['status']=='ok': low,high=-old['actual_udp_pdr'],1.-old['actual_udp_pdr']
                else: low,high=-1.,1.
                lo+=low/len(scenarios); hi+=high/len(scenarios)
            bounds.append([lo,hi])
            for key in METRICS:
                delta=np.asarray([n[key]-o[key] for _,n,o in pairs if n['status']=='ok' and o['status']=='ok' and n.get(key) is not None and o.get(key) is not None],float)
                if len(delta):
                    indices=rng.integers(0,len(delta),size=(resamples,len(delta)))
                    boot=delta[indices].mean(axis=1); interval=np.quantile(boot,[.025,.975]).tolist()
                    pooled[key].append(float(delta.mean())); pooled_boot[key].append(boot)
                    per_regime[regime][key]=dict(complete_pairs=len(delta),expected_pairs=len(scenarios),mean_difference=float(delta.mean()),scenario_bootstrap_95pct_ci=interval)
                else: per_regime[regime][key]=dict(complete_pairs=0,expected_pairs=len(scenarios),mean_difference=None,scenario_bootstrap_95pct_ci=None)
        comparisons[reference]=dict(regimes=per_regime,pooled_equal_regime={k:dict(mean_difference=float(np.mean(pooled[k])) if pooled[k] else None,
                scenario_bootstrap_95pct_ci=np.quantile(np.mean(pooled_boot[k],axis=0),[.025,.975]).tolist() if pooled_boot[k] else None,
                represented_regimes=len(pooled[k]),expected_regimes=len(regimes)) for k in METRICS})
        sensitivity[reference]=dict(endpoint='actual_udp_pdr',assumption='each failed endpoint may lie anywhere in [0,1]; no zero-PDR imputation',
                                    pooled_equal_regime_mean_difference_bounds=np.mean(bounds,axis=0).tolist())
    means={p:{k:float(np.mean([r[k] for r in records if r['policy']==p and r['status']=='ok' and r.get(k) is not None]))
               if any(r['policy']==p and r['status']=='ok' and r.get(k) is not None for r in records) else None for k in METRICS} for p in protocol['policies']}
    failures=[r for r in records if r['status']!='ok']
    return dict(schema='path-service-evaluation-v1',complete_primary_grid=not failures,attempted=len(records),failures=failures,
                means=means,comparisons= comparisons,scenario_differences=differences,failure_sensitivity=sensitivity,
                settings=settings,interpretation='Descriptive paired scenario percentile intervals; small regime samples, correlated comparisons, no multiplicity-adjusted superiority or causal prevention claim')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists(): raise ValueError('Evaluation output must be fresh')
    records=strict_json(args.run/'manifest.json'); protocol=strict_json(args.run/'protocol.json')
    provenance=strict_json(args.run/'provenance.json')
    freeze=strict_json(args.run/'freeze.json')
    if provenance['protocol_sha256']!=freeze['files'][protocol['phase']+'.json']:
        raise ValueError('Protocol source provenance differs from frozen source inventory')
    protocol_provenance=verify_protocol_representation(args.run/'protocol.json',provenance['protocol_sha256'])
    for row in records:
        for relative,digest in row.get('artifact_hashes',{}).items():
            path=(args.run/relative).resolve()
            if not path.is_relative_to(args.run.resolve()) or sha256(path)!=digest: raise ValueError('Raw artifact hash mismatch: '+relative)
        if row['status']=='ok' and not row.get('artifact_hashes'): raise ValueError('Missing successful-case artifact provenance')
        if row['status']=='ok':
            case=(args.run/row['case_dir']).resolve()
            if not case.is_relative_to(args.run.resolve()): raise ValueError('Unsafe case directory')
            paths=list(case.rglob('transitions.jsonl'))
            if len(paths)!=1: raise ValueError('Expected exactly one raw episode per case')
            episode=paths[0].parent
            config=PreventionConfig(**strict_json(episode/'config.json')['config'])
            frames=[json.loads(l)['frame'] for l in paths[0].read_text().splitlines()]
            decisions=[json.loads(l) for l in next(case.rglob('decisions.jsonl')).read_text().splitlines()]
            recomputed=episode_metrics(frames,config,episode,decisions)
            recomputed.update(reconcile_episode(episode,config))
            for key,value in recomputed.items():
                if row.get(key)!=value: raise ValueError('Manifest/raw metric mismatch: '+key)
    result=evaluate(records,protocol)
    result['protocol_provenance']=protocol_provenance
    result['input_hashes']={p.name:sha256(p) for p in [args.run/'manifest.json',args.run/'protocol.json',args.run/'provenance.json']}
    args.output.mkdir(parents=True,exist_ok=False); write(args.output/'evaluation.json',result,True)
    text=['# Báo cáo độc lập: điều khiển UAV theo đường đi và dịch vụ','',
          f'Giai đoạn: {protocol["phase"]}; số lượt đã thử: {len(records)}; lỗi kỹ thuật: {len(result["failures"])}.',
          'Đây là bộ lập kế hoạch xác định kết hợp bộ dự báo liên kết giám sát đã đóng băng; không huấn luyện PPO mới.',
          'Chỉ số chính: số gói UDP thực nhận tại ứng dụng đích / số gói ứng dụng nguồn chào gửi trên toàn thời gian cấu hình. Độ trễ chỉ tính trên gói đã nhận, không có gói nhận thì để trống.',
          '', '| Chính sách | PDR ứng dụng | Tổng quãng đường relay (m) |','|---|---:|---:|']
    for p,values in result['means'].items(): text.append(f'| {p} | {values["actual_udp_pdr"]} | {values["total_relay_travel"]} |')
    text+=['','## Giới hạn và diễn giải','',
           'Các khoảng mô tả được bootstrap 10.000 lần theo toàn bộ kịch bản ghép cặp, seed 91011, phân tầng và cho trọng số bằng nhau giữa chế độ. Bốn chế độ được ấn định trước; không chọn nhóm thuận lợi sau khi xem kết quả.',
           'PDR probe, đường OLSR sử dụng được và tỷ lệ đến/chào gửi gần đây không thay thế PDR UDP. Tỷ lệ gần đây có thể lớn hơn 1 do gói đến muộn.',
           'Tổng rủi ro biên liên kết là điểm heuristic; không phải xác suất mất đường đã hiệu chuẩn hay xác suất phản thực của hành động. Chuyển miền hình học compact và quỹ đạo do chính sách tạo ra có thể làm thay đổi hiệu chuẩn.',
           'Chuỗi relay dự kiến không cài đặt đường OLSR; relay không được chọn vẫn phát sóng và có thể tham gia đường thực. Chuyển động không bảo đảm sửa lỗi OLSR/ARP/chuyển tiếp.',
           'Hành động trước một lần đứt kết nối quan sát được không chứng minh đã ngăn một lần đứt kết nối. Các hành động sau suy giảm dịch vụ được ghi là khôi phục.',
           'Nếu còn lỗi kỹ thuật, phân tích chính là chưa đầy đủ; xem danh sách lỗi và cận nhạy cảm riêng trong evaluation.json. Không thay lỗi kỹ thuật bằng PDR bằng 0.',
           'Chưa đủ cơ sở tuyên bố tính mới, ưu thế thống kê sau nhiều so sánh hoặc sẵn sàng công bố. Cần kết quả đầy đủ và đánh giá độc lập.',
           '', 'Chi tiết từng cặp kịch bản, khoảng PDR/quãng đường/outage, thất bại và hash tái lập: `evaluation.json`.']
    (args.output/'BAO_CAO.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    return result

if __name__=='__main__': main()
