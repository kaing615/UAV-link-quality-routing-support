"""Run frozen 18-pilot/720-formal grids in one exclusive bounded worker pool.

No overwrite/resume/skip. Failed attempts retain all raw outputs. The copied
source snapshot is the repository root; model paths resolve relative to it.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import threading

if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.control.bridge import make_launcher
from src.control.evaluation import runtime_versions,verify_build_manifest
from src.control.prevention.prediction import LearnedPredictor,sha256,strict_json
from src.control.path_service.study import collect_case,validate_protocol

ROOT=Path(__file__).resolve().parents[2]


def write(path,value,exclusive=False):
    with Path(path).open('x' if exclusive else 'w',encoding='utf-8') as stream: json.dump(value,stream,indent=2,allow_nan=False)


def inventory():
    paths=list((ROOT/'src/control').rglob('*.py'))
    paths+=list((ROOT/'simulation/ns3').glob('*.h'))+[ROOT/'simulation/ns3/uav-olsr-dataset.cc',ROOT/'simulation/ns3/CMakeLists.txt']
    paths+=[ROOT/p for p in ('scripts/analysis/run_path_service_study.py','scripts/analysis/evaluate_path_service.py','scripts/analysis/freeze_path_service_protocol.py','scripts/utils/build_recovery.sh')]
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(paths)}


def validate_seed_audit(protocol):
    spec=ROOT/'.superpowers/sdd/2026-11-11-routing-service'
    audit=strict_json(spec/'task-3-seed-preregistration-audit.json')
    expected=audit['pilot_scenarios'] if protocol['phase']=='pilot' else audit['formal_scenarios']
    if [(s['layout_seed'],s['radio_seed']) for s in protocol['scenarios']]!=[(s['layout_seed'],s['radio_seed']) for s in expected]: raise ValueError('Actual protocol differs from audited seed grid')
    if not audit['all_scenario_pairs_disjoint'] or audit['diagnostic_seed_overlap']: raise ValueError('Seed inventory overlap')
    return dict(audit_sha256=sha256(spec/'task-3-seed-preregistration-audit.json'),old_scenario_pairs_checked=audit['old_scenario_pairs_checked'],disjoint=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True); parser.add_argument('--phase',choices=['pilot','formal'],required=True)
    parser.add_argument('--binary',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-workers',type=int,default=4); parser.add_argument('--wsl'); parser.add_argument('--model',type=Path)
    parser.add_argument('--collection-lock',type=Path,help='One shared lock for both phases; default output parent/.path-service.lock')
    args=parser.parse_args(argv)
    if not 1<=args.max_workers<=4: raise ValueError('At most four simulations total; workers 1..4 required')
    if args.output.exists(): raise ValueError('Output already exists; use a fresh directory')
    protocol=validate_protocol(strict_json(args.protocol),args.protocol,args.phase)
    seed_audit=validate_seed_audit(protocol)
    contract_path=ROOT/'.superpowers/sdd/2026-11-11-routing-service/task-3-model-baseline-check.json'
    contract=strict_json(contract_path)
    for relative,check in contract['source_contract_checks_before_task3'].items():
        if sha256(ROOT/relative)!=check['frozen_selected_sha256']: raise ValueError('Frozen baseline/model contract source changed: '+relative)
    model_path=args.model or ROOT/protocol['model']
    if sha256(model_path)!=protocol['model_sha256']: raise ValueError('Frozen model hash mismatch')
    model=LearnedPredictor.load(model_path); model.validate_test_scenarios([(s['layout_seed'],s['radio_seed']) for s in protocol['scenarios']])
    binary=args.binary.resolve(); build=verify_build_manifest(binary)
    if not build: raise ValueError('Source/binary manifest required')
    launcher=make_launcher(binary,args.wsl)
    provenance=dict(schema='path-service-provenance-v1',command=[sys.executable,*sys.argv],source_hashes=inventory(),
                    protocol_sha256=sha256(args.protocol),model_sha256=sha256(model_path),binary_sha256=sha256(binary),build_manifest=build,
                    model_contract_baseline_sha256=sha256(contract_path),
                    versions=runtime_versions(),seed_audit=seed_audit,
                    environment={k:os.environ.get(k) for k in ('LD_LIBRARY_PATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    # Hash the actual runtime files, not merely a version label.
    runtime_paths=[]
    for directory in os.environ.get('LD_LIBRARY_PATH','').split(os.pathsep):
        if directory and Path(directory).is_dir(): runtime_paths+=list(Path(directory).glob('*.so*'))
    provenance['runtime_hashes']={str(p.resolve()):sha256(p.resolve()) for p in sorted(runtime_paths) if p.is_file()}
    # Keep protocol/method/model adjacent to outputs so downloaded raw artifacts
    # retain their own immutable run contract independently of source locations.
    lock=args.collection_lock or args.output.parent/'.path-service.lock'
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('x',encoding='utf-8') as stream: stream.write(json.dumps(dict(pid=os.getpid(),output=str(args.output.resolve()),phase=args.phase)))
    try:
        args.output.mkdir(parents=True,exist_ok=False)
        write(args.output/'protocol.json',protocol,True); write(args.output/'provenance.json',provenance,True)
        for name,source in [('method.json',args.protocol.parent/'method.json'),('freeze.json',args.protocol.parent/'freeze.json'),('predictor.json',model_path)]:
            with (args.output/name).open('xb') as stream:stream.write(source.read_bytes())
        rows=[dict(policy=p,**s,status='pending',case_dir=f'cases/{s["regime"]}/layout_{s["layout_seed"]}_radio_{s["radio_seed"]}/{p}') for s in protocol['scenarios'] for p in protocol['policies']]
        write(args.output/'manifest.json',rows,True)
        manifest_lock=threading.Lock()
        def task(index):
            row=rows[index]; case=args.output/row['case_dir']
            with manifest_lock:
                row['status']='running'; write(args.output/'manifest.json',rows)
            try:
                result=collect_case(scenario=row,policy=row['policy'],protocol=protocol,launcher=launcher,predictor=model,output=case,provenance=provenance)
                result['artifact_hashes']={p.relative_to(args.output).as_posix():sha256(p) for p in sorted(case.rglob('*')) if p.is_file()}
                return index,result
            except Exception as exc:
                return index,dict(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures={}
            for index,row in enumerate(rows):
                futures[pool.submit(task,index)]=index
            for future in as_completed(futures):
                index,result=future.result()
                with manifest_lock:
                    rows[index].update(result)
                    rows[index]['status']='ok' if result['status']=='ok' else 'failed'
                    write(args.output/'manifest.json',rows)
                print(f'{index+1}/{len(rows)} {rows[index]["policy"]} {rows[index]["status"]}',flush=True)
        summary=dict(expected_episodes=protocol['expected_episodes'],attempted_episodes=len(rows),ok=sum(r['status']=='ok' for r in rows),failed=sum(r['status']=='failed' for r in rows),complete=all(r['status']=='ok' for r in rows))
        write(args.output/'status.json',summary,True)
        return summary
    finally: lock.unlink()

if __name__=='__main__':
    result=main()
    if not result['complete']: raise SystemExit(2)
