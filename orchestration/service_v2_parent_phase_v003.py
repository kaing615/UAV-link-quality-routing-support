"""Parent-only orchestration with separate immutable adoption/launch gates.

Importing this module has no operational effect. Gate records are parent-owned;
this program verifies them and never creates an adoption or approval.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


def api():
    from src.control.service_v2.protocol import strict_json, sha256, load_phase, verify_runtime, write_json
    return strict_json,sha256,load_phase,verify_runtime,write_json


def raw_reference(record):
    _,sha256,*_=api()
    if not isinstance(record,dict) or set(record)!={'path','sha256'}:
        raise ValueError('Expected exact immutable path/hash reference')
    path=Path(record['path'])
    if not path.is_absolute() or not path.is_file() or sha256(path)!=record['sha256']:
        raise ValueError('Missing or mismatched immutable record: '+str(path))
    return path


def reference(record):
    strict_json,*_=api()
    path=raw_reference(record)
    return path,strict_json(path)


def identities(phase):
    _,sha256,*_=api()
    package=Path(phase['_path']).parent
    return dict(phase=phase['phase'],protocol_sha256=sha256(phase['_path']),
        freeze_sha256=phase['_freeze_sha256'],source_manifest_sha256=sha256(package/'source-manifest.json'),
        spec_sha256=phase['spec_sha256'],method_sha256=phase['method_sha256'],
        new_model_sha256=phase['new_model'],reference_model_sha256=phase['reference_model'],
        seed_plan_sha256=phase['audited_seed_plan_sha256'],
        inherited_pilot_freeze_sha256=phase['inherited_pilot_freeze_sha256'])


def artifact(record,filename):
    strict_json,sha256,*_=api()
    path,value=reference(record)
    root=path.parent
    files=value.get('files')
    if not isinstance(files,dict) or filename not in files:
        raise ValueError('Missing immutable artifact file: '+filename)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual!=set(files)|{path.name}:
        raise ValueError('Artifact file inventory mismatch')
    for name,digest in files.items():
        item=(root/name).resolve(strict=True)
        if not item.is_relative_to(root.resolve()) or sha256(item)!=digest:
            raise ValueError('Artifact data identity mismatch')
    return strict_json(root/filename)


def accepted_review(record,scope,kind,bindings):
    _,value=reference(record)
    expected={'schema','scope','review_kind','accepted','critical','important','bindings','report'}
    if (set(value)!=expected or value['schema']!='service-v2-independent-review-acceptance-v1'
            or value['scope']!=scope or value['review_kind']!=kind or value['accepted'] is not True
            or type(value['critical']) is not int or value['critical']!=0
            or type(value['important']) is not int or value['important']!=0 or value['bindings']!=bindings):
        raise ValueError('Independent review not accepted for these exact artifacts')
    raw_reference(value['report'])
    return value


def verify_adoption(path,study,source):
    strict_json,sha256,load_phase,verify_runtime,_=api()
    value=strict_json(path)
    expected={'schema','adopted','adopted_by','identities','artifact_bindings','references'}
    if (set(value)!=expected or value['schema']!='service-v2-task4-parent-adoption-v1'
            or value['adopted'] is not True or value['adopted_by']!='parent'):
        raise ValueError('Explicit immutable parent Task 4 adoption required')
    refs=value['references']
    required={'pilot_protocol','development_manifest','engineering_manifest','train_validation',
        'calibration_validation','development_validation','pilot_validation','specification_review','quality_review','formal_seed_plan'}
    if not isinstance(refs,dict) or set(refs)!=required:
        raise ValueError('Incomplete adopted Task 4 evidence references')
    paths={name:reference(record)[0] for name,record in refs.items()}
    phase=load_phase(paths['pilot_protocol'])
    if phase['phase']!='pilot' or phase['expected_episodes']!=48 or value['identities']!=identities(phase):
        raise ValueError('Adoption pilot identity mismatch')
    verify_runtime(phase,source)
    if sha256(study/'model/predictor.json')!=phase['new_model']:
        raise ValueError('Adopted fitted model no longer matches current model')
    bindings=dict(pilot=identities(phase),development_manifest_sha256=sha256(paths['development_manifest']),
                  engineering_manifest_sha256=sha256(paths['engineering_manifest']))
    if value['artifact_bindings']!=bindings:
        raise ValueError('Adoption artifact bindings mismatch')
    development=artifact(refs['development_manifest'],'summary.json')
    engineering=artifact(refs['engineering_manifest'],'engineering.json')
    candidate=engineering['pilot_gate_candidate']
    if (development.get('complete') is not True or development.get('model_gates_passed') is not True
            or candidate.get('engineering_complete') is not True or candidate.get('engineering_checks_passed') is not True
            or candidate.get('healthy_start_240m_all_six_hold_by20') is not True
            or candidate.get('engineering_issues')!=0
            or candidate.get('development_audit_manifest_sha256')!=bindings['development_manifest_sha256']
            or candidate.get('accepted_identities',{}).get('freeze_sha256')!=phase['_freeze_sha256']):
        raise ValueError('Adopted development/model/pilot fixture or engineering gate failed')
    counts={}
    for name,total in (('train',96),('calibration',48),('development',32),('pilot',48)):
        validation=strict_json(paths[name+'_validation'])
        if (validation.get('schema')!='service-v2-validation-v1' or validation.get('phase')!=name
                or validation.get('complete') is not True or validation.get('planned_cells')!=total
                or validation.get('ok_cells')!=total or len(validation.get('rows',[]))!=total):
            raise ValueError('Adoption requires exact complete 176+48 inventories')
        rows=validation['rows']
        if any(r.get('status')!='ok' for r in rows) or len({r.get('cell_id') for r in rows})!=total:
            raise ValueError('Adopted complete inventory contains failed or duplicate cells')
        expected_freeze=(candidate['development_176_inventory'][name]['freeze_sha256'] if name!='pilot' else phase['_freeze_sha256'])
        if validation.get('freeze_sha256')!=expected_freeze:
            raise ValueError('Adopted validation freeze identity mismatch')
        counts[name]=total
    for name,kind in (('specification_review','specification'),('quality_review','quality')):
        accepted_review(refs[name],'task4',kind,bindings)
    formal_plan=source/'reports/uav_service_prevention_v2_2026-11-12/seed-plans/formal.json'
    if paths['formal_seed_plan'].resolve()!=formal_plan.resolve():
        raise ValueError('Adoption formal seed plan path differs from intended freeze input')
    return dict(adoption_path=str(Path(path).resolve()),adoption_sha256=sha256(path),
        verified_pilot_identities=identities(phase),artifact_bindings=bindings,inventory_counts=counts,
        formal_seed_plan_sha256=sha256(formal_plan),references=refs)


def verify_formal_inheritance(phase,adoption):
    current=identities(phase)
    prior=adoption['verified_pilot_identities']
    if current['phase']!='formal' or phase['expected_episodes']!=384:
        raise ValueError('Expected full formal package')
    for key in ('source_manifest_sha256','spec_sha256','method_sha256','new_model_sha256','reference_model_sha256'):
        if current[key]!=prior[key]:
            raise ValueError('Formal package differs from adopted pilot: '+key)
    if (current['inherited_pilot_freeze_sha256']!=prior['freeze_sha256']
            or current['seed_plan_sha256']!=adoption['formal_seed_plan_sha256']):
        raise ValueError('Formal inheritance/seed plan differs from parent adoption')
    return current


def verify_launch(path,phase,adoption):
    strict_json,sha256,*_=api()
    value=strict_json(path)
    expected={'schema','approved','approved_by','adoption_sha256','identities','prelaunch_review'}
    if (set(value)!=expected or value['schema']!='service-v2-formal-launch-approval-v1'
            or value['approved'] is not True or value['approved_by']!='parent'):
        raise ValueError('Explicit immutable parent formal prelaunch approval required')
    current=verify_formal_inheritance(phase,adoption)
    if value['adoption_sha256']!=adoption['adoption_sha256'] or value['identities']!=current:
        raise ValueError('Formal approval does not bind this actual formal package/adoption')
    accepted_review(value['prelaunch_review'],'formal-prelaunch','prelaunch',
        dict(formal=current,adoption_sha256=adoption['adoption_sha256']))
    return dict(approval_path=str(Path(path).resolve()),approval_sha256=sha256(path),
        adoption_sha256=adoption['adoption_sha256'],verified_formal_identities=current,
        prelaunch_review=value['prelaunch_review'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('pilot','formal'))
    parser.add_argument('operation',choices=('prepare','execute'))
    parser.add_argument('--root',type=Path,default=Path('/home/sac/uav-service/2026-11-12-service-prevention-v2'))
    args=parser.parse_args()
    root=args.root.resolve(strict=True)
    source=root/'source-runtime-v002'
    study=root/'reports/uav_service_prevention_v2_2026-11-12'
    sys.path.insert(0,str(source))
    strict_json,sha256,load_phase,verify_runtime,write_json=api()
    phase=args.phase
    evidence=root/'evidence'/(phase+'-'+args.operation+'-v003')
    evidence.mkdir(parents=True,exist_ok=False)
    protocol_path=study/'frozen'/phase/'protocol.json'
    environment=dict(os.environ,LD_LIBRARY_PATH='/home/sac/uav-service/2026-11-11-path-control/runtime/lib',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def run(name,argv,cwd=source):
        started=time.time()
        with (evidence/(name+'.stdout.log')).open('xb') as stdout:
            with (evidence/(name+'.stderr.log')).open('xb') as stderr:
                child=subprocess.run(argv,cwd=cwd,env=environment,stdout=stdout,stderr=stderr,check=False)
        write_json(evidence/(name+'.json'),dict(argv=argv,cwd=str(cwd),direct_exit=child.returncode,
            wall_seconds=time.time()-started,finished_unix=time.time(),environment={k:environment[k] for k in
            ('LD_LIBRARY_PATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}))
        if child.returncode:
            raise RuntimeError(name+' failed; evidence retained')
    try:
        development_marker=root/'evidence/development-resume-v003/completed.json'
        completed=strict_json(development_marker)
        if completed.get('status')!='passed' or completed.get('planned_episodes')!=176:
            raise ValueError('Complete 176-episode development marker required')
        write_json(evidence/'prerequisite.json',dict(path=str(development_marker),sha256=sha256(development_marker)))
        adoption=None
        if phase=='formal':
            adoption=verify_adoption(study/'engineering/task4-parent-adoption-v001.json',study,source)
            write_json(evidence/'verified-task4-adoption.json',adoption)
        if args.operation=='prepare':
            run('freeze',[sys.executable,'-m','src.control.service_v2.cli','freeze','--phase',phase,
                '--seed-plan',str(source/f'reports/uav_service_prevention_v2_2026-11-12/seed-plans/{phase}.json'),
                '--design',str(source/'docs/superpowers/specs/2026-11-12-service-prevention-v2-design.md'),
                '--output',str(protocol_path.parent),'--model',str(study/'model/predictor.json'),
                '--reference-model',str(source/'reports/uav_prevention_pilot_2026-11-11/predictor.json')])
            protocol=load_phase(protocol_path)
            if adoption is not None:
                write_json(evidence/'verified-formal-inheritance.json',verify_formal_inheritance(protocol,adoption))
            probe="""from pathlib import Path
import sys
from src.control.service_v2.protocol import load_phase, verify_runtime, ROOT
p = load_phase(sys.argv[1])
if p['run_argv'][0] != sys.executable or sys.prefix == sys.base_prefix or not Path(sys.executable).is_symlink():
    raise ValueError('Frozen invocation/venv/symlink check failed')
verify_runtime(p, ROOT)
print({'invocation': sys.executable, 'passed': True})
"""
            run('actual-frozen-runtime-probe',[protocol['run_argv'][0],'-c',probe,str(protocol_path)],protocol_path.parent/'source')
            run('queue-dry-run',[sys.executable,str(root/'orchestration/service_v2_parent_queue_v002.py'),
                '--protocol',str(protocol_path),'--run-root',str(study/'runs'/phase),
                '--evidence',str(root/'evidence'/('queue-dry-run-'+phase+'-v003')),
                '--lock',str(root/'global-simulations.lock'),'--dry-run'])
        else:
            preparation=root/'evidence'/(phase+'-prepare-v003')/'completed.json'
            prepared=strict_json(preparation)
            if prepared!={'phase':phase,'operation':'prepare','status':'passed'}:
                raise ValueError('Successful matching v003 preparation required')
            protocol=load_phase(protocol_path)
            verify_runtime(protocol,source)
            write_json(evidence/'verified-preparation.json',dict(path=str(preparation),sha256=sha256(preparation),identities=identities(protocol)))
            if phase=='formal':
                # Re-read adoption and package immediately before queue dispatch.
                adoption=verify_adoption(study/'engineering/task4-parent-adoption-v001.json',study,source)
                protocol=load_phase(protocol_path)
                launch=verify_launch(study/'engineering/formal-launch-approval-v001.json',protocol,adoption)
                write_json(evidence/'verified-formal-launch.json',launch)
            run('queue',[sys.executable,str(root/'orchestration/service_v2_parent_queue_v002.py'),
                '--protocol',str(protocol_path),'--run-root',str(study/'runs'/phase),
                '--evidence',str(root/'evidence'/('queue-'+phase+'-v003')),'--lock',str(root/'global-simulations.lock')])
            (study/'validation').mkdir(parents=True,exist_ok=True)
            run('validate',[sys.executable,'-m','src.control.service_v2.cli','validate',
                '--protocol',str(protocol_path),'--run-root',str(study/'runs'/phase),
                '--output',str(study/'validation'/(phase+'.json'))])
            run('analyze',[sys.executable,'-m','src.control.service_v2.cli','analyze',
                '--protocol',str(protocol_path),'--run-root',str(study/'runs'/phase),'--output',str(study/(phase+'-analysis'))])
        write_json(evidence/'completed.json',dict(phase=phase,operation=args.operation,status='passed'))
    except BaseException as error:
        write_json(evidence/'failed.json',dict(error=f'{type(error).__name__}: {error}'))
        raise


if __name__=='__main__':
    main()
