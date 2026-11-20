"""Exclusive, hash-verified study packages. No simulator launch occurs here.

Seed plans contain ``phases: {phase: [scenario, ...]}``; each scenario has id,
regime (n4/n5), context (static/gradual), layout_seed and radio_seed. Runtime
contains a literal launcher argv, ns3_identity and compiler_identity. Audit
adds historical file hashes and identifier inventories before freeze.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import sys
from uuid import uuid4

from .config import ADOPTED_SPEC_SHA256, POLICIES, ServiceV2Config

REFERENCE_SHA256 = '3b74301ccb0e659ba6d2b593250decabd0f317386adfa4cc5d5b4e5174bacae8'
COUNTS = {'train': (12, 12), 'calibration': (6, 6), 'development': (4, 4),
          'pilot': (1, 2), 'formal': (4, 20)}  # static, gradual per regime
ROOT = Path(__file__).resolve().parents[3]

ACCEPTED_REFERENCE_SOURCES = {'src/control/bridge.py': '65deebc1460506b961d36077b0d9621b8549f204040eaa85dd6ebc6343126e2b', 'src/control/path_service/__init__.py': '9fb18fcfd67fddf6de5560f8ab4cd8c2b57c7e63e74b3a6704eef4b50c444539', 'src/control/path_service/config.py': 'e344232184fb26eb999e491b6eca7ec2cbebfd9d9c15815b5c6c2c35a7679490', 'src/control/path_service/controller.py': '3d9089c5081c89248ca0616dad691323dd4384172b22699d49eb2e7207696bc6', 'src/control/path_service/observation.py': '9e76194f01165aaead253d611b0a828c1eef84ce4e03676914a7ec3c19847f52', 'src/control/path_service/planner.py': '7a2a3d07ffeefb93a955ee2db6c9d723f92091bad6a1c0380ca56bc19fde0299', 'src/control/path_service/risk.py': '1572bcde037413be20d58ee5951a818ad66d2cf6b6a6202c76b04ebda62a4685', 'src/control/path_service/study.py': 'c26aaca1de26944358bf567b5d1d28671764ab98031d2c123029d9f723d901fb', 'src/control/prevention/__init__.py': '1e70120097963e6ccb6308142f92433af8de09e034249cdf9a4d84bb541f4cbb', 'src/control/prevention/config.py': '9aa7dff2cf4b8fcc86d093c1e10006330bb9db34a8a3bc771239648e21435fc8', 'src/control/prevention/controller.py': '92ad1311661e9ebab68b86446d2d68c7796ba3159b37b0870fdc9ebe14e0ea8b', 'src/control/prevention/environment.py': '54b56236c8b535d01d25ac62db96480beca23a1e3d0ac88219adf84f31a7f43b', 'src/control/prevention/evaluation.py': '273eebe4a2cd5fc2282d8fb8700b031ab5945c482546387d8038b815878e7a3b', 'src/control/prevention/features.py': '05410d037ba60d70700cf1b9ba099e555e70ee5919b8b2aea7da2f2e316d27c1', 'src/control/prevention/prediction.py': '7c3bf5aba1135b367bbd6ee26f6408f86bda510f174000681a47bc9af2eb88f6'}



def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_loads(text):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError('Duplicate JSON key: '+key)
            out[key] = value
        return out
    def reject(value): raise ValueError('Nonfinite JSON: '+value)
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=reject)
    # 1e999 parses as infinity without parse_constant.
    json.dumps(value, allow_nan=False)
    return value


def strict_json(path):
    return strict_loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')
    path = Path(path)
    # Link an already completed same-filesystem temporary file atomically.
    # os.link refuses existing destinations on both Windows and POSIX.
    temporary=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    with temporary.open('xb') as stream: stream.write(raw)
    try: os.link(temporary,path)
    finally: temporary.unlink()


def safe_path(root, name):
    if not isinstance(name, str) or not name or '\\' in name: raise ValueError('Invalid relative path')
    p = Path(name)
    if p.is_absolute() or '..' in p.parts: raise ValueError('Path escape')
    target = (Path(root)/p).resolve()
    if not target.is_relative_to(Path(root).resolve()): raise ValueError('Path escape')
    return target


def reserve(path):
    path = Path(path)
    if path.exists() or path.is_symlink(): raise FileExistsError(str(path))
    path.mkdir(parents=True, exist_ok=False)
    return path


def _seeds(value, layout, radio):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ('layout_seed', 'layoutSeed', 'recoveryLayoutSeed') and type(child) is int: layout.add(child)
            elif key in ('radio_seed', 'radioSeed') and type(child) is int: radio.add(child)
            elif key=='seed' and type(child) is int:
                # Historical single-seed runs may use the same identifier for
                # layout and radio; conservatively reserve it in both spaces.
                layout.add(child); radio.add(child)
            elif key in ('layout_seeds','radio_seeds','seeds') and isinstance(child,list):
                for seed in child:
                    if type(seed) is int:
                        if key!='radio_seeds': layout.add(seed)
                        if key!='layout_seeds': radio.add(seed)
            elif key in ('scenarios', 'train_scenarios', 'calibration_scenarios', 'test_scenarios',
                         'train','calibration','development','pilot','formal','test') and isinstance(child, list):
                for pair in child:
                    if isinstance(pair, list) and len(pair)==2 and all(type(x) is int for x in pair):
                        layout.add(pair[0]); radio.add(pair[1])
            _seeds(child, layout, radio)
    elif isinstance(value, list):
        for child in value: _seeds(child, layout, radio)


def validate_seed_plan(plan):
    if plan.get('schema') != 'service-v2-seed-plan-v1': raise ValueError('Seed plan schema')
    phases = plan.get('phases')
    if not isinstance(phases, dict) or not phases or set(phases)-COUNTS.keys(): raise ValueError('Seed phases')
    seen = {'layout_seed': {}, 'radio_seed': {}}
    for phase, scenarios in phases.items():
        ids, groups = set(), {}
        if not isinstance(scenarios, list): raise ValueError('Literal scenario inventory required')
        for s in scenarios:
            if set(s) != {'id','regime','context','layout_seed','radio_seed'}: raise ValueError('Scenario fields')
            if not isinstance(s['id'],str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*',s['id']) or s['id'] in ids:
                raise ValueError('Duplicate/unsafe scenario id')
            ids.add(s['id'])
            if s['regime'] not in ('n4','n5') or s['context'] not in ('static','gradual'): raise ValueError('Scenario regime/context')
            pair=(s['layout_seed'],s['radio_seed'])
            for key in seen:
                seed=s[key]
                if type(seed) is not int or not 0<=seed<2**31: raise ValueError('Seed range')
                identity=(phase,s['regime'],pair)
                if seed in seen[key] and seen[key][seed]!=identity: raise ValueError('Seed identifier collision: '+key)
                seen[key][seed]=identity
            key=(s['regime'],pair)
            contexts=groups.setdefault(key,[]); contexts.append(s['context'])
            if len(contexts)!=len(set(contexts)): raise ValueError('Duplicate scenario')
        for regime in ('n4','n5'):
            for context, expected in zip(('static','gradual'),COUNTS[phase]):
                if sum(s['regime']==regime and s['context']==context for s in scenarios)!=expected:
                    raise ValueError('Wrong phase cardinality')
        if phase in ('train','calibration','development'):
            if any(sorted(contexts)!=['gradual','static'] for contexts in groups.values()): raise ValueError('Within-split pairing')
        elif any(len(contexts)!=1 for contexts in groups.values()): raise ValueError('Static/gradual seed reuse')
    runtime=plan.get('runtime',{})
    if (not isinstance(runtime.get('launcher'),list) or not runtime['launcher']
            or any(not isinstance(x,str) or not x for x in runtime['launcher'])
            or not runtime.get('ns3_identity') or not runtime.get('compiler_identity')): raise ValueError('Explicit runtime identities/argv required')
    return plan


def audit_seed_inventory(candidate, historical_root):
    validate_seed_plan(candidate)
    root=Path(historical_root).resolve()
    if not root.is_dir(): raise ValueError('Historical root missing')
    layout,radio=set([940001,940002,940003]),set([941001,941002,941003])
    files={}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix not in ('.json','.jsonl'): continue
        files[p.relative_to(root).as_posix()]=sha256(p)
        # A malformed historical candidate must not quietly disappear from audit.
        values=[strict_loads(line) for line in p.read_text(encoding='utf-8').splitlines() if line.strip()] if p.suffix=='.jsonl' else [strict_json(p)]
        for value in values: _seeds(value,layout,radio)
    for scenarios in candidate['phases'].values():
        for s in scenarios:
            if s['layout_seed'] in layout or s['radio_seed'] in radio: raise ValueError('Historical seed collision')
    result=strict_loads(json.dumps(candidate,allow_nan=False))
    evidence=dict(root=str(root),files=files,layout_seeds=sorted(layout),radio_seeds=sorted(radio),
        inventory_sha256=_canonical_hash(candidate['phases']),
        provenance=dict(protocol_source_sha256=sha256(__file__),python=sys.version,executable=sys.executable,
            host=platform.platform(),argv=list(sys.argv),cwd=str(Path.cwd()),audited_at=datetime.now(timezone.utc).isoformat()))
    result['historical_audit']=dict(evidence,evidence_sha256=_canonical_hash(evidence))
    return result


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()


def _validate_audit(plan):
    audit=plan.get('historical_audit')
    fields={'root','files','layout_seeds','radio_seeds','inventory_sha256','provenance','evidence_sha256'}
    if not isinstance(audit,dict) or set(audit)!=fields: raise ValueError('Detached historical audit evidence required')
    evidence={k:v for k,v in audit.items() if k!='evidence_sha256'}
    if audit['evidence_sha256']!=_canonical_hash(evidence) or audit['inventory_sha256']!=_canonical_hash(plan['phases']):
        raise ValueError('Historical audit integrity/inventory binding mismatch')
    if any(not isinstance(name,str) or not re.fullmatch('[0-9a-f]{64}',digest) for name,digest in audit['files'].items()):
        raise ValueError('Historical audit file hashes')
    for key in ('layout_seeds','radio_seeds'):
        values=audit[key]
        if not isinstance(values,list) or any(type(v) is not int or not 0<=v<2**31 for v in values) or values!=sorted(set(values)):
            raise ValueError('Historical audit seed inventory')
    return audit


def scenario_config(s):
    from ..prevention.config import PreventionConfig
    n=int(s['regime'][1:])
    return PreventionConfig(num_uavs=n,destination=n-1,steps=240,x_max=1800.,
        comm_range=240.,initial_spacing_fraction=.15 if n==4 else .10,
        mission_speed=1.,mission_start_step=30 if s['context']=='gradual' else 1000)


def source_paths(root):
    paths=list((root/'src').rglob('*.py'))
    paths += list((root/'scripts').rglob('*.py'))
    paths += list((root/'simulation/ns3').glob('*.h'))+list((root/'simulation/ns3').glob('*.cc'))
    return sorted(p for p in paths if '__pycache__' not in p.parts)


def verify_source_bytes(raw, entry):
    digest=hashlib.sha256(raw).hexdigest()
    if digest!=entry['executed_sha256']: raise ValueError('Executed source mismatch')
    if entry['transformation']=='identity':
        if digest!=entry['original_sha256']: raise ValueError('Undeclared normalization')
    elif entry['transformation']=='crlf_to_lf_only':
        if b'\r\n' in raw or hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()!=entry['normalized_sha256']:
            raise ValueError('Invalid named CRLF transformation')
    else: raise ValueError('Unsupported source transformation')


def runtime_identity(declared):
    import gymnasium
    import numpy
    import site
    import sysconfig
    # A virtualenv's Python may be a symlink to the base interpreter. Preserve
    # its invocation path: dereferencing it changes Python's environment.
    if not sys.executable: raise ValueError('Python invocation path unavailable')
    executable=Path(os.path.abspath(sys.executable))
    launcher=Path(os.path.abspath(declared['launcher'][0]))
    if not launcher.is_file(): raise ValueError('Launcher executable missing')
    packages={}
    for name,module in (('numpy',numpy),('gymnasium',gymnasium)):
        origin=Path(module.__file__).resolve()
        packages[name]=dict(version=module.__version__,file=str(origin),sha256=sha256(origin))
    config=Path(sys.prefix)/'pyvenv.cfg'
    environment=dict(prefix=os.path.abspath(sys.prefix),base_prefix=os.path.abspath(sys.base_prefix),
        exec_prefix=os.path.abspath(sys.exec_prefix),base_exec_prefix=os.path.abspath(sys.base_exec_prefix),
        site_packages=[os.path.abspath(p) for p in site.getsitepackages()],
        user_site=os.path.abspath(site.getusersitepackages()),enable_user_site=site.ENABLE_USER_SITE,
        package_paths={k:os.path.abspath(sysconfig.get_path(k)) for k in ('purelib','platlib')},
        pyvenv_cfg_sha256=sha256(config) if config.is_file() else None)
    return dict(schema='service-v2-runtime-v2',declared=declared,python=sys.version,python_executable=str(executable),
        python_resolved_executable=str(executable.resolve()),python_environment=environment,packages=packages,
        python_sha256=sha256(executable),numpy=numpy.__version__,os=platform.platform(),
        launcher_executable=str(launcher),launcher_resolved_executable=str(launcher.resolve()),launcher_sha256=sha256(launcher),
        environment={k:os.environ.get(k) for k in ('LD_LIBRARY_PATH','PATH','OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONPATH','PYTHONHOME','VIRTUAL_ENV')})


def freeze_phase(phase, seed_plan, design, output, model, reference_model):
    out=reserve(output)  # refusal precedes all reads, never touches existing bytes
    plan=validate_seed_plan(strict_json(seed_plan))
    if phase not in plan['phases']: raise ValueError('Missing phase')
    if sha256(design)!=ADOPTED_SPEC_SHA256: raise ValueError('Adopted spec mismatch')
    if sha256(reference_model)!=REFERENCE_SHA256: raise ValueError('Exact reference model mismatch')
    for name,digest in ACCEPTED_REFERENCE_SOURCES.items():
        if sha256(safe_path(ROOT,name))!=digest: raise ValueError('Accepted legacy reference source changed: '+name)
    from ..prevention.prediction import LearnedPredictor
    old=LearnedPredictor.load(reference_model)
    if old.artifact['comm_range']!=240: raise ValueError('Reference range')
    if (phase in ('pilot','formal')) != (model is not None): raise ValueError('New model required only for pilot/formal')
    # The parent performs the complete historical audit before transfer. Its
    # raw file hashes, source/host/command provenance and bound literal inventory
    # travel intact; source-host absolute paths are provenance, not runtime I/O.
    audit=_validate_audit(plan)
    for scenarios in plan['phases'].values():
        for s in scenarios:
            if s['layout_seed'] in audit['layout_seeds'] or s['radio_seed'] in audit['radio_seeds']: raise ValueError('Historical collision')
    scenarios=plan['phases'][phase]
    old.validate_test_scenarios([(s['layout_seed'],s['radio_seed']) for s in scenarios])
    new=None
    if model is not None:
        from .prediction import ServicePredictor
        new=ServicePredictor.load(model)
        new.validate_scenarios([dict(s,config=asdict(scenario_config(s))) for s in scenarios])
    inherited=None
    if phase=='formal':
        inherited=load_phase(plan['inherits_pilot'])
        if inherited['phase']!='pilot' or inherited['new_model']!=sha256(model): raise ValueError('Formal model must inherit reviewed pilot')
        from .study import validate_collection, read_frames
        from .analysis import episode_endpoints
        from .prediction import ServiceEpisode
        pilot=validate_collection(plan['inherits_pilot'],plan['pilot_run'])
        if not pilot['complete'] or pilot['planned_cells']!=48: raise ValueError('Pilot engineering completeness gate')
        for row in pilot['rows']:
            if row['policy']!='hold': continue
            from ..prevention.config import PreventionConfig
            episode=Path(plan['pilot_run'])/row['cell_id']/f"attempt-{row['attempt']}"/row['episode_path']
            e=ServiceEpisode(row['cell_id'],'pilot','hold',row['regime'],row['context'],row['layout_seed'],row['radio_seed'],
                PreventionConfig(**row['config']),read_frames(episode,PreventionConfig(**row['config'])),{},row['trace_sha256'])
            if not episode_endpoints(e)['armed_by_step20_numerator']: raise ValueError('Pilot HOLD healthy-start gate')
    runtime=runtime_identity(plan['runtime'])
    sources={}
    transforms=plan.get('source_transformations',{})
    names={p.relative_to(ROOT).as_posix() for p in source_paths(ROOT)}
    if set(transforms)-names: raise ValueError('Unknown named source transformation')
    for p in source_paths(ROOT):
        name=p.relative_to(ROOT).as_posix(); raw=p.read_bytes(); transformation=transforms.get(name,'identity')
        if transformation not in ('identity','crlf_to_lf_only'): raise ValueError('Unsupported transformation')
        executed=raw.replace(b'\r\n',b'\n') if transformation=='crlf_to_lf_only' else raw
        target=out/'source'/name; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(executed)
        digest=lambda b:hashlib.sha256(b).hexdigest()
        sources[name]=dict(original_sha256=digest(raw),executed_sha256=digest(executed),
            normalized_sha256=digest(raw.replace(b'\r\n',b'\n')),transformation=transformation)
    if inherited is not None and strict_json(Path(inherited['_path']).parent/'source-manifest.json')!=sources:
        raise ValueError('Formal source differs from reviewed pilot preparation')
    for name,source in [('design.md',design),('reference-model.json',reference_model)]+([('new-model.json',model)] if model is not None else []):
        (out/name).write_bytes(Path(source).read_bytes())
    (out/'seed-plan.json').write_bytes(Path(seed_plan).read_bytes())
    write_json(out/'method.json',asdict(ServiceV2Config()))
    write_json(out/'source-manifest.json',sources)
    write_json(out/'runtime.json',runtime)
    policies=list(POLICIES if phase in ('pilot','formal') else ('hold','geometry'))
    cells=[dict(s,cell_id=s['id']+'__'+p,policy=p,config=asdict(scenario_config(s)),
        output_path=s['id']+'__'+p,attempts=[0,1]) for s in scenarios for p in policies]
    protocol=dict(schema='service-v2-phase-v1',phase=phase,policies=policies,cells=cells,
        expected_episodes=len(cells),spec_sha256=ADOPTED_SPEC_SHA256,reference_model=REFERENCE_SHA256,
        audited_seed_plan_sha256=sha256(seed_plan),
        new_model=sha256(model) if model is not None else None,method_sha256=sha256(out/'method.json'),
        source_root_at_freeze=str(ROOT),package_root=str(out.resolve()),runtime=runtime,
        protocol_path_at_freeze=str((out/'protocol.json').resolve()),
        working_directory=str((out/'source').resolve()),
        inherited_pilot_freeze_sha256=inherited['_freeze_sha256'] if inherited else None,
        retry='one identical-cell technical/infrastructure retry only',service_observation=True,
        run_argv=[runtime['python_executable'],'-m','src.control.service_v2.cli','run-case',
            '--protocol',str((out/'protocol.json').resolve()),'--cell-id','{cell_id}',
            '--attempt','{attempt}','--output','{exclusive_attempt_output}'])
    write_json(out/'protocol.json',protocol)
    files={p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'freeze.json',dict(schema='service-v2-freeze-v1',files=files))
    load_phase(out/'protocol.json')
    return out/'protocol.json'


def load_phase(path):
    path=Path(path).resolve(); root=path.parent
    freeze=strict_json(root/'freeze.json')
    if freeze.get('schema')!='service-v2-freeze-v1': raise ValueError('Freeze schema')
    files=freeze['files']
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual!=set(files)|{'freeze.json'}: raise ValueError('Unlisted/missing frozen file')
    for name,digest in files.items():
        if sha256(safe_path(root,name))!=digest: raise ValueError('Frozen file mismatch: '+name)
    p=strict_json(path)
    if p.get('schema')!='service-v2-phase-v1' or p.get('phase') not in COUNTS: raise ValueError('Phase schema')
    plan=validate_seed_plan(strict_json(root/'seed-plan.json'))
    _validate_audit(plan)
    if sha256(root/'seed-plan.json')!=p['audited_seed_plan_sha256']: raise ValueError('Audited seed plan raw identity mismatch')
    if p['runtime']!=strict_json(root/'runtime.json'): raise ValueError('Runtime manifest disagreement')
    # Old immutable packages remain inspectable/analyzable. Only v2 packages
    # carry sufficient environment evidence to authorize a fresh execution.
    runtime_schema=p['runtime'].get('schema')
    if runtime_schema is not None:
        if runtime_schema!='service-v2-runtime-v2': raise ValueError('Unsupported runtime identity schema')
        invocation=p['runtime'].get('python_executable')
        if (not isinstance(invocation,str) or not (PurePosixPath(invocation).is_absolute() or PureWindowsPath(invocation).is_absolute())
                or not p.get('run_argv') or p['run_argv'][0]!=invocation):
            raise ValueError('Runtime identity/command invocation disagreement')
    policies=list(POLICIES if p['phase'] in ('pilot','formal') else ('hold','geometry'))
    expected=[dict(s,cell_id=s['id']+'__'+role,policy=role,config=asdict(scenario_config(s)),
        output_path=s['id']+'__'+role,attempts=[0,1]) for s in plan['phases'][p['phase']] for role in policies]
    if p['policies']!=policies or p['cells']!=expected or p['expected_episodes']!=len(expected): raise ValueError('Cell inventory mismatch')
    if (p['spec_sha256']!=ADOPTED_SPEC_SHA256 or sha256(root/'design.md')!=ADOPTED_SPEC_SHA256
            or p['reference_model']!=REFERENCE_SHA256 or sha256(root/'reference-model.json')!=REFERENCE_SHA256
            or strict_json(root/'method.json')!=strict_loads(json.dumps(asdict(ServiceV2Config())))
            or p['method_sha256']!=sha256(root/'method.json')): raise ValueError('Method/model/spec mismatch')
    if p['new_model'] is not None:
        if p['phase'] not in ('pilot','formal'): raise ValueError('Collection-only package must have explicit new_model=null')
        from .prediction import ServicePredictor
        if sha256(root/'new-model.json')!=p['new_model']: raise ValueError('Model mismatch')
        ServicePredictor.load(root/'new-model.json').validate_scenarios(p['cells'])
    elif p['phase'] in ('pilot','formal'): raise ValueError('Missing new model')
    p['_path']=str(path)
    p['_freeze_sha256']=sha256(root/'freeze.json')
    return p


def verify_runtime(phase, runtime_root):
    if phase['runtime'].get('schema')!='service-v2-runtime-v2':
        raise ValueError('Runtime identity lacks environment binding; create a new frozen package')
    root=Path(runtime_root).resolve(); package=Path(phase['_path']).parent
    entries=strict_json(package/'source-manifest.json')
    for name,entry in entries.items():
        verify_source_bytes(safe_path(root,name).read_bytes(),entry)
        original=(package/'source'/name).read_bytes()
        verify_source_bytes(original,entry)
    if {p.relative_to(root).as_posix() for p in source_paths(root)}!=set(entries): raise ValueError('Runtime source inventory mismatch')
    if runtime_identity(phase['runtime']['declared'])!=phase['runtime']: raise ValueError('Runtime identity changed')


def publish_status(attempt_root, status):
    allowed=('planned','running','ok','technical_error','interrupted')
    if status.get('status') not in allowed: raise ValueError('Status state')
    if status['status']=='ok' and (status.get('actual_exit')!=0 or status.get('validated') is not True): raise ValueError('ok requires validated process/terminal accounting')
    root=Path(attempt_root); history=root/'status-history'; history.mkdir(parents=True,exist_ok=True)
    revision=len(list(history.glob('*.json')))+1
    value=dict(status,revision=revision,published_at=datetime.now(timezone.utc).isoformat(),
        previous_revision_sha256=sha256(history/f'{revision-1:06d}.json') if revision>1 else None)
    write_json(history/f'{revision:06d}.json',value)
    temporary=root/('status-'+uuid4().hex+'.tmp'); write_json(temporary,value)
    os.replace(temporary,root/'status.json')  # only documented mutable snapshot
