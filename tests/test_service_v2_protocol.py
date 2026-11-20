import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.control.service_v2.protocol import (strict_json, audit_seed_inventory,
    validate_seed_plan, publish_status, verify_source_bytes, freeze_phase, load_phase,
    verify_runtime, write_json, sha256)
from src.control.service_v2 import protocol as protocol_module


def plan(phase='development'):
    count = {'train': 12, 'calibration': 6, 'development': 4}[phase]
    scenarios = []
    for n in (4, 5):
        for i in range(count):
            for context in ('static', 'gradual'):
                scenarios.append(dict(id=f'n{n}-{i}-{context}', regime=f'n{n}', context=context,
                    layout_seed=n*1000+i, radio_seed=n*1000+100+i))
    return dict(schema='service-v2-seed-plan-v1', phases={phase: scenarios},
                runtime=dict(launcher=[sys.executable], ns3_identity='fixture', compiler_identity='fixture'))


def test_strict_json(tmp_path):
    for text in ('{"a":1,"a":2}', '{"a":NaN}', '{'):
        p = tmp_path/'bad.json'; p.write_text(text)
        with pytest.raises(ValueError): strict_json(p)


def test_cardinality_and_collision(tmp_path):
    p = plan(); validate_seed_plan(p)
    p['phases']['development'].pop()
    with pytest.raises(ValueError): validate_seed_plan(p)
    p = plan(); p['phases']['train'] = p['phases']['development']
    with pytest.raises(ValueError): validate_seed_plan(p)
    p = plan(); historical = tmp_path/'history'; historical.mkdir()
    (historical/'failed.json').write_text(json.dumps({'layout_seed':4000, 'radio_seed':999}))
    with pytest.raises(ValueError): audit_seed_inventory(p, historical)


def test_exclusive_publication_preserves_existing(tmp_path):
    target = tmp_path/'frozen'; target.mkdir(); marker=target/'keep'; marker.write_bytes(b'original')
    with pytest.raises(FileExistsError): freeze_phase('train', tmp_path/'missing', tmp_path/'missing', target, None, tmp_path/'missing')
    assert marker.read_bytes() == b'original'


def test_named_line_endings_only():
    import hashlib
    raw=b'a\r\nb\r\n'; transformed=raw.replace(b'\r\n', b'\n')
    sha=lambda b:hashlib.sha256(b).hexdigest()
    entry=dict(original_sha256=sha(raw), executed_sha256=sha(transformed), normalized_sha256=sha(transformed), transformation='crlf_to_lf_only')
    verify_source_bytes(transformed, entry)
    with pytest.raises(ValueError): verify_source_bytes(b'a\nb!\n', entry)
    with pytest.raises(ValueError): verify_source_bytes(transformed, dict(entry,transformation='identity',executed_sha256=sha(raw)))


def test_status_retains_revisions(tmp_path):
    publish_status(tmp_path, dict(status='running', actual_exit=None))
    publish_status(tmp_path, dict(status='technical_error', actual_exit=7))
    assert len(list((tmp_path/'status-history').glob('*.json'))) == 2
    assert strict_json(tmp_path/'status.json')['actual_exit'] == 7
    with pytest.raises(ValueError): publish_status(tmp_path, dict(status='ok',actual_exit=0))


def test_cli_invalid_arguments():
    r=subprocess.run([sys.executable,'-m','src.control.service_v2.cli','run-case','--attempt','2'],capture_output=True)
    assert r.returncode != 0


@pytest.fixture(scope='module')
def frozen(tmp_path_factory):
    root=tmp_path_factory.mktemp('v2-package'); history=root/'history'; history.mkdir()
    candidate=audit_seed_inventory(plan(),history)
    seed=root/'seeds.json'; write_json(seed,candidate)
    protocol=freeze_phase('development',seed,Path('docs/superpowers/specs/2026-11-12-service-prevention-v2-design.md'),
        root/'frozen',None,Path('reports/uav_prevention_pilot_2026-11-11/predictor.json'))
    return protocol


def test_real_freeze_manifest_and_runtime(frozen):
    p=load_phase(frozen)
    assert p['new_model'] is None and len(p['cells'])==32
    assert p['policies']==['hold','geometry']
    verify_runtime(p,Path(p['working_directory']))
    before=(frozen.parent/'source/src/control/service_v2/config.py').read_bytes()
    path=frozen.parent/'source/src/control/service_v2/config.py'
    try:
        path.write_bytes(before+b'!')
        with pytest.raises(ValueError): verify_runtime(p,Path(p['working_directory']))
        with pytest.raises(ValueError): load_phase(frozen)
    finally: path.write_bytes(before)


@pytest.mark.parametrize('mutation', ['missing','duplicate','unknown','phase','model','spec'])
def test_manifest_mutation_rejected(frozen,mutation):
    raw=frozen.read_bytes(); p=json.loads(raw)
    if mutation=='missing': p['cells'].pop()
    elif mutation=='duplicate': p['cells'].append(p['cells'][0])
    elif mutation=='unknown': p['cells'][0]['policy']='invented'
    elif mutation=='phase': p['phase']='formal'
    elif mutation=='model': p['new_model']='0'*64
    else: p['spec_sha256']='0'*64
    try:
        frozen.write_text(json.dumps(p))
        with pytest.raises(ValueError): load_phase(frozen)
    finally: frozen.write_bytes(raw)


def test_cli_exclusive_and_nonzero_propagation(tmp_path,frozen):
    marker=tmp_path/'keep.json'; marker.write_bytes(b'unchanged')
    result=subprocess.run([sys.executable,'-m','src.control.service_v2.cli','validate',
        '--protocol',str(frozen),'--run-root',str(tmp_path),'--output',str(marker)],capture_output=True)
    assert result.returncode==1 and marker.read_bytes()==b'unchanged'
    result=subprocess.run([sys.executable,'-m','src.control.service_v2.cli','run-case',
        '--protocol',str(frozen),'--cell-id','unknown','--attempt','0','--output',str(tmp_path/'new')],capture_output=True)
    assert result.returncode==1 and not (tmp_path/'new').exists()


def test_either_identifier_collision_across_phases_and_regimes():
    from copy import deepcopy
    p=plan(); p['phases']['calibration']=plan('calibration')['phases']['calibration']
    with pytest.raises(ValueError): validate_seed_plan(p)
    for key in ('layout_seed','radio_seed'):
        p=plan(); p['phases']['development'][-1][key]=p['phases']['development'][0][key]
        with pytest.raises(ValueError): validate_seed_plan(p)


def test_pilot_requires_new_model(tmp_path):
    p=plan(); p['phases']={'pilot':[dict(id=f'n{n}-{i}-{context}',regime=f'n{n}',context=context,
        layout_seed=n*1000+i,radio_seed=n*1000+100+i) for n in (4,5) for i,context in enumerate(('static','gradual','gradual'))]}
    history=tmp_path/'h'; history.mkdir(); p=audit_seed_inventory(p,history)
    seeds=tmp_path/'seed.json'; write_json(seeds,p)
    with pytest.raises(ValueError,match='New model'):
        freeze_phase('pilot',seeds,Path('docs/superpowers/specs/2026-11-12-service-prevention-v2-design.md'),
            tmp_path/'frozen',None,Path('reports/uav_prevention_pilot_2026-11-11/predictor.json'))


@pytest.mark.parametrize('historical',[{'seed':4100},{'layout_seeds':[4000]},
    {'radio_seeds':[4100]},{'splits':{'train':[[4000,4100]]}}])
def test_historical_seed_manifest_variants(tmp_path,historical):
    history=tmp_path/'history'; history.mkdir()
    (history/'failed-attempt.json').write_text(json.dumps(historical))
    with pytest.raises(ValueError,match='Historical seed collision'):
        audit_seed_inventory(plan(),history)


def test_detached_audit_is_portable_and_inventory_bound(tmp_path):
    history=tmp_path/'windows-history'; history.mkdir()
    (history/'old.json').write_text('{"layout_seed":777,"radio_seed":888}')
    audited=audit_seed_inventory(plan(),history)
    history.rename(tmp_path/'retained-original-history')
    seeds=tmp_path/'audited.json'; seeds.write_text(' \n'+json.dumps(audited,separators=(',',':'))+'\n\n')
    frozen=freeze_phase('development',seeds,Path('docs/superpowers/specs/2026-11-12-service-prevention-v2-design.md'),
        tmp_path/'frozen',None,Path('reports/uav_prevention_pilot_2026-11-11/predictor.json'))
    assert load_phase(frozen)['expected_episodes']==32
    assert (frozen.parent/'seed-plan.json').read_bytes()==seeds.read_bytes()
    audited['phases']['development'][0]['layout_seed']+=123
    # Change its paired context too, so failure is audit binding rather than
    # an unrelated within-split pairing check.
    audited['phases']['development'][1]['layout_seed']+=123
    altered=tmp_path/'altered.json'; write_json(altered,audited)
    with pytest.raises(ValueError,match='audit'):
        freeze_phase('development',altered,Path('docs/superpowers/specs/2026-11-12-service-prevention-v2-design.md'),
            tmp_path/'refused',None,Path('reports/uav_prevention_pilot_2026-11-11/predictor.json'))


def test_runtime_identity_binds_environment_and_both_executable_paths():
    identity=protocol_module.runtime_identity(plan()['runtime'])
    assert identity['schema']=='service-v2-runtime-v2'
    assert identity['python_executable']==os.path.abspath(sys.executable)
    assert identity['python_resolved_executable']==str(Path(sys.executable).resolve())
    assert identity['python_sha256']==sha256(sys.executable)
    assert identity['python_environment']['prefix']==os.path.abspath(sys.prefix)
    assert identity['python_environment']['site_packages']
    for package in ('numpy','gymnasium'):
        entry=identity['packages'][package]
        assert Path(entry['file']).is_file() and entry['sha256']==sha256(entry['file'])


@pytest.mark.parametrize('invocation',['/research/venv/bin/python',r'C:\Research\venv\Scripts\python.exe'])
def test_freeze_uses_bound_invocation_path(tmp_path,monkeypatch,invocation):
    identity=protocol_module.runtime_identity(plan()['runtime'])
    identity['python_executable']=invocation
    monkeypatch.setattr(protocol_module,'runtime_identity',lambda declared: identity)
    history=tmp_path/'history'; history.mkdir()
    seed=tmp_path/'seed.json'; write_json(seed,audit_seed_inventory(plan(),history))
    frozen=freeze_phase('development',seed,Path('docs/superpowers/specs/2026-11-12-service-prevention-v2-design.md'),
        tmp_path/'frozen',None,Path('reports/uav_prevention_pilot_2026-11-11/predictor.json'))
    p=load_phase(frozen)
    assert p['run_argv'][0]==invocation==p['runtime']['python_executable']
    assert p['runtime']==strict_json(frozen.parent/'runtime.json')


@pytest.mark.parametrize('change',['prefix','site_packages','numpy_origin','gymnasium_origin','legacy_identity'])
def test_verify_runtime_refuses_wrong_environment(frozen,monkeypatch,change):
    import site
    import numpy
    import gymnasium
    p=load_phase(frozen)
    if change=='prefix': monkeypatch.setattr(sys,'prefix',sys.prefix+'-other-env')
    elif change=='site_packages': monkeypatch.setattr(site,'getsitepackages',lambda: ['different-site-packages'])
    elif change=='numpy_origin': monkeypatch.setattr(numpy,'__file__',str(Path(__file__).resolve()))
    elif change=='gymnasium_origin': monkeypatch.setattr(gymnasium,'__file__',str(Path(__file__).resolve()))
    else: p['runtime'].pop('schema',None)
    with pytest.raises(ValueError,match='Runtime identity|runtime identity'):
        verify_runtime(p,Path(p['working_directory']))


@pytest.mark.parametrize('mutation',['legacy','command_mismatch'])
def test_rehashed_runtime_compatibility_boundary(tmp_path,frozen,mutation):
    import shutil
    root=tmp_path/'copied'; shutil.copytree(frozen.parent,root)
    path=root/'protocol.json'; p=strict_json(path)
    if mutation=='legacy':
        for key in ('schema','python_resolved_executable','python_environment','packages','launcher_resolved_executable'):
            p['runtime'].pop(key)
        p['runtime']['python_executable']=str(Path(sys.executable).resolve())
        p['run_argv'][0]=p['runtime']['python_executable']
    else: p['run_argv'][0]='/different/environment/bin/python'
    (root/'runtime.json').write_text(json.dumps(p['runtime']))
    path.write_text(json.dumps(p))
    manifest=strict_json(root/'freeze.json')
    for name in ('runtime.json','protocol.json'): manifest['files'][name]=sha256(root/name)
    (root/'freeze.json').write_text(json.dumps(manifest))
    if mutation=='legacy':
        loaded=load_phase(path)  # inspection/analysis can read retained old evidence
        with pytest.raises(ValueError,match='Runtime identity lacks environment binding'):
            verify_runtime(loaded,Path(loaded['working_directory']))
    else:
        with pytest.raises(ValueError,match='invocation disagreement'): load_phase(path)


@pytest.mark.skipif(os.name=='nt',reason='POSIX symlink virtualenv regression; actual Linux3.12 validation belongs to parent')
def test_posix_symlink_venv_public_freeze_preserves_environment(tmp_path):
    import venv
    environment=tmp_path/'environment'
    venv.EnvBuilder(with_pip=False,system_site_packages=True,symlinks=True).create(environment)
    invocation=environment/'bin/python'
    assert invocation.is_symlink() and invocation.resolve()!=invocation
    history=tmp_path/'history'; history.mkdir()
    write_json(tmp_path/'seed.json',audit_seed_inventory(plan(),history))
    # A real interpreter invocation through the filesystem symlink freezes a
    # package; no simulation, launcher process, pip or dependency install occurs.
    code='''
import json,sys
from pathlib import Path
from src.control.service_v2.protocol import *
root=Path(sys.argv[1])
frozen=freeze_phase('development',root/'seed.json',Path('docs/superpowers/specs/2026-11-12-service-prevention-v2-design.md'),root/'frozen',None,Path('reports/uav_prevention_pilot_2026-11-11/predictor.json'))
p=load_phase(frozen); verify_runtime(p,Path(p['working_directory']))
print(json.dumps(dict(argv=p['run_argv'][0],runtime=p['runtime'])))
'''
    result=subprocess.run([str(invocation),'-c',code,str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout)
    assert value['argv']==str(invocation)==value['runtime']['python_executable']
    assert value['runtime']['python_resolved_executable']==str(invocation.resolve())
    assert value['runtime']['python_environment']['prefix']==str(environment)
