"""Parent orchestration: actual-host smoke, immutable phase freeze, no simulation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('/home/sac/uav-service/2026-11-12-service-prevention-v2')
source = root / 'source'
study = root / 'reports/uav_service_prevention_v2_2026-11-12'
evidence = root / 'evidence/preflight-v001'
evidence.mkdir(exist_ok=False)
environment = dict(os.environ, LD_LIBRARY_PATH='/home/sac/uav-service/2026-11-11-path-control/runtime/lib',
    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')


def run(name, argv, cwd=source):
    started = time.time()
    with (evidence / f'{name}.stdout.log').open('xb') as stdout:
        with (evidence / f'{name}.stderr.log').open('xb') as stderr:
            child = subprocess.run(argv, cwd=cwd, env=environment, stdout=stdout, stderr=stderr, check=False)
    value = dict(argv=argv, cwd=str(cwd), direct_exit=child.returncode,
        wall_seconds=time.time() - started,
        environment={key:environment[key] for key in ('PATH','LD_LIBRARY_PATH',
            'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    (evidence / f'{name}.json').open('x').write(json.dumps(value, indent=2))
    if child.returncode:
        raise RuntimeError(f'{name} failed, retained exit {child.returncode}')


smoke = '''
import sys, numpy as np
from src.control.service_v2.config import ServiceV2Config
from src.control.service_v2.observation import ServiceHistory
from src.control.service_v2.features import service_features
from src.control.service_v2.planner import encode_displacement, enumerate_plans
from src.control.service_v2.controller import ServiceV2Controller
from src.control.prevention.prediction import LearnedPredictor
from tests.service_v2_support import fixture_config, service_frames
c=fixture_config(); frames=service_frames([60]*10); history=ServiceHistory(c,ServiceV2Config())
states=[history.update(f) for f in frames]
assert next(s.step for s in states if s.ever_armed)==7
features=service_features(frames,c)
assert len(features)==25
np.testing.assert_allclose(encode_displacement(np.array([1.,0.,0.]),2.,8.),[.5,0.,0.])
assessment=enumerate_plans(frames[-1],c,ServiceV2Config(),'service-v2')
assert len(assessment.candidates)<=9
old=LearnedPredictor.load('reports/uav_prevention_pilot_2026-11-11/predictor.json')
assert old.artifact['comm_range']==240
print({'python':sys.version,'features':len(features),'earliest_arm':7,'runtime_smoke':'passed','simulations':0})
'''
try:
    run('runtime-smoke', [sys.executable, '-c', smoke])
    run('binary-linkage', ['/usr/bin/ldd', str(source / 'build/recovery/uav-olsr-dataset')])
    for phase in ('train','calibration','development'):
        run('freeze-'+phase, [sys.executable,'-m','src.control.service_v2.cli','freeze',
            '--phase',phase,'--seed-plan',str(source / f'reports/uav_service_prevention_v2_2026-11-12/seed-plans/{phase}.json'),
            '--design',str(source / 'docs/superpowers/specs/2026-11-12-service-prevention-v2-design.md'),
            '--output',str(study / 'frozen' / phase),
            '--reference-model',str(source / 'reports/uav_prevention_pilot_2026-11-11/predictor.json')])
        run('queue-dry-run-'+phase, [sys.executable,str(root / 'orchestration/service_v2_parent_queue.py'),
            '--protocol',str(study / 'frozen' / phase / 'protocol.json'),
            '--run-root',str(study / 'runs' / phase),
            '--evidence',str(root / 'evidence' / ('queue-dry-run-'+phase)),
            '--lock',str(root / 'global-simulations.lock'), '--dry-run'])
    (evidence / 'completed.json').open('x').write(json.dumps({'status':'passed','simulations':0}))
except BaseException as error:
    (evidence / 'failed.json').open('x').write(json.dumps({'error':f'{type(error).__name__}: {error}'}))
    raise
