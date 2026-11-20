"""Parent fixed development stages; stops on any process/validation/model failure."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('/home/sac/uav-service/2026-11-12-service-prevention-v2')
source = root / 'source'
study = root / 'reports/uav_service_prevention_v2_2026-11-12'
evidence = root / 'evidence/development-v001'
evidence.mkdir(exist_ok=False)
environment = dict(os.environ, LD_LIBRARY_PATH='/home/sac/uav-service/2026-11-11-path-control/runtime/lib',
    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
assert (root / 'evidence/preflight-v001/completed.json').is_file()


def run(name, argv):
    started = time.time()
    with (evidence / f'{name}.stdout.log').open('xb') as stdout:
        with (evidence / f'{name}.stderr.log').open('xb') as stderr:
            child = subprocess.run(argv, cwd=source, env=environment, stdout=stdout, stderr=stderr, check=False)
    value = dict(argv=argv, cwd=str(source), direct_exit=child.returncode,
        wall_seconds=time.time() - started, finished_unix=time.time())
    (evidence / f'{name}.json').open('x').write(json.dumps(value, indent=2))
    if child.returncode:
        raise RuntimeError(f'{name} failed, retained exit {child.returncode}')


try:
    for phase in ('train','calibration','development'):
        run('queue-'+phase, [sys.executable,str(root / 'orchestration/service_v2_parent_queue.py'),
            '--protocol',str(study / 'frozen' / phase / 'protocol.json'),
            '--run-root',str(study / 'runs' / phase),
            '--evidence',str(root / 'evidence' / ('queue-'+phase)),
            '--lock',str(root / 'global-simulations.lock')])
        run('validate-'+phase, [sys.executable,'-m','src.control.service_v2.cli','validate',
            '--protocol',str(study / 'frozen' / phase / 'protocol.json'),
            '--run-root',str(study / 'runs' / phase),
            '--output',str(study / 'validation' / (phase+'.json'))])
    run('fit-model', [sys.executable,'-m','src.control.service_v2.cli','fit',
        '--train-run',str(study/'runs/train'),'--calibration-run',str(study/'runs/calibration'),
        '--output',str(study/'model/predictor.json')])
    (evidence / 'completed.json').open('x').write(json.dumps({'status':'passed','planned_episodes':176}))
except BaseException as error:
    (evidence / 'failed.json').open('x').write(json.dumps({'error':f'{type(error).__name__}: {error}'}))
    raise
