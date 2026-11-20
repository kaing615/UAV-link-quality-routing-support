"""Parent-only phase orchestration, using immutable reviewed public CLI calls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('phase', choices=('pilot', 'formal'))
parser.add_argument('operation', choices=('prepare', 'execute'))
args = parser.parse_args()
root = Path('/home/sac/uav-service/2026-11-12-service-prevention-v2')
source = root / 'source-runtime-v002'
study = root / 'reports/uav_service_prevention_v2_2026-11-12'
phase = args.phase
evidence = root / 'evidence' / (phase + '-' + args.operation + '-v001')
evidence.mkdir(exist_ok=False)
environment = dict(os.environ,
    LD_LIBRARY_PATH='/home/sac/uav-service/2026-11-11-path-control/runtime/lib',
    OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
protocol_path = study / 'frozen' / phase / 'protocol.json'


def run(name, argv, cwd=source):
    started = time.time()
    with (evidence / (name + '.stdout.log')).open('xb') as stdout:
        with (evidence / (name + '.stderr.log')).open('xb') as stderr:
            child = subprocess.run(argv, cwd=cwd, env=environment,
                                   stdout=stdout, stderr=stderr, check=False)
    value = dict(argv=argv, cwd=str(cwd), direct_exit=child.returncode,
                 wall_seconds=time.time()-started, finished_unix=time.time(),
                 environment={k:environment.get(k) for k in
                     ('LD_LIBRARY_PATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
    (evidence / (name + '.json')).open('x').write(json.dumps(value, indent=2))
    if child.returncode:
        raise RuntimeError(name + ' failed; evidence retained')


try:
    assert (root / 'evidence/development-runtime-v002/completed.json').is_file()
    if phase == 'formal' and args.operation == 'execute':
        # Parent publishes only after independent engineering review and the
        # hash-only formal prelaunch review; orchestration cannot adopt it.
        gate = json.loads((study / 'engineering/formal-launch-gate.json').read_text())
        assert gate['approved'] is True
    if args.operation == 'prepare':
        run('freeze', [sys.executable,'-m','src.control.service_v2.cli','freeze',
            '--phase',phase,'--seed-plan',str(source / f'reports/uav_service_prevention_v2_2026-11-12/seed-plans/{phase}.json'),
            '--design',str(source/'docs/superpowers/specs/2026-11-12-service-prevention-v2-design.md'),
            '--output',str(protocol_path.parent),'--model',str(study/'model/predictor.json'),
            '--reference-model',str(source/'reports/uav_prevention_pilot_2026-11-11/predictor.json')])
        protocol = json.loads(protocol_path.read_bytes())
        probe = "from pathlib import Path; import sys; from src.control.service_v2.protocol import load_phase,verify_runtime,ROOT; p=load_phase(sys.argv[1]); assert p['run_argv'][0]==sys.executable; assert sys.prefix!=sys.base_prefix; assert Path(sys.executable).is_symlink(); verify_runtime(p,ROOT); print({'invocation':sys.executable,'passed':True})"
        run('actual-frozen-runtime-probe',[protocol['run_argv'][0],'-c',probe,str(protocol_path)],protocol_path.parent/'source')
        run('queue-dry-run',[sys.executable,str(root/'orchestration/service_v2_parent_queue_v002.py'),
            '--protocol',str(protocol_path),'--run-root',str(study/'runs'/phase),
            '--evidence',str(root/'evidence'/('queue-dry-run-'+phase+'-v001')),
            '--lock',str(root/'global-simulations.lock'),'--dry-run'])
    else:
        assert (root/'evidence'/(phase+'-prepare-v001')/'completed.json').is_file()
        run('queue',[sys.executable,str(root/'orchestration/service_v2_parent_queue_v002.py'),
            '--protocol',str(protocol_path),'--run-root',str(study/'runs'/phase),
            '--evidence',str(root/'evidence'/('queue-'+phase+'-v001')),
            '--lock',str(root/'global-simulations.lock')])
        run('validate',[sys.executable,'-m','src.control.service_v2.cli','validate',
            '--protocol',str(protocol_path),'--run-root',str(study/'runs'/phase),
            '--output',str(study/'validation'/(phase+'.json'))])
        run('analyze',[sys.executable,'-m','src.control.service_v2.cli','analyze',
            '--protocol',str(protocol_path),'--run-root',str(study/'runs'/phase),
            '--output',str(study/(phase+'-analysis'))])
    (evidence/'completed.json').open('x').write(json.dumps(dict(phase=phase,operation=args.operation,status='passed')))
except BaseException as error:
    (evidence/'failed.json').open('x').write(json.dumps(dict(error=f'{type(error).__name__}: {error}')))
    raise
