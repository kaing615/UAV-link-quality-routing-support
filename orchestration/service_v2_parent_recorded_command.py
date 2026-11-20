"""Parent orchestration: retain a single child invocation and direct exit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('--evidence',type=Path,required=True)
parser.add_argument('--cwd',type=Path,required=True)
parser.add_argument('argv',nargs=argparse.REMAINDER)
args=parser.parse_args()
command=args.argv[1:] if args.argv[:1]==['--'] else args.argv
if not command:
    raise ValueError('Explicit command required')
args.evidence.mkdir(parents=True,exist_ok=False)
environment=dict(os.environ,LD_LIBRARY_PATH='/home/sac/uav-service/2026-11-11-path-control/runtime/lib',
    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
started=time.time()
with (args.evidence/'stdout.log').open('xb') as stdout:
    with (args.evidence/'stderr.log').open('xb') as stderr:
        child=subprocess.run(command,cwd=args.cwd,env=environment,stdin=subprocess.DEVNULL,
                             stdout=stdout,stderr=stderr,check=False)
value=dict(argv=command,cwd=str(args.cwd),direct_exit=child.returncode,
    wall_seconds=time.time()-started,finished_unix=time.time(),
    recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    environment={k:environment[k] for k in
        ('LD_LIBRARY_PATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
with (args.evidence/'result.json').open('x') as stream:
    json.dump(value,stream,indent=2)
raise SystemExit(child.returncode)
