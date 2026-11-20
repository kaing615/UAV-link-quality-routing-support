"""Launch only the parent recorder; scientific gates remain in phase helper."""
import json
from pathlib import Path
import subprocess

root = Path('/home/sac/uav-service/2026-11-12-service-prevention-v2')
python = '/home/sac/uav-service/2026-11-11-path-control/.venv/bin/python'
destination = root/'evidence/formal-execute-recorded-v004'
argv = [python, str(root/'orchestration/service_v2_parent_recorded_command.py'),
        '--evidence', str(destination), '--cwd', str(root/'source-runtime-v003'), '--',
        python, str(root/'orchestration/service_v2_parent_phase_v004.py'), 'formal', 'execute']
if destination.exists():
    raise FileExistsError(str(destination))
with Path(str(destination)+'.launcher.stdout.log').open('xb') as stdout:
    with Path(str(destination)+'.launcher.stderr.log').open('xb') as stderr:
        child = subprocess.Popen(argv, cwd=root/'source-runtime-v003', stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, start_new_session=True)
value = dict(pid=child.pid, argv=argv)
with Path(str(destination)+'.launch.json').open('x') as stream:
    json.dump(value, stream, indent=2)
print(json.dumps(value))
