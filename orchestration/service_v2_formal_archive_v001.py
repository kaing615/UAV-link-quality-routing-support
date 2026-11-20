"""Parent-only archive after recorded formal validation and analysis complete."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path('/home/sac/uav-service/2026-11-12-service-prevention-v2')
study=root/'reports/uav_service_prevention_v2_2026-11-12'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
result=read(root/'evidence/formal-execute-recorded-v004/result.json')
validation=read(study/'validation/formal.json')
analysis=read(study/'formal-analysis/analysis.json')
if result.get('direct_exit')!=0 or validation.get('complete') is not True or validation.get('ok_cells')!=384:
    raise ValueError('Recorded complete formal execution/validation required')
if analysis.get('complete') is not True or analysis.get('planned_cells')!=384 or analysis.get('ok_cells')!=384:
    raise ValueError('Complete formal analysis required')
relative=[
 'reports/uav_service_prevention_v2_2026-11-12/runs/formal',
 'reports/uav_service_prevention_v2_2026-11-12/frozen/formal',
 'reports/uav_service_prevention_v2_2026-11-12/formal-analysis',
 'reports/uav_service_prevention_v2_2026-11-12/validation/formal.json',
 'reports/uav_service_prevention_v2_2026-11-12/engineering/task4-parent-adoption-v001.json',
 'reports/uav_service_prevention_v2_2026-11-12/engineering/task4-specification-accepted-v001.json',
 'reports/uav_service_prevention_v2_2026-11-12/engineering/task4-quality-accepted-v001.json',
 'reports/uav_service_prevention_v2_2026-11-12/engineering/formal-prelaunch-accepted-v001.json',
 'reports/uav_service_prevention_v2_2026-11-12/engineering/formal-launch-approval-v001.json',
 'evidence/formal-execute-v004','evidence/formal-execute-recorded-v004',
 'evidence/formal-execute-recorded-v004.launch.json',
 'evidence/formal-execute-recorded-v004.launcher.stdout.log',
 'evidence/formal-execute-recorded-v004.launcher.stderr.log',
 'evidence/queue-formal-v004','evidence/formal-prepare-v004',
 'evidence/formal-prepare-recorded-v004','evidence/queue-dry-run-formal-v004',
 'evidence/dependency-retained-inventory-v001.json','task-reviews','orchestration']
files={}
for name in relative:
    p=root/name
    if not p.exists(): raise FileNotFoundError(str(p))
    for f in ([p] if p.is_file() else sorted(p.rglob('*'))):
        if f.is_file(): files[f.relative_to(root).as_posix()]=dict(sha256=sha(f),bytes=f.stat().st_size)
out=study/'remote-evidence'
inventory=out/'formal-retained-files-v001.json'
with inventory.open('x') as f:
    json.dump(dict(schema='service-v2-retained-files-v1',original_root=str(root),
        relative_root='repository root after extraction',files=files,
        external_dependencies='Retained read-only at original dependency roots; exact identities are in evidence/dependency-retained-inventory-v001.json. Not a self-contained runtime.'),f,indent=2)
archive=out/'formal-complete-v001.tar.gz'
with archive.open('xb') as stream:
    with tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for name in files: tar.add(root/name,arcname=name,recursive=False)
        tar.add(inventory,arcname=inventory.relative_to(root).as_posix(),recursive=False)
print(json.dumps(dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
    inventory_sha256=sha(inventory),files=len(files)),indent=2))
