import json
from pathlib import Path
import time
root=Path('/home/sac/uav-service/2026-11-12-service-prevention-v2')
e=root/'evidence'
out={}
q=e/'queue-formal-v004'
if q.exists():
    statuses=sorted(q.glob('status-*.json'))
    if statuses:
        value=json.loads(statuses[-1].read_text())
        out['queue_latest']=value
for name in ('formal-execute-recorded-v004/result.json','formal-execute-v004/completed.json',
             'formal-execute-v004/failed.json','formal-execute-v004/verified-formal-launch.json'):
    p=e/name
    if p.exists(): out[name]=json.loads(p.read_text())
print(json.dumps(out,indent=2))
