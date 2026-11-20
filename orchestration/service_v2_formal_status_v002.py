import collections
import json
from pathlib import Path
root=Path('/home/sac/uav-service/2026-11-12-service-prevention-v2')
e=root/'evidence'
out={}
q=e/'queue-formal-v004'
if q.exists():
    statuses=sorted(q.glob('status-*.json'))
    if statuses:
        value=json.loads(statuses[-1].read_text())
        cells=list(value['cells'].values())
        out['queue']=dict(revision=value['revision'],time_unix=value['time_unix'],
            states=dict(collections.Counter(c['state'] for c in cells)),
            exits=dict(collections.Counter(str(c.get('direct_wrapper_exit')) for c in cells if c['state']=='wrapper_finished')))
for name in ('formal-execute-recorded-v004/result.json','formal-execute-v004/completed.json',
             'formal-execute-v004/failed.json','formal-execute-v004/queue.json',
             'formal-execute-v004/validate.json','formal-execute-v004/analyze.json'):
    p=e/name
    if p.exists(): out[name]=json.loads(p.read_text())
print(json.dumps(out,indent=2))
