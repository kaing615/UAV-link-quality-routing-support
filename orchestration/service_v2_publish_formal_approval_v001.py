"""Parent publication after accepting an actual independent prelaunch review."""
import json
from pathlib import Path
import sys
root=Path('/home/sac/uav-service/2026-11-12-service-prevention-v2')
sys.path[:0]=[str(root/'orchestration'),str(root/'source-runtime-v003')]
import service_v2_parent_phase_v004 as parent
strict_json,sha256,load_phase,verify_runtime,write_json=parent.api()
study=root/'reports/uav_service_prevention_v2_2026-11-12'
def ref(path): return dict(path=str(path.resolve(strict=True)),sha256=sha256(path))
adoption=parent.verify_adoption(study/'engineering/task4-parent-adoption-v001.json',study,root/'source-runtime-v003')
phase=load_phase(study/'frozen/formal/protocol.json')
current=parent.verify_formal_inheritance(phase,adoption)
review=dict(schema='service-v2-independent-review-acceptance-v1',scope='formal-prelaunch',
    review_kind='prelaunch',accepted=True,critical=0,important=0,
    bindings=dict(formal=current,adoption_sha256=adoption['adoption_sha256']),
    report=ref(root/'task-reviews/task-5-prelaunch-review.md'))
review_path=study/'engineering/formal-prelaunch-accepted-v001.json'
write_json(review_path,review)
approval=dict(schema='service-v2-formal-launch-approval-v1',approved=True,approved_by='parent',
    adoption_sha256=adoption['adoption_sha256'],identities=current,prelaunch_review=ref(review_path))
approval_path=study/'engineering/formal-launch-approval-v001.json'
write_json(approval_path,approval)
print(json.dumps(parent.verify_launch(approval_path,phase,adoption),indent=2))
