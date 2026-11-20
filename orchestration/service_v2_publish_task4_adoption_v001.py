"""Parent publication of reviews actually accepted in this session."""
import importlib.util
import json
from pathlib import Path
import sys

root = Path('/home/sac/uav-service/2026-11-12-service-prevention-v2')
source = root / 'source-runtime-v003'
study = root / 'reports/uav_service_prevention_v2_2026-11-12'
sys.path.insert(0, str(source))
spec = importlib.util.spec_from_file_location('parent_phase', root/'orchestration/service_v2_parent_phase_v004.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
strict_json, sha256, load_phase, verify_runtime, write_json = parent.api()

def ref(path):
    return dict(path=str(path.resolve(strict=True)), sha256=sha256(path))

protocol = study/'frozen/pilot-integration-v003/protocol.json'
pilot = parent.identities(load_phase(protocol))
development = study/'engineering/development-diagnostic-v001/manifest.json'
engineering = study/'engineering/pilot-engineering-v003/manifest.json'
bindings = dict(pilot=pilot, development_manifest_sha256=sha256(development),
                engineering_manifest_sha256=sha256(engineering))
for kind in ('specification', 'quality'):
    report = root/'task-reviews'/f'task-4-{kind}-review.md'
    # Human-readable reports are supplied and accepted by parent before invocation.
    record = dict(schema='service-v2-independent-review-acceptance-v1', scope='task4',
        review_kind=kind, accepted=True, critical=0, important=0, bindings=bindings, report=ref(report))
    write_json(study/'engineering'/f'task4-{kind}-accepted-v001.json', record)

references = dict(pilot_protocol=ref(protocol), development_manifest=ref(development),
    engineering_manifest=ref(engineering),
    train_validation=ref(study/'validation/train-runtime-v002.json'),
    calibration_validation=ref(study/'validation/calibration-runtime-v002.json'),
    development_validation=ref(study/'validation/development-runtime-v002.json'),
    pilot_validation=ref(study/'validation/pilot-integration-v003.json'),
    specification_review=ref(study/'engineering/task4-specification-accepted-v001.json'),
    quality_review=ref(study/'engineering/task4-quality-accepted-v001.json'),
    formal_seed_plan=ref(source/'reports/uav_service_prevention_v2_2026-11-12/seed-plans/formal.json'))
adoption = dict(schema='service-v2-task4-parent-adoption-v1', adopted=True, adopted_by='parent',
                identities=pilot, artifact_bindings=bindings, references=references)
path = study/'engineering/task4-parent-adoption-v001.json'
write_json(path, adoption)
verified = parent.verify_adoption(path, study, source)
print(json.dumps(verified, indent=2))
