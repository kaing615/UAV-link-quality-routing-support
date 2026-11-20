"""Publish parent records after the parent has read accepted review reports.

This is record authoring only. It runs neither research jobs nor model fitting.
The existing reviewed verifier checks each proposed record before publication.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

parser=argparse.ArgumentParser()
parser.add_argument('operation',choices=('adoption','formal-approval'))
parser.add_argument('--spec-review',type=Path)
parser.add_argument('--quality-review',type=Path)
parser.add_argument('--prelaunch-review',type=Path)
parser.add_argument('--parent-confirmed-reviews-accepted',action='store_true',required=True)
args=parser.parse_args()
root=Path('/home/sac/uav-service/2026-11-12-service-prevention-v2')
source=root/'source-runtime-v003'
study=root/'reports/uav_service_prevention_v2_2026-11-12'
sys.path.insert(0,str(source))
spec=importlib.util.spec_from_file_location('phase_gate',root/'orchestration/service_v2_parent_phase_v004.py')
gate=importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
strict_json,sha256,load_phase,verify_runtime,write_json=gate.api()
evidence=root/'evidence'/('parent-'+args.operation+'-publication-v001')
evidence.mkdir(exist_ok=False)


def ref(path):
    path=Path(path).resolve(strict=True)
    return dict(path=str(path),sha256=sha256(path))


def acceptance(kind,scope,bindings,report,destination):
    if report is None:
        raise ValueError('An actual accepted review report is required')
    value=dict(schema='service-v2-independent-review-acceptance-v1',scope=scope,review_kind=kind,
        accepted=True,critical=0,important=0,bindings=bindings,report=ref(report))
    write_json(destination,value)
    return ref(destination)


if args.operation=='adoption':
    pilot_protocol=study/'frozen/pilot-integration-v003/protocol.json'
    phase=load_phase(pilot_protocol)
    dev_manifest=study/'engineering/development-diagnostic-v001/manifest.json'
    eng_manifest=study/'engineering/pilot-engineering-v002/manifest.json'
    bindings=dict(pilot=gate.identities(phase),development_manifest_sha256=sha256(dev_manifest),
                  engineering_manifest_sha256=sha256(eng_manifest))
    refs=dict(pilot_protocol=ref(pilot_protocol),development_manifest=ref(dev_manifest),
        engineering_manifest=ref(eng_manifest),
        train_validation=ref(study/'validation/train-runtime-v002.json'),
        calibration_validation=ref(study/'validation/calibration-runtime-v002.json'),
        development_validation=ref(study/'validation/development-runtime-v002.json'),
        pilot_validation=ref(study/'validation/pilot-integration-v003.json'),
        specification_review=acceptance('specification','task4',bindings,args.spec_review,
            study/'engineering/task4-specification-acceptance-v001.json'),
        quality_review=acceptance('quality','task4',bindings,args.quality_review,
            study/'engineering/task4-quality-acceptance-v001.json'),
        formal_seed_plan=ref(source/'reports/uav_service_prevention_v2_2026-11-12/seed-plans/formal.json'))
    value=dict(schema='service-v2-task4-parent-adoption-v1',adopted=True,adopted_by='parent',
               identities=gate.identities(phase),artifact_bindings=bindings,references=refs)
    proposal=evidence/'adoption-proposal.json'
    write_json(proposal,value)
    verified=gate.verify_adoption(proposal,study,source)
    destination=study/'engineering/task4-parent-adoption-v001.json'
    destination.open('xb').write(proposal.read_bytes())
    write_json(evidence/'published.json',dict(publication=ref(destination),verified_proposal=verified))
else:
    adoption_path=study/'engineering/task4-parent-adoption-v001.json'
    adoption=gate.verify_adoption(adoption_path,study,source)
    phase=load_phase(study/'frozen/formal/protocol.json')
    identities=gate.verify_formal_inheritance(phase,adoption)
    bindings=dict(formal=identities,adoption_sha256=sha256(adoption_path))
    review=acceptance('prelaunch','formal-prelaunch',bindings,args.prelaunch_review,
        study/'engineering/formal-prelaunch-acceptance-v001.json')
    value=dict(schema='service-v2-formal-launch-approval-v1',approved=True,approved_by='parent',
        adoption_sha256=sha256(adoption_path),identities=identities,prelaunch_review=review)
    proposal=evidence/'approval-proposal.json'
    write_json(proposal,value)
    verified=gate.verify_launch(proposal,phase,adoption)
    destination=study/'engineering/formal-launch-approval-v001.json'
    destination.open('xb').write(proposal.read_bytes())
    write_json(evidence/'published.json',dict(publication=ref(destination),verified_proposal=verified))
print({'operation':args.operation,'published':str(destination),'sha256':sha256(destination)})
