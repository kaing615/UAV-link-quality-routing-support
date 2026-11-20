"""Aggregate Tasks1–3 against earliest task before-images, excluding Git dirt.

Inputs are explicit accepted task inventories/patches. Files added by a task
have absent original content, even if a later fix has an intermediate backup.
Do not compare working tree to Git HEAD: that would mix prior unrelated work.
"""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[2]
SPEC=ROOT/'.superpowers/sdd/2026-11-11-routing-service'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def sections(path):
    """Get initial file status from the first hunk, not later fix backups."""
    result={};current=None
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('+++ '):
            current=line[4:].removeprefix('b/')
        elif line.startswith('@@ ') and current is not None and current not in result:
            result[current]=line.startswith('@@ -0,0 ')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args(argv)
    if args.output.exists():raise ValueError('Review package output must be fresh')
    origins={}
    inputs=['task-1-diff-final.patch','task-2-diff.patch','task-3-code-diff.patch']
    for task,filename in enumerate(inputs,start=1):
        for name,added in sections(SPEC/filename).items():
            if name in origins:continue
            candidates=([SPEC/'before'/name,SPEC/'before'/Path(name).name] if task==1 else
                        [SPEC/f'task-{task}-before'/name])
            before=next((p for p in candidates if p.is_file()),None)
            # Task1's historical diff generator missed flattened build-script
            # backups and represented them as additions. Actual retained bytes
            # take precedence over that derived patch's empty-before marker.
            if before is None and added:
                origins[name]=dict(kind='added',before=None,origin=filename)
                continue
            if before is None:raise ValueError('Earliest before-image missing: '+name)
            origins[name]=dict(kind='modified',before=before.relative_to(ROOT).as_posix(),origin=filename)
    additions=['tests/test_bridge_stdout_shutdown.py','scripts/analysis/summarize_path_service_pilot_engineering.py',
               'scripts/analysis/build_routing_service_review_package.py',
               'tests/test_path_service_protocol_provenance.py',
               'scripts/analysis/record_path_service_command.py',
               'scripts/analysis/summarize_path_service_results.py',
               'scripts/analysis/write_path_service_final_report.py',
               'scripts/analysis/export_path_service_submission_pdf.py',
               '.superpowers/sdd/2026-11-11-routing-service/submission-docfix1-report.md',
               '.superpowers/sdd/2026-11-11-routing-service/task-3-report.md',
               'reports/uav_path_service_study_2026-11-11/BAO_CAO_KET_QUA.md',
               'reports/uav_path_service_study_2026-11-11/PHU_LUC_KET_QUA.md',
               'reports/uav_path_service_study_2026-11-11/FINAL_WRITING_AUDIT.md',
               'reports/uav_path_service_study_2026-11-11/FINAL_ANALYSIS_PROVENANCE.json',
               'reports/uav_path_service_study_2026-11-11/stage2progress.md',
               'reports/uav_path_service_study_2026-11-11/stage2-v2-preparation-v1/analysis-plan.json',
               'reports/uav_path_service_study_2026-11-11/stage2-v2-preparation-v1/BAO_CAO_DU_THAO_V2.md',
               'reports/uav_path_service_study_2026-11-11/stage2-v2-preparation-v1/STATUS.md']
    for name in additions:
        if name not in origins:origins[name]=dict(kind='added',before=None,origin='runtime-fix1/analysisfix1/stage2/final new file')
    # A runtime fix must already be represented by the earliest Task3 image.
    for name in sections(SPEC/'task-3-runtime-fix1-diff.patch'):
        if name not in origins:raise ValueError('Unaccounted runtime fix file: '+name)
    args.output.mkdir(parents=True,exist_ok=False)
    patch=[];inventory=[]
    for name,origin in sorted(origins.items()):
        after=ROOT/name
        if not after.is_file():raise ValueError('Current scoped file missing: '+name)
        before=ROOT/origin['before'] if origin['before'] else None
        old=before.read_text(encoding='utf-8').splitlines(keepends=True) if before else []
        new=after.read_text(encoding='utf-8').splitlines(keepends=True)
        patch.extend(difflib.unified_diff(old,new,fromfile='a/'+name if before else '/dev/null',tofile='b/'+name,n=5))
        target=args.output/'current'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(after.read_bytes())
        if before:
            target=args.output/'before'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(before.read_bytes())
        inventory.append(dict(path=name,**origin,before_sha256=digest(before) if before else None,current_sha256=digest(after),
                              changed=old!=new,current_bytes=after.stat().st_size))
    (args.output/'task-1-3-current-before.patch').write_text(''.join(patch),encoding='utf-8')
    (args.output/'inventory.json').write_text(json.dumps(dict(schema='routing-service-aggregate-review-v1',files=inventory,
        inputs={n:digest(SPEC/n) for n in inputs},scope='Task1–3 final source/reports including runtimefix1, analysisfix1 and formal-v2 offline analyses; no raw frame matrices or unrelated pre-existing dirty work',
        excluded='Git HEAD diff, Dockerfile, prior report deletions, tests/test_stage6_runner.py, prior recovery/prevention baseline implementation outside task-scoped edits'),indent=2),encoding='utf-8')
    task3_origins={}
    for name,added in sections(SPEC/'task-3-code-diff.patch').items():
        before=SPEC/'task-3-before'/name
        if before.is_file():
            task3_origins[name]=dict(kind='modified',before=before.relative_to(ROOT).as_posix(),origin='task-3-code-diff.patch')
        elif added:
            task3_origins[name]=dict(kind='added',before=None,origin='task-3-code-diff.patch')
        else:
            raise ValueError('Task3 before-image missing: '+name)
    for name in additions:
        if name not in task3_origins:task3_origins[name]=dict(kind='added',before=None,origin='Task3 later fix/offline/report addition')
    task3_patch=[];task3_inventory=[]
    for name,origin in sorted(task3_origins.items()):
        after=ROOT/name;before=ROOT/origin['before'] if origin['before'] else None
        old=before.read_text(encoding='utf-8').splitlines(keepends=True) if before else []
        new=after.read_text(encoding='utf-8').splitlines(keepends=True)
        task3_patch.extend(difflib.unified_diff(old,new,fromfile='a/'+name if before else '/dev/null',tofile='b/'+name,n=5))
        if before:
            target=args.output/'task-3-before'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(before.read_bytes())
        task3_inventory.append(dict(path=name,**origin,before_sha256=digest(before) if before else None,
                                   current_sha256=digest(after),changed=old!=new,current_bytes=after.stat().st_size))
    (args.output/'task-3-current-before.patch').write_text(''.join(task3_patch),encoding='utf-8')
    (args.output/'task-3-inventory.json').write_text(json.dumps(dict(schema='routing-service-task3-final-review-v1',files=task3_inventory,
        input_sha256=digest(SPEC/'task-3-code-diff.patch'),scope='Task3 initial before-images through runtimefix1/analysisfix1/formal-v2 reporting; excludes Task1–2-only edits'),indent=2),encoding='utf-8')
    references=['task-1-brief.md','task-1-report.md','task-1-review.md','task-2-brief.md','task-2-report.md','task-2-review.md',
        'task-2-fix1-review.md','task-2-fix2-review.md','task-3-brief.md','task-3-design-proposal.md','task-3-design-accepted.md',
        'task-3-code-report.md','task-3-preflight-review.md','task-3-model-contract.md','task-3-runtime-fix1-report.md',
        'task-3-runtime-fix1-review.md','task-3-model-baseline-check.json','task-3-seed-preregistration-audit.json',
        'task-3-analysis-fix1-report.md','task-3-analysis-fix1-review.md','task-3-analysis-fix1-inventory.json',
        'task-3-report.md','task-3-remote-upload-v2.json','task-3-remote-setup.json',
        'task-3-final-review.md','final-whole-review.md','submission-docfix1-report.md',
        'submission-docfix1-generated-task3-report.md','submission-docfix1-final-diff.patch',
        'submission-docfix1-final-inventory.json']
    evidence={}
    for name in references:
        path=SPEC/name;evidence[path.relative_to(ROOT).as_posix()]=digest(path)
        target=args.output/'context'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    for path in (SPEC/'task-3-frozen').glob('*.json'):
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
        target=args.output/'context/task-3-frozen'/path.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    for directory in ('task-3-runtime-fix1-evidence','task-3-analysis-fix1-evidence','task-3-engineering',
                      'submission-docfix1-evidence'):
        for path in (SPEC/directory).rglob('*'):
            if not path.is_file():continue
            evidence[path.relative_to(ROOT).as_posix()]=digest(path)
            target=args.output/'context'/directory/path.relative_to(SPEC/directory);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    study=ROOT/'reports/uav_path_service_study_2026-11-11'
    for path in [study/'PILOT_V2_ENGINEERING_GATE.json',study/'stage2-pilot-v2-engineering-v1/pilot-engineering.json',
                 *sorted((study/'stage2-pilot-v2-command-v1').glob('*')),
                 *sorted((study/'stage2-formal-failure-branch-check-v1').glob('*'))]:
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
        target=args.output/'context/stage2-v2'/path.relative_to(study);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    final_paths=[study/'FORMAL_V2_ENGINEERING_GATE.json',study/'formal-v2-archives.json',
                 study/'FINAL_ANALYSIS_PROVENANCE.json',study/'BAO_CAO_KET_QUA_NOP_DOCFIX1_V1.pdf']
    for directory in ('formal-v2-evaluation-v1','formal-v2-coincidence-v1','formal-v2-summary-v1',
                      'formal-v2-evaluation-command-v1','formal-v2-coincidence-command-v1',
                      'formal-v2-summary-command-v1','formal-v2-report-command-v1','formal-v2-raw/execution',
                      'submission-docfix1-v1'):
        final_paths.extend(p for p in sorted((study/directory).rglob('*')) if p.is_file())
    for name in ('provenance.json','protocol.json','freeze.json','method.json','status.json'):
        path=study/'formal-v2-raw/formal-v2'/name
        if path.is_file():final_paths.append(path)
    for path in final_paths:
        evidence[path.relative_to(ROOT).as_posix()]=digest(path)
        target=args.output/'context/final-v2'/path.relative_to(study);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    raw=study/'formal-v2-raw/formal-v2'
    (args.output/'raw-evidence-pointers.json').write_text(json.dumps(dict(
        root=raw.relative_to(ROOT).as_posix(),manifest_sha256=digest(raw/'manifest.json'),
        provenance_sha256=digest(raw/'provenance.json'),archive_index='reports/uav_path_service_study_2026-11-11/formal-v2-archives.json',
        engineering_gate='reports/uav_path_service_study_2026-11-11/FORMAL_V2_ENGINEERING_GATE.json',
        retained_failed_attempt='reports/uav_path_service_study_2026-11-11/formal-v1-failed-attempt',
        note='Raw per-frame matrices and transport archives remain at original locations; not embedded. Formal-v1 outcomes excluded.'),indent=2),encoding='utf-8')
    (args.output/'evidence-index.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    (args.output/'README.md').write_text('''# Task1–3 current/before review package

This is the documentation-corrected submission snapshot. Formal-v2 collection/offline analysis and independent final Task3/aggregate Tasks1–3 reviews are complete with nonblocking Minors. The Part1 documentation follow-up clarifies the legacy scheduled-offer PDR denominator and historical fixture provenance, preserves numerical outputs, and adds a fresh14-page submission PDF. Dense formatting remains a separate maintainability item.

For Task3 alone read task-3-current-before.patch/task-3-inventory.json against task-3-before/. For all Tasks1–3 read task-1-3-current-before.patch/inventory.json against before/. Both use current/. Earliest initial-task patch establishes additions; later fix backups never replace an earlier absent baseline. Task1 flattened build-script before-images are explicitly mapped. REPORT.md uses task-2-before; new MECHANISM.md is an addition. Aggregate shared simulator/build scripts use Task1 originals; Task3-only comparison uses its own initial images. Runtimefix1, analysisfix1 tests and all final offline/report scripts are included.

Context includes accepted specs, earlier reviews, runtimefix1/analysisfix1 evidence, complete formal-v2 gate, raw metadata/execution provenance, captured offline commands, CSV tables and PNG/PDF figures. Raw63 diagnostic cases, pilot-v2, formal-v1 failed attempt and full formal-v2 frame matrices remain separately available through original paths/provenance and raw-evidence-pointers.json. Formal-v2 has720/720ok and12240verified artifact hashes; direct background job exit code was not captured. Offline endpoint/coincidence/summary commands returned0. Included independent final review records approve the bounded negative/null simulation study; no efficacy superiority is claimed.

Resolved documentation Minors: historical PDR is FlowMonitor receives/scheduled offers ceil(duration/interval), primary is application sink receives/observed application offers, and source-IP Tx is an accounting check. The historical route-present exact-zero fixture is n2/.55/r256/seed71001; n8/.55/r256/seed71001 with.004267 is a separate degradation case; the route0/PDR0 test is illustrative. Fixture test AST is unchanged except docstrings. Dense formatting remains maintainability only. Old-planner coincidence omission was resolved offline with same-frame/tolerance/denominator definitions. Nominal T=d/V is not a realized arrival guarantee near remaining-budget scaling. Centralized synchronous-state/control-cost/timing limits remain explicit. Full/no-deadline actions/frames exact80/80 and negative/null results remain. Formal-v1 is retained technical history, scientific corruption not established.

Unrelated prior Git dirt is excluded by explicit task inventories and before-image selection. No commits or identity changes.
''',encoding='utf-8')
    entries={p.relative_to(args.output).as_posix():digest(p) for p in sorted(args.output.rglob('*')) if p.is_file()}
    (args.output/'package-hashes.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    archive=args.output/'review.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for p in sorted(args.output.rglob('*')):
            if p.is_file() and p!=archive:tar.add(p,arcname=p.relative_to(args.output).as_posix(),recursive=False)
    (args.output/'archive.sha256').write_text(digest(archive)+'  review.tar.gz\n',encoding='ascii')
    print(json.dumps(dict(files=len(inventory),patch_lines=len(patch),archive=str(archive.resolve()),sha256=digest(archive))))

if __name__=='__main__':main()
