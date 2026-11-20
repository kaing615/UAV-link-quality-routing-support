"""Read-only Task 4 audits. No fitting, simulation, tuning, or formal reads.

Run from the frozen Linux runtime with --source-root explicitly. All output
roots must be absent; failed partial output remains for inspection. Public v2
APIs own scientific definitions. This helper only checks their actual evidence.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)


def read_json(path):
    from src.control.service_v2.protocol import strict_json
    return strict_json(path)


def jsonl(path):
    from src.control.service_v2.protocol import strict_loads
    return [strict_loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def quantiles(values):
    import numpy as np
    if not values:
        return dict(count=0, median=None, p95=None, max=None)
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError('Invalid timing sample')
    return dict(count=len(values), median=float(np.median(array)),
                p95=float(np.quantile(array, .95)), max=float(np.max(array)))


def manifest(root, schema):
    new_json(root / 'manifest.json', dict(schema=schema, files={
        p.relative_to(root).as_posix(): digest(p)
        for p in sorted(root.rglob('*')) if p.is_file()}))


def phase_inputs(study, phase, alias, source_root):
    from src.control.service_v2.protocol import load_phase, verify_runtime
    from src.control.service_v2.study import validate_collection
    protocol = study / 'frozen' / alias / 'protocol.json'
    run_root = study / 'runs' / alias
    frozen = load_phase(protocol)
    if frozen['phase'] != phase:
        raise ValueError('Phase/alias mismatch')
    verify_runtime(frozen, source_root)
    validation = validate_collection(protocol, run_root)
    expected = dict(train=96, calibration=48, development=32, pilot=48)[phase]
    if validation['planned_cells'] != expected:
        raise ValueError('Wrong bounded phase inventory')
    return frozen, validation, run_root


def bound_inputs(frozen, validation, run_root):
    package = Path(frozen['_path']).parent
    return dict(protocol_path=frozen['_path'], protocol_sha256=digest(frozen['_path']),
        freeze_sha256=frozen['_freeze_sha256'], source_manifest_sha256=digest(package/'source-manifest.json'),
        method_sha256=frozen['method_sha256'], spec_sha256=frozen['spec_sha256'],
        runtime=frozen['runtime'], reference_model_sha256=frozen['reference_model'],
        new_model_sha256=frozen['new_model'], collection_sha256=digest(run_root/'collection.json'),
        run_root=str(run_root), planned_cells=validation['planned_cells'], ok_cells=validation['ok_cells'],
        attempts=[dict(cell_id=row['cell_id'], attempt=a['attempt'], status=a['status'],
            actual_exit=a['actual_exit'], error=a.get('error'),
            result_sha256=digest(run_root/row['cell_id']/f"attempt-{a['attempt']}"/'result.json'),
            artifacts_sha256=digest(run_root/row['cell_id']/f"attempt-{a['attempt']}"/'artifacts.json'),
            trace_sha256=a.get('trace_sha256'))
            for row in validation['rows'] for a in row['attempt_records']])


def model_audit(study, source_root, aliases):
    """Validate artifact and recompute provenance/count/normalizer evidence only."""
    import numpy as np
    from src.control.service_v2.prediction import ServicePredictor, build_service_samples
    from src.control.service_v2.study import load_service_episodes
    predictor = ServicePredictor.load(study/'model/predictor.json')
    artifact = predictor.artifact
    result = dict(strict_loader_passed=True, model_path=str(study/'model/predictor.json'),
        model_sha256=digest(study/'model/predictor.json'), phases={}, horizons={},
        calibration_diagnostics=artifact['calibration_diagnostics'],
        interpretation='Calibration HOLD fitted once; calibration geometry diagnostic only. No refit in this helper.')
    episodes = {}
    for name in ('train', 'calibration'):
        phase, validation, root = phase_inputs(study, name, aliases[name], source_root)
        result['phases'][name] = bound_inputs(phase, validation, root)
        episodes[name] = load_service_episodes(root, name)
        metadata = [dict(identity=e.identity, phase=e.phase, policy=e.policy, regime=e.regime,
            context=e.context, layout_seed=e.layout_seed, radio_seed=e.radio_seed,
            config=asdict(e.config), source_hashes=e.source_hashes, trace_sha256=e.trace_sha256)
            for e in episodes[name]]
        if artifact['splits'][name] != metadata:
            raise ValueError('Fitted model raw episode/source provenance mismatch: '+name)
    for horizon in (10, 20):
        model = artifact['models'][str(horizon)]
        rows = build_service_samples(episodes['train'], horizon)
        cal = build_service_samples([e for e in episodes['calibration'] if e.policy=='hold'], horizon)
        if rows.counts != model['train_counts'] or cal.counts != model['calibration_counts']:
            raise ValueError('Model counts disagree with complete actual traces')
        mean = np.average(rows.x, axis=0, weights=rows.weight)
        scale = np.sqrt(np.average((rows.x-mean)**2, axis=0, weights=rows.weight))
        scale[scale < artifact['method']['minimum_scale']] = 1.
        if not np.array_equal(mean, model['mean']) or not np.array_equal(scale, model['scale']):
            raise ValueError('Training-only normalizer does not reproduce exactly')
        result['horizons'][str(horizon)] = dict(train_counts=rows.counts,
            calibration_hold_counts=cal.counts, train_fit=model['train_fit'], calibration_fit=model['calibration_fit'],
            normalization_exact=True, class_gates_passed=True)
    return predictor, result


def development(args, out):
    from src.control.service_v2.study import load_service_episodes
    from src.control.service_v2.analysis import analyze_collection, forecast_scores
    aliases = dict(train=args.train_alias, calibration=args.calibration_alias, development=args.development_alias)
    predictor, model = model_audit(args.study, args.source_root, aliases)
    new_json(out/'model-audit.json', model)
    phase, validation, root = phase_inputs(args.study, 'development', args.development_alias, args.source_root)
    if not validation['complete']:
        raise ValueError('Incomplete development; survivor-only diagnosis forbidden')
    episodes = load_service_episodes(root, 'development')
    predictor.validate_scenarios([dict(layout_seed=e.layout_seed, radio_seed=e.radio_seed, config=asdict(e.config)) for e in episodes])
    analysis = analyze_collection(phase['_path'], root, out/'collection-analysis')
    forecast = forecast_scores(episodes, predictor)
    new_json(out/'development-forecast.json', forecast)
    new_json(out/'inputs.json', dict(development=bound_inputs(phase,validation,root),
        model_sha256=model['model_sha256'], public_call='forecast_scores(load_service_episodes(run_root, "development"), ServicePredictor.load(model_path))',
        analysis_manifest_sha256=digest(out/'collection-analysis/manifest.json'),
        no_fitting=True, frozen_development_new_model=phase['new_model'],
        shadow_identity_note='Public forecast_scores uses internal a*64/b*64 shadow-controller identity placeholders. They do not identify fitted artifacts. Actual raw model/method/source identities are bound in this inputs artifact.'))
    return dict(schema='service-v2-task4-development-audit-v1', complete=analysis['complete'],
        model_gates_passed=True, planned_development_cells=32,
        note='Heldout diagnostic results remain visible; predictive accuracy and positive efficacy are not engineering gates.')


def strip_timing(decision):
    result = dict(decision)
    result.pop('timing_seconds', None)
    return result


def pilot(args, out):
    import numpy as np
    from src.control.service_v2.analysis import analyze_collection
    from src.control.service_v2.config import ServiceV2Config
    from src.control.service_v2.protocol import REFERENCE_SHA256
    from src.control.service_v2.study import make_controller_factory, read_frames
    from src.control.service_v2.prediction import ServicePredictor
    from src.control.prevention.prediction import LearnedPredictor
    from src.control.prevention.config import PreventionConfig
    from src.control.path_service.controller import PathServiceController
    from src.control.path_service.config import PathServiceConfig
    if args.development_audit is None:
        raise ValueError('Pilot audit requires the completed development audit root')
    development_root=args.development_audit.resolve(strict=True)
    inventory=read_json(development_root/'manifest.json')['files']
    actual={p.relative_to(development_root).as_posix() for p in development_root.rglob('*') if p.is_file()}
    if actual!=set(inventory)|{'manifest.json'}:
        raise ValueError('Development audit inventory mismatch')
    for name,expected in inventory.items():
        target=(development_root/name).resolve(strict=True)
        if not target.is_relative_to(development_root) or digest(target)!=expected:
            raise ValueError('Development audit file identity mismatch')
    development_summary=read_json(development_root/'summary.json')
    development_model=read_json(development_root/'model-audit.json')
    if not development_summary['complete'] or not development_summary['model_gates_passed']:
        raise ValueError('Development/model gate has not passed')
    if development_model['model_sha256']!=digest(args.study/'model/predictor.json'):
        raise ValueError('Development audit model differs from pilot input')
    phase, validation, root = phase_inputs(args.study, 'pilot', args.pilot_alias, args.source_root)
    # Analysis preserves every planned row, including missing/technical outcomes.
    analysis = analyze_collection(phase['_path'], root, out/'pilot-analysis')
    new_json(out/'inputs.json', bound_inputs(phase,validation,root))
    package = Path(phase['_path']).parent
    if digest(args.study/'model/predictor.json') != phase['new_model']:
        raise ValueError('Pilot differs from accepted fitted model')
    predictor = ServicePredictor.load(package/'new-model.json')
    old = LearnedPredictor.load(package/'reference-model.json')
    method = ServiceV2Config()
    factory = make_controller_factory(predictor, old, method, phase['new_model'], phase['method_sha256'])
    issues, records, fixture = [], [], []
    components, process_times, controller_times = {}, [], []
    def issue(cell, step, reason, **detail):
        issues.append(dict(cell_id=cell, step=step, reason=reason, **detail))
    for row in analysis['rows']:
        cell = row['cell_id']
        cell_issue_start=len(issues)
        for attempt in row['attempt_records']:
            process_times.append(attempt['wall_seconds'])
        if row['status'] != 'ok':
            issue(cell, None, 'unavailable_planned_cell', status=row['status'])
            continue
        if row['policy']=='hold':
            fixture.append(dict(cell_id=cell, regime=row['regime'], context=row['context'],
                layout_seed=row['layout_seed'], radio_seed=row['radio_seed'],
                first_armed_step=row['endpoints']['first_armed_step'],
                armed_by20=bool(row['endpoints']['armed_by_step20_numerator']),
                received=row['application_received_packets'], offered=row['application_offered_packets']))
        config = PreventionConfig(**row['config'])
        attempt = root/cell/f"attempt-{row['attempt']}"
        episode = attempt/row['episode_path']
        decisions_paths = list(episode.parent.glob('decisions.jsonl'))
        if len(decisions_paths)!=1:
            raise ValueError('Expected exactly one sibling decision trace')
        decisions = jsonl(decisions_paths[0])
        frames = read_frames(episode, config)
        transitions = jsonl(episode/'transitions.jsonl')
        if len(decisions)!=config.steps or [d['step'] for d in decisions]!=list(range(config.steps)):
            raise ValueError('Decision trace is not complete')
        timings = read_json(attempt/'timings.json')
        if len(timings)!=config.steps or [t['step'] for t in timings]!=list(range(config.steps)):
            issue(cell, None, 'incomplete_controller_timing')
        controller_times.extend(t['decision_seconds'] for t in timings)
        replay = factory(row['policy'], config)
        direct = PathServiceController('path-service',config,old,method=PathServiceConfig(),model_hash=REFERENCE_SHA256) if row['policy']=='path-service-v1-reference' else None
        v2 = row['policy'].startswith('service-v2')
        gated = v2 and row['policy']!='service-v2-no-motion-gate'
        previous_burst = None
        previous_plan = None
        moving_run = 0
        max_residual = 0.
        max_travel_residual = 0.
        rejections = Counter()
        starts = []
        for step, (frame, decision, transition) in enumerate(zip(frames,decisions,transitions[1:])):
            action, replay_decision = replay.act(frame)  # prefix only: never a future frame
            actual_action = np.asarray(transition['proposed_action'],float)
            if not np.array_equal(action, actual_action) or strip_timing(replay_decision)!=strip_timing(decision):
                issue(cell,step,'stored_action_or_decision_replay_mismatch')
            if direct is not None:
                da, dd = direct.act(frame)
                if not np.array_equal(da,action) or dd!=replay_decision:
                    issue(cell,step,'exact_v1_factory_parity_mismatch')
            following = frames[step+1]
            p, q = np.asarray(frame['positions'],float), np.asarray(following['positions'],float)
            travelled = np.asarray(following['travel'],float)-np.asarray(frame['travel'],float)
            # Existing wire encodes action with .9g. Frame positions/travel are
            # decimal snapshots; retain raw residuals and use a 1e-5 m evidence
            # tolerance only, without changing the controller's 1e-8 gates.
            wire = np.asarray([float(format(float(x),'.9g')) for x in actual_action]).reshape(-1,3)
            predicted = np.zeros_like(p)
            for index, node in enumerate(config.relay_ids):
                scale = min(config.max_speed,max(0,config.budget-frame['travel'][node]))
                predicted[node] = scale*wire[index]/max(1,float(np.linalg.norm(wire[index])))
                clipped = np.clip(p[node]+predicted[node],[0,0,config.z_min],[config.x_max,config.y_max,config.z_max])
                predicted[node] = clipped-p[node]
            if following['safety_rejected']:
                predicted[:] = 0
            relay_delta = q[list(config.relay_ids)]-p[list(config.relay_ids)]
            residual = float(np.max(np.abs(relay_delta-predicted[list(config.relay_ids)])))
            travel_residual = max(abs(float(travelled[i])-float(np.linalg.norm(q[i]-p[i]))) for i in config.relay_ids)
            max_residual = max(max_residual,residual)
            max_travel_residual = max(max_travel_residual,travel_residual)
            if residual>1e-5 or travel_residual>1e-5:
                issue(cell,step,'actual_displacement_or_travel_mismatch',position_residual_m=residual,travel_residual_m=travel_residual)
            if any(travelled[i]<-1e-5 or following['travel'][i]>config.budget+1e-5 for i in config.relay_ids):
                issue(cell,step,'budget_violation')
            if np.any(q < np.array([0,0,config.z_min])-1e-5) or np.any(q > np.array([config.x_max,config.y_max,config.z_max])+1e-5):
                issue(cell,step,'bounds_violation')
            if any(np.linalg.norm(q[i]-q[j])<config.min_separation-1e-5 for i in range(config.num_uavs) for j in range(i)):
                issue(cell,step,'measured_separation_violation')
            moving = bool(np.any(actual_action))
            moving_run = moving_run+1 if moving else 0
            if v2:
                for name,value in decision['timing_seconds'].items():
                    components.setdefault(name,[]).append(value)
                rejections.update(decision['rejection_counts'])
                if np.max(np.linalg.norm(relay_delta,axis=1))>method.requested_speed+1e-5:
                    issue(cell,step,'v2_requested_speed_exceeded')
                if decision['model_hash']!=phase['new_model'] or decision['method_hash']!=phase['method_sha256']:
                    issue(cell,step,'decision_model_or_method_identity')
                if decision['start']:
                    starts.append(dict(step=step,context=decision['context'],reason=decision['reason']))
                    if decision['context']=='forecast':
                        state = decision['service_state']
                        support = state['ever_armed'] and state['valid_history'] if row['policy']=='service-v2-no-online-service-state' else state['forecast_eligible']
                        if not support or row['policy']=='service-v2-no-forecast':
                            issue(cell,step,'forbidden_forecast_start')
                        if decision['risk_crossing_horizon'] is None or decision['risk_crossing_horizon']>decision['urgency_limit']+method.numerical_tolerance:
                            issue(cell,step,'forecast_urgency_violation')
                        if gated and (len(decision['crossing_steps'])!=2 or decision['crossing_steps']!=[step-1,step]):
                            issue(cell,step,'forecast_two_frame_violation')
                    plan=decision['plan']
                    if not plan or not plan['accepted']:
                        issue(cell,step,'start_without_accepted_plan')
                    elif gated and (plan['total_motion']<=0 or plan['gain']<2-method.numerical_tolerance or plan['efficiency']<.10-method.numerical_tolerance):
                        issue(cell,step,'start_benefit_gate_violation')
                if gated and moving_run>method.burst_commands:
                    issue(cell,step,'burst_command_cap_violation')
                if gated and moving and moving_run>1 and not decision['start']:
                    if previous_plan is None or any(decision['plan'][name]!=previous_plan[name] for name in ('chain','targets')):
                        issue(cell,step,'plan_changed_inside_burst')
                if gated and moving and previous_burst is not None and step-previous_burst>1 and step-previous_burst<=method.cooldown_intervals:
                    issue(cell,step,'five_zero_intervals_violation',previous_moving_step=previous_burst)
                if moving:
                    previous_burst=step
                    previous_plan=decision['plan']
        # Feed the final frame only after every executed command was audited;
        # this exercises the next-observation execution ledger, including step239.
        if v2:
            replay.act(frames[-1])
        records.append(dict(cell_id=cell,policy=row['policy'],steps=len(decisions),
            source_trace_sha256=row['trace_sha256'],decision_trace_sha256=digest(decisions_paths[0]),
            max_position_prediction_residual_m=max_residual,max_travel_position_residual_m=max_travel_residual,
            exact_prefix_replay=not any(i['reason']=='stored_action_or_decision_replay_mismatch' for i in issues[cell_issue_start:]),
            exact_v1_factory_parity=None if direct is None else not any(i['reason']=='exact_v1_factory_parity_mismatch' for i in issues[cell_issue_start:]),
            starts=starts,candidate_rejection_counts=dict(rejections),
            actual_udp_pdr=row['actual_udp_pdr'],total_relay_travel_m=row['total_relay_travel_m']))
    new_json(out/'engineering-audit.json',dict(schema='service-v2-engineering-audit-v1',
        issues=issues,cell_checks=records,hold_fixture=fixture,
        displacement_evidence_tolerance_m=1e-5,controller_gate_tolerance_unchanged=1e-8,
        timing=dict(controller_seconds=quantiles(controller_times),subprocess_seconds=quantiles(process_times),
                    components_seconds={name:quantiles(values) for name,values in components.items()})))
    fixture_pass = len(fixture)==6 and all(f['armed_by20'] for f in fixture)
    complete = validation['complete'] and validation['planned_cells']==48
    gate = dict(schema='service-v2-preformal-gate-candidate-v1', engineering_complete=complete,
        healthy_start_240m_all_six_hold_by20=fixture_pass, engineering_issues=len(issues),
        engineering_checks_passed=complete and fixture_pass and not issues,
        parent_adoption=False, independent_specification_review='pending',independent_quality_review='pending',
        formal_authorized=False,formal_outcomes_read=False,positive_efficacy_required=False,
        accepted_identities=bound_inputs(phase,validation,root),
        development_audit_root=str(development_root),development_audit_manifest_sha256=digest(development_root/'manifest.json'),
        development_model_gates_passed=True,development_176_inventory=dict(
            **development_model['phases'],development=read_json(development_root/'inputs.json')['development']),
        synchronous_bridge='Simulator progression blocks during centralized pipe decisions; processing time is not modeled control-plane delay.',
        limitations=['3s routing allowance is a planning hypothesis.', 'No latency, energy, endurance or distributed-implementation claim.',
            'Shared scenario seeds do not match packet-level randomness.', 'Negative efficacy and undefined metrics remain visible.'])
    new_json(out/'preformal-gate-candidate.json',gate)
    report=['# Task 4 engineering evidence','',
        'Status: candidate evidence; independent reviews and parent adoption pending.','',
        f"Pilot complete: {complete}; all six HOLD arms by20: {fixture_pass}; engineering issues: {len(issues)}.",'',
        'The full cell, attempt, failure, timing, motion, and identity inventories are in engineering.json and engineering-audit.json.',
        'Pilot descriptive PDR/motion, paired points and negative effects are retained in pilot-analysis/analysis.json.',
        'The synchronous bridge blocks simulator progression while deciding; processing timings do not model radio/control delay.',
        'Positive efficacy is not a gate. No formal outcome was read.','', '## HOLD fixture','',
        '| Cell | First armed step | Armed by20 | Received/offered |','| --- | ---: | --- | ---: |']
    report.extend(f"| {f['cell_id']} | {f['first_armed_step']} | {f['armed_by20']} | {f['received']}/{f['offered']} |" for f in fixture)
    with (out/'engineering-report.md').open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(report)+'\n')
    return dict(schema='service-v2-engineering-v1',pilot_gate_candidate=gate,
        audit=read_json(out/'engineering-audit.json'),timing=analysis['timing'],
        complete_inventory=analysis['rows'], descriptive=analysis['descriptive'],contrasts=analysis['contrasts'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('development','pilot'))
    for name in ('source-root','study','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for phase in ('train','calibration','development'):
        parser.add_argument('--'+phase+'-alias',default=phase+'-runtime-v002')
    parser.add_argument('--pilot-alias',default='pilot')
    parser.add_argument('--development-audit',type=Path)
    args=parser.parse_args()
    args.source_root=args.source_root.resolve(strict=True)
    args.study=args.study.resolve(strict=True)
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.source_root))
    import numpy as np
    import src.control.service_v2.protocol as protocol
    if not Path(protocol.__file__).resolve().is_relative_to(args.source_root):
        raise ValueError('Imports are outside specified frozen runtime')
    new_json(args.output/'invocation.json',dict(schema='service-v2-task4-invocation-v1',
        argv=sys.argv,cwd=str(Path.cwd()),source_root=str(args.source_root),helper_path=str(Path(__file__).resolve()),
        helper_sha256=digest(__file__),python_executable=sys.executable,python_executable_sha256=digest(sys.executable),
        python_version=sys.version,numpy_version=np.__version__,platform=platform.platform(),
        mode=args.mode,automatic_simulation=False,fitting=False,formal_outcomes_read=False))
    started=time.monotonic()
    try:
        value=development(args,args.output) if args.mode=='development' else pilot(args,args.output)
        new_json(args.output/('engineering.json' if args.mode=='pilot' else 'summary.json'),value)
        new_json(args.output/'completed.json',dict(status='artifact_generation_completed',wall_seconds=time.monotonic()-started,
            note='Artifact completion does not adopt the preformal gate.'))
        manifest(args.output,'service-v2-engineering-manifest-v1' if args.mode=='pilot' else 'service-v2-development-diagnostic-manifest-v1')
    except BaseException as error:
        new_json(args.output/'failed.json',dict(error=f'{type(error).__name__}: {error}',wall_seconds=time.monotonic()-started))
        raise


if __name__=='__main__':
    main()
