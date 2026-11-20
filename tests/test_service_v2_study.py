import csv
import json
from pathlib import Path

import pytest
import numpy as np

from tests.service_v2_support import fixture_config, service_frames
from src.control.service_v2.study import reconcile_episode, make_controller_factory
from src.control.service_v2.config import POLICIES, ServiceV2Config
from tests.test_service_v2_protocol import frozen


def episode(tmp_path):
    frames=service_frames([44]*168+[43]*72, offers=[63]*120+[62]*120); frames[-1]['terminal']=True
    (tmp_path/'transitions.jsonl').write_text(''.join(json.dumps(dict(frame=f))+'\n' for f in frames))
    final=dict(schema='service-final-accounting-v1',counting_origin_s=0,sim_time_s=251.,offered_packets=15000,source_tx_packets=15000,sink_rx_packets=10488,
        flowmonitor_rx_packets=10496,last_live_frame_time_s=251.,offered_after_last_frame=0,
        source_tx_after_last_frame=0,late_sink_rx_after_last_frame=0)
    (tmp_path/'service_final_accounting.json').write_text(json.dumps(final))
    (tmp_path/'closed_loop_metrics.csv').write_text('offered_packets,tx_packets,rx_packets,pdr\n15000,15000,10496,0.6997333333333334\n')
    return tmp_path


def test_cross_layer_difference_is_reported(tmp_path):
    p=episode(tmp_path); row=reconcile_episode(p,fixture_config())
    assert row['application_received_packets']==10488
    assert row['flowmonitor_received_packets']==10496
    assert row['sink_minus_flowmonitor_received']==-8
    final=json.loads((p/'service_final_accounting.json').read_text()); final['offered_packets']+=1
    (p/'service_final_accounting.json').write_text(json.dumps(final))
    with pytest.raises(ValueError): reconcile_episode(p,fixture_config())


def test_missing_last_frame_refused(tmp_path):
    p=episode(tmp_path); lines=(p/'transitions.jsonl').read_text().splitlines()
    (p/'transitions.jsonl').write_text('\n'.join(lines[:-1]))
    with pytest.raises(ValueError): reconcile_episode(p,fixture_config())


def test_reference_factory_exact_parity():
    from src.control.prevention.prediction import LearnedPredictor
    from src.control.path_service.controller import PathServiceController
    from src.control.path_service.config import PathServiceConfig
    model=LearnedPredictor.load('reports/uav_prevention_pilot_2026-11-11/predictor.json')
    factory=make_controller_factory(None,model,ServiceV2Config(),None,'a'*64)
    from src.control.service_v2.protocol import REFERENCE_SHA256
    direct=PathServiceController('path-service',fixture_config(),model,method=PathServiceConfig(),model_hash=REFERENCE_SHA256)
    wrapped=factory('path-service-v1-reference',fixture_config(),threshold=.99)
    for f in service_frames([60]*15):
        a,d=direct.act(f); b,e=wrapped.act(f)
        np.testing.assert_array_equal(a,b); assert d==e


@pytest.fixture(scope='module')
def collection(frozen,tmp_path_factory):
    from src.control.service_v2.protocol import load_phase,write_json,sha256,publish_status
    from src.control.service_v2.study import _collection_manifest
    root=tmp_path_factory.mktemp('v2-complete'); phase=load_phase(frozen)
    _collection_manifest(root,phase)
    for c in phase['cells']:
        a=root/c['cell_id']/'attempt-0'; a.mkdir(parents=True)
        e=a/'collector/episodes'/c['policy']/'episode_fixture'; e.mkdir(parents=True)
        frames=service_frames([60]*240,offers=[63]*120+[62]*120)
        if c['regime']=='n5':
            for f in frames:
                f['positions'].append([444.,600.,100.]); f['velocities'].append([0.,0.,0.]); f['travel'].append(0.)
                for name in ('rx_ratio','rssi','rssi_mask'):
                    for row in f[name]: row.append(True if name=='rssi_mask' else 1. if name=='rx_ratio' else -70.)
                    f[name].append([False if name=='rssi_mask' else 0.]*5)
                s=f['service_observation']; s['flow']['destination']=4; s['route']['nodes']=[0,4]; s['route_probe_hops'][0]['to']=4
        frames[-1]['terminal']=True
        (e/'transitions.jsonl').write_text(''.join(json.dumps(dict(frame=f))+'\n' for f in frames))
        write_json(e/'service_final_accounting.json',dict(schema='service-final-accounting-v1',counting_origin_s=0,sim_time_s=251.,offered_packets=15000,source_tx_packets=15000,sink_rx_packets=14400,
            flowmonitor_rx_packets=14400,last_live_frame_time_s=251.,offered_after_last_frame=0,
            source_tx_after_last_frame=0,late_sink_rx_after_last_frame=0))
        (e/'closed_loop_metrics.csv').write_text('offered_packets,tx_packets,rx_packets,pdr\n15000,15000,14400,.96\n')
        write_json(e/'config.json',dict(config=c['config'],layout_seed=c['layout_seed'],radio_seed=c['radio_seed']))
        write_json(e/'policy_metadata.json',dict(collection_policy=c['policy'],split='test',layout_seed=c['layout_seed'],radio_seed=c['radio_seed']))
        for name in ('simulator.stdout.log','simulator.stderr.log'): (e/name).write_bytes(b'fixture only; no process')
        (e.parent/'decisions.jsonl').write_text(''.join(json.dumps(dict(step=i,commanded_motion=False,intervention_started=False))+'\n' for i in range(240)))
        write_json(a/'timings.json',[dict(step=i,decision_seconds=.001) for i in range(240)])
        from src.control.prevention.config import PreventionConfig
        result=dict(cell=c,attempt=0,freeze_sha256=phase['_freeze_sha256'],status='ok',actual_exit=0,error=None,
            started_unix=0.,finished_unix=1.,wall_seconds=1.,episode_path=e.relative_to(a).as_posix(),
            trace_sha256=sha256(e/'transitions.jsonl'),**reconcile_episode(e,PreventionConfig(**c['config'])))
        write_json(a/'result.json',result)
        write_json(a/'artifacts.json',{p.relative_to(a).as_posix():sha256(p) for p in a.rglob('*') if p.is_file()})
        publish_status(a,dict(status='ok',actual_exit=0,validated=True,result_sha256=sha256(a/'result.json'),artifacts_sha256=sha256(a/'artifacts.json')))
    return frozen,root


def test_full_manifest_loader_validates_before_selection(collection):
    from src.control.service_v2.study import load_service_episodes,validate_collection
    protocol,root=collection
    assert validate_collection(protocol,root)['complete']
    assert len(load_service_episodes(root,'development',['hold']))==16
    other=next((root/p/'attempt-0/result.json' for p in json.loads((root/'collection.json').read_text())['cells'] if p.endswith('__geometry')))
    before=other.read_bytes()
    try:
        other.write_bytes(b'{}')
        with pytest.raises(ValueError): load_service_episodes(root,'development',['hold'])
    finally: other.write_bytes(before)


def test_unlisted_trace_and_missing_cell_refused(collection):
    from src.control.service_v2.study import load_service_episodes
    protocol,root=collection
    path=root/'unlisted'; path.mkdir()
    try:
        (path/'transitions.jsonl').write_text('{}')
        with pytest.raises(ValueError): load_service_episodes(root,'development',['hold'])
    finally:
        (path/'transitions.jsonl').unlink(); path.rmdir()
    cell=next(p for p in root.iterdir() if p.is_dir()); renamed=root.parent/'retained-cell'
    cell.rename(renamed)
    try:
        with pytest.raises(ValueError,match='Incomplete'): load_service_episodes(root,'development',['hold'])
    finally: renamed.rename(cell)


@pytest.mark.parametrize('field,value',[('actual_exit',7),('trace_sha256','0'*64)])
def test_forged_success_refused_even_with_rehashed_result(collection,field,value):
    from src.control.service_v2.study import load_service_episodes
    from src.control.service_v2.protocol import sha256
    _,root=collection; a=next(p for p in root.iterdir() if p.is_dir())/'attempt-0'
    paths=[a/'result.json',a/'artifacts.json',a/'status.json',a/'status-history/000001.json']
    raw={p:p.read_bytes() for p in paths}
    try:
        r=json.loads(raw[paths[0]]); r[field]=value; paths[0].write_text(json.dumps(r))
        art=json.loads(raw[paths[1]]); art['result.json']=sha256(paths[0]); paths[1].write_text(json.dumps(art))
        s=json.loads(raw[paths[2]]); s['result_sha256']=sha256(paths[0]); s['artifacts_sha256']=sha256(paths[1])
        if field=='actual_exit': s[field]=value
        paths[2].write_text(json.dumps(s)); paths[3].write_text(json.dumps(s))
        with pytest.raises(ValueError): load_service_episodes(root,'development')
    finally:
        for p,b in raw.items(): p.write_bytes(b)


def test_rc0_without_terminal_data_is_refused(tmp_path):
    p=episode(tmp_path); (p/'service_final_accounting.json').unlink()
    with pytest.raises(OSError): reconcile_episode(p,fixture_config())


def failed_collection(frozen,tmp_path,*,status='technical_error',exit_code=7,streams=('stdout',),evidence=None):
    from src.control.service_v2.protocol import load_phase,write_json,sha256,publish_status
    from src.control.service_v2.study import _collection_manifest
    root=tmp_path/'failed-collection'; root.mkdir(); phase=load_phase(frozen)
    _collection_manifest(root,phase); cell=phase['cells'][0]
    attempt=root/cell['cell_id']/'attempt-0'; attempt.mkdir(parents=True)
    episode=attempt/'collector/episode_fixture'; episode.mkdir(parents=True)
    for stream in streams: (episode/f'simulator.{stream}.log').write_bytes(b'retained failed-process fixture output')
    result=dict(cell=cell,attempt=0,freeze_sha256=phase['_freeze_sha256'],status=status,actual_exit=exit_code,
        error='Fixture prelaunch/setup failure; no simulator is run by this test.',
        episode_path=episode.relative_to(attempt).as_posix(),wall_seconds=0.)
    if evidence is not None:
        write_json(attempt/'process-evidence.json',evidence)
        result['process_evidence_path']='process-evidence.json'
    write_json(attempt/'result.json',result)
    write_json(attempt/'artifacts.json',{p.relative_to(attempt).as_posix():sha256(p) for p in attempt.rglob('*') if p.is_file()})
    publish_status(attempt,dict(status=status,actual_exit=exit_code,validated=False,result_sha256=sha256(attempt/'result.json'),artifacts_sha256=sha256(attempt/'artifacts.json')))
    return root,attempt


@pytest.mark.parametrize('status',['technical_error','interrupted'])
def test_failed_attempt_missing_stderr_is_refused(frozen,tmp_path,status):
    from src.control.service_v2.study import validate_collection
    root,_=failed_collection(frozen,tmp_path,status=status)
    with pytest.raises(ValueError,match='simulator.*log|process.*evidence'):
        validate_collection(frozen,root)


def test_failed_attempt_with_both_logs_retains_direct_outcome(frozen,tmp_path):
    from src.control.service_v2.study import validate_collection
    root,_=failed_collection(frozen,tmp_path,streams=('stdout','stderr'))
    result=validate_collection(frozen,root)
    assert not result['complete']
    assert result['rows'][0]['attempt_records'][0]['actual_exit']==7


def test_prelaunch_no_process_requires_explicit_evidence(frozen,tmp_path):
    from src.control.service_v2.study import validate_collection
    root,_=failed_collection(frozen,tmp_path,exit_code=None,streams=())
    with pytest.raises(ValueError,match='process.*evidence'):
        validate_collection(frozen,root)


def test_legitimate_prelaunch_no_process_evidence(frozen,tmp_path):
    from src.control.service_v2.study import validate_collection
    evidence=dict(schema='service-v2-process-evidence-v1',process_started=False,actual_exit=None,
        outcome='not_started',error='Controller/setup failed before environment construction; no process existed.',
        streams={name:dict(status='unavailable',reason='process_not_started',error='No process was created.') for name in ('stdout','stderr')})
    root,attempt=failed_collection(frozen,tmp_path,exit_code=None,streams=(),evidence=evidence)
    result=validate_collection(frozen,root)
    assert not result['complete'] and result['rows'][0]['attempt_records'][0]['actual_exit'] is None
    # A direct-outcome contradiction must fail even after consistent outer
    # artifact/status rehashing, rather than being hidden by those hashes.
    from src.control.service_v2.protocol import sha256
    path=attempt/'process-evidence.json'; value=json.loads(path.read_text()); value['actual_exit']=9; path.write_text(json.dumps(value))
    art=attempt/'artifacts.json'; value=json.loads(art.read_text()); value['process-evidence.json']=sha256(path); art.write_text(json.dumps(value))
    status=attempt/'status.json'; value=json.loads(status.read_text()); value['artifacts_sha256']=sha256(art)
    status.write_text(json.dumps(value)); (attempt/'status-history/000001.json').write_text(json.dumps(value))
    with pytest.raises(ValueError,match='process.*outcome|process.*evidence'):
        validate_collection(frozen,root)


@pytest.mark.parametrize('started,exit_code',[(False,None),(True,9),(True,0)])
def test_process_capture_writer_records_unavailable_streams(tmp_path,started,exit_code):
    from src.control.service_v2.study import _record_process_evidence,_validate_process_evidence
    from src.control.service_v2.protocol import sha256,strict_json
    result=dict(status='technical_error',actual_exit=exit_code,error='Hand fixture setup/capture error; no process launched.')
    _record_process_evidence(tmp_path,result,started,[])
    artifacts={'process-evidence.json':sha256(tmp_path/'process-evidence.json')}
    _validate_process_evidence(tmp_path,result,artifacts)
    evidence=strict_json(tmp_path/'process-evidence.json')
    assert evidence['actual_exit']==exit_code and evidence['process_started'] is started
    assert all(e['status']=='unavailable' and e['error'] for e in evidence['streams'].values())
    if exit_code==0:
        # A validation failure may have direct exit0; JSON false is not that
        # integer outcome, despite Python's False == 0 comparison.
        path=tmp_path/'process-evidence.json'; evidence['actual_exit']=False
        path.write_text(json.dumps(evidence)); artifacts['process-evidence.json']=sha256(path)
        with pytest.raises(ValueError,match='process outcome'):
            _validate_process_evidence(tmp_path,result,artifacts)


@pytest.fixture(scope='module')
def collector_integration(tmp_path_factory):
    """Actual unchanged collector over hand measurements; no process or fit."""
    from copy import deepcopy
    from dataclasses import asdict
    from src.control.prevention.evaluation import run_grid
    from src.control.prevention.prediction import LearnedPredictor
    from src.control.service_v2.config import POLICIES
    from src.control.service_v2.prediction import ServicePredictor
    from src.control.service_v2.protocol import sha256
    model=Path('reports/uav_service_prevention_v2_2026-11-12/model/predictor.json')
    expected='719816f80b9df51c5544422522265e50bd454768869523036a8b8f4b1c657c10'
    assert sha256(model)==expected
    predictor=ServicePredictor.load(model)
    old=LearnedPredictor.load('reports/uav_prevention_pilot_2026-11-11/predictor.json')
    factory=make_controller_factory(predictor,old,ServiceV2Config(),expected,'b'*64)
    config=fixture_config(); environments={}

    class MeasuredFixtureEnv:
        def __init__(self,config,launcher,output_root):
            assert launcher==['fixture-no-process']
            self.config=config; self.root=Path(output_root)
            self.policy=self.root.parent.name
            self.episode_output_dir=self.root/'episode_hand_measurements'
            self.episode_output_dir.mkdir()
            environments[self.policy]=self
            self.frames=service_frames([60]*30+[0]*15+[60]*195,offers=[63]*120+[62]*120)
            self.frames[0]['positions']=[[300.,500.,100.],[336.,600.,100.],[372.,600.,100.],[408.,700.,100.]]
            self.actions=[]; self.last_frame=None
        def record(self,row):
            with (self.episode_output_dir/'transitions.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
        def reset(self,options):
            self.last_frame=self.frames[0]
            (self.episode_output_dir/'config.json').write_text(json.dumps(dict(config=asdict(config),**options)))
            for stream in ('stdout','stderr'):
                (self.episode_output_dir/f'simulator.{stream}.log').write_text('Hand fixture; no simulator/process created.\n')
            self.record(dict(event='initial',frame=self.last_frame))
            return None,{}
        def step(self,action):
            previous=self.last_frame; k=previous['step']+1
            sent=np.asarray(action).copy(); self.actions.append(sent)
            frame=self.frames[k]
            positions=np.asarray(previous['positions']).copy()
            travel=np.asarray(previous['travel']).copy()
            velocity=np.zeros_like(positions)
            rejected=k==35 and bool(np.any(sent))
            for node,command in zip(config.relay_ids,sent.reshape(-1,3)):
                # Independent literal LimitMotion arithmetic, with one declared
                # measured rejection. It is a deterministic test, not ns-3.
                scale=min(config.max_speed,max(0.,config.budget-travel[node]))
                displacement=scale*command/max(1.,float(np.linalg.norm(command)))
                target=np.clip(positions[node]+displacement,[0.,0.,config.z_min],
                    [config.x_max,config.y_max,config.z_max])
                delta=np.zeros(3) if rejected else target-positions[node]
                positions[node]+=delta; travel[node]+=np.linalg.norm(delta); velocity[node]=delta
            if previous['step']>=30:
                for node,dy in ((0,-1.),(3,1.)):
                    positions[node,1]+=dy; velocity[node,1]=dy
            frame.update(positions=positions.tolist(),velocities=velocity.tolist(),travel=travel.tolist(),
                safety_rejected=rejected,terminal=k==config.steps)
            self.last_frame=frame
            self.record(dict(event='transition',previous_step=k-1,proposed_action=sent.tolist(),frame=frame))
            if k==config.steps:
                s=frame['service_observation']; c=s['cumulative']; offered=c['offered_packets']; received=c['sink_rx_packets']
                (self.episode_output_dir/'closed_loop_metrics.csv').write_text(
                    f'offered_packets,tx_packets,rx_packets,pdr,mean_delay_ms\n{offered},{offered},{received},{received/offered},1.0\n')
                (self.episode_output_dir/'service_final_accounting.json').write_text(json.dumps(dict(
                    schema='service-final-accounting-v1',counting_origin_s=0,sim_time_s=251.,
                    offered_packets=offered,source_tx_packets=offered,sink_rx_packets=received,
                    flowmonitor_rx_packets=received,last_live_frame_time_s=251.,offered_after_last_frame=0,
                    source_tx_after_last_frame=0,late_sink_rx_after_last_frame=0)))
            return None,0.,False,k==config.steps,{}
        def close(self): pass

    # Capture raw controller outputs before the factory's adapter sees them,
    # proving all original fields and actions survive without mutation.
    from src.control.service_v2.controller import ServiceV2Controller
    raw={policy:[] for policy in POLICIES[4:]}
    original=ServiceV2Controller.act
    def observed(controller,frame):
        action,decision=original(controller,frame)
        raw[controller.policy].append((action.copy(),deepcopy(decision)))
        return action,decision
    root=tmp_path_factory.mktemp('real-collector-hand-measurements')/'collector'
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ServiceV2Controller,'act',observed)
        summary=run_grid(config=config,launcher=['fixture-no-process'],output=root,policies=list(POLICIES),
            scenarios=[(11,22)],predictor=None,env_factory=MeasuredFixtureEnv,
            controller_factory=factory,supported_policies=POLICIES)
    assert sha256(model)==expected
    return root,summary,environments,raw,old


@pytest.mark.parametrize('policy',POLICIES)
def test_actual_collector_full_horizon_decision_contract(collector_integration,policy):
    from src.control.service_v2.config import POLICIES
    from src.control.prevention.controller import PreventionController
    from src.control.path_service.controller import PathServiceController
    from src.control.path_service.config import PathServiceConfig
    from src.control.service_v2.protocol import strict_json,REFERENCE_SHA256
    root,summary,environments,raw,old=collector_integration
    record=next(r for r in strict_json(root/'episodes.json') if r['policy']==policy)
    assert record['status']=='ok',record['error']
    env=environments[policy]
    decisions=[json.loads(line) for line in (env.root/'decisions.jsonl').read_text().splitlines()]
    assert [d['step'] for d in decisions]==list(range(240))
    assert [f['step'] for f in env.frames]==list(range(241)) and env.frames[-1]['terminal']
    assert len(env.actions)==240
    assert record['intervention_starts']==sum(d['intervention_started'] for d in decisions)
    assert record['first_action_step']==next((d['step'] for d in decisions if d['commanded_motion']),None)
    assert reconcile_episode(env.episode_output_dir,fixture_config())['total_relay_travel_m']==pytest.approx(record['total_relay_travel'])
    if policy in POLICIES[:4]:
        direct=(PathServiceController('path-service',fixture_config(),old,method=PathServiceConfig(),model_hash=REFERENCE_SHA256)
            if policy=='path-service-v1-reference' else PreventionController(policy,fixture_config()))
        for frame,action,decision in zip(env.frames,env.actions,decisions):
            expected,diagnostic=direct.act(frame)
            np.testing.assert_array_equal(action,expected)
            assert decision==json.loads(json.dumps(diagnostic))
        return
    assert len(raw[policy])==240
    for step,((action,original),decision) in enumerate(zip(raw[policy],decisions)):
        np.testing.assert_array_equal(env.actions[step],action)
        assert {key:decision[key] for key in original}==json.loads(json.dumps(original))
        assert set(decision)==set(original)|{'intervention_started','commanded_motion','collector_compatibility'}
        assert decision['intervention_started'] is original['start']
        assert decision['commanded_motion'] is bool(np.any(action))
        assert decision['sent_action']==action.tolist()
        # Include command239 -> terminal frame240, which has no subsequent
        # controller.act call to publish a previous_command entry.
        following=env.frames[step+1]
        executed=(np.asarray(following['positions'])-np.asarray(env.frames[step]['positions']))[list(fixture_config().relay_ids)]
        np.testing.assert_allclose(executed,
            np.zeros_like(executed) if following['safety_rejected'] else decision['predicted_displacement'],atol=1e-8)
        if step:
            ledger=decision['previous_command']; before=env.frames[step-1]; after=env.frames[step]
            assert ledger['command_step']==step-1 and ledger['observation_step']==step
            assert ledger['sent_action']==env.actions[step-1].tolist()
            np.testing.assert_allclose(ledger['executed_position_delta'],
                (np.asarray(after['positions'])-np.asarray(before['positions']))[list(fixture_config().relay_ids)])
            np.testing.assert_allclose(ledger['executed_travel_delta'],
                (np.asarray(after['travel'])-np.asarray(before['travel']))[list(fixture_config().relay_ids)])
            assert ledger['observed_safety_rejected'] is after['safety_rejected']
            if not after['safety_rejected']:
                np.testing.assert_allclose(ledger['executed_position_delta'],ledger['predicted_displacement'],atol=1e-8)
    assert any(d['commanded_motion'] for d in decisions)
    assert any(d['previous_command'] and d['previous_command']['observed_safety_rejected'] for d in decisions)
    assert summary['paired_valid'] and summary['errors']==[]


@pytest.mark.parametrize('start,action',[(False,[0.,0.,0.]),(True,[0.,0.,0.]),
    (False,[.25,0.,0.]),(True,[0.,-.25,0.]),(False,[1e-12,0.,0.])])
def test_collector_aliases_preserve_start_vs_command_vs_execution(start,action):
    from copy import deepcopy
    from src.control.service_v2.study import _collector_decision
    emitted=np.asarray(action)
    raw=dict(start=start,sent_action=action,previous_command=dict(executed_travel_delta=[0.],observed_safety_rejected=True))
    before=deepcopy(raw); converted=_collector_decision(emitted,raw)
    assert raw==before and converted is not raw
    assert converted['intervention_started'] is start
    assert converted['commanded_motion'] is bool(np.any(emitted))
    assert converted['previous_command']==raw['previous_command']


@pytest.mark.parametrize('mutation',['missing_start','invalid_start','missing_action','contradictory_action','existing_alias'])
def test_collector_aliases_refuse_missing_or_contradictory_fields(mutation):
    from src.control.service_v2.study import _collector_decision
    raw=dict(start=False,sent_action=[0.,0.,0.])
    if mutation=='missing_start': raw.pop('start')
    elif mutation=='invalid_start': raw['start']=0
    elif mutation=='missing_action': raw.pop('sent_action')
    elif mutation=='contradictory_action': raw['sent_action']=[1.,0.,0.]
    else: raw['intervention_started']=False
    with pytest.raises((KeyError,ValueError)): _collector_decision(np.zeros(3),raw)


@pytest.mark.parametrize('policy',POLICIES[4:])
def test_retained_failed_pilot_decisions_pass_unchanged_summarizer(policy):
    """Read-only replay of the real failed post-run summary, no launch or fit."""
    from src.control.prevention.config import PreventionConfig
    from src.control.prevention.evaluation import episode_metrics
    from src.control.service_v2.study import _collector_decision,read_frames
    root=Path('reports/uav_service_prevention_v2_2026-11-12/remote-evidence/pilot-failure-v001/runs/pilot')
    attempt=root/f'pilot-n4-static-01__{policy}'/'attempt-0'
    result=json.loads((attempt/'result.json').read_text())
    assert result['status']=='technical_error' and result['actual_exit']==0
    assert "KeyError: 'intervention_started'" in result['error']
    trace=next(attempt.rglob('transitions.jsonl')); config=PreventionConfig(**result['cell']['config'])
    frames=read_frames(trace.parent,config)
    raw=[json.loads(line) for line in next(attempt.rglob('decisions.jsonl')).read_text().splitlines()]
    assert len(raw)==240 and len(frames)==241
    with pytest.raises(KeyError,match='intervention_started'): episode_metrics(frames,config,trace.parent,raw)
    adapted=[_collector_decision(np.asarray(row['sent_action']),row) for row in raw]
    metrics=episode_metrics(frames,config,trace.parent,adapted)
    assert metrics['elapsed_steps']==240
    assert metrics['intervention_starts']==sum(row['start'] for row in raw)
    assert metrics['first_action_step']==next((row['step'] for row in raw if any(row['sent_action'])),None)
