from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pytest
from src.control.path_service.config import PathServiceConfig
from src.control.path_service.controller import PathServiceController
from src.control.path_service.observation import ServiceHistory,validate_service_observation
from src.control.path_service.planner import enumerate_plans,one_step_action,separated
from src.control.path_service.risk import assess_risk
from src.control.prevention.config import PreventionConfig
from src.control.prevention.controller import PreventionController,target_plan
from src.control.prevention.features import edge_features


def config(**kw): return PreventionConfig(num_uavs=5,source=0,destination=4,**kw)


def frame(step=0,route=None,success=True,offers=0,received=0,old_offers=0,old_received=0):
    route=route or [0,1,2,3,4]
    f=dict(version=1,step=step,positions=[[100,200,100],[175,240,100],[250,200,100],[325,240,100],[400,200,100]],
           velocities=[[0,0,0] for _ in range(5)],travel=[0]*5,rx_ratio=[[int(i!=j) for j in range(5)] for i in range(5)],
           rssi=[[0]*5 for _ in range(5)],rssi_mask=[[False]*5 for _ in range(5)],components=1,
           source_dest_connected=True,olsr_route_usable=True,connected_streak=0,safety_rejected=False,terminal=False)
    counts=dict(offered_packets=offers,source_tx_packets=offers,sink_rx_packets=received)
    f['service_observation']=dict(schema='service-observation-v1',sim_time_s=11+step,window_start_s=10+step if step else 0,window_end_s=11+step,
             flow=dict(source=0,destination=4,destination_port=9000,protocol='udp'),
             cumulative=dict(offered_packets=old_offers+offers,source_tx_packets=old_offers+offers,sink_rx_packets=old_received+received),window=counts,
             route=dict(nodes=route,success=success,reason='success' if success else 'missing_entry',failing_node=None if success else route[-1]),
             route_probe_hops=[dict(zip(('from','to','rx_ratio_forward','rx_ratio_reverse'),(a,b,1.,1.))) for a,b in zip(route,route[1:])])
    return f


class Risk:
    def __init__(self,horizon=3,value=.2): self.horizon,self.value=horizon,value
    def predict(self,current,previous,comm_range):
        return {h:{(i,j):(self.value if h>=self.horizon else 0.) for i in range(5) for j in range(i+1,5)} for h in (3,5,10,20)}


def test_nested_counts_origin_booleans_and_extension_compatibility():
    f=frame(); c=config(); validate_service_observation(f,c)
    for mutation in ['bool','window','time','flow','hop']:
        bad=deepcopy(f)
        if mutation=='bool': bad['service_observation']['cumulative']['offered_packets']=True
        elif mutation=='window': bad['service_observation']['window']['offered_packets']=1
        elif mutation=='time': bad['service_observation']['window_start_s']=10
        elif mutation=='flow': bad['service_observation']['flow']['source']=1
        else: bad['service_observation']['route_probe_hops'][0]['to']=4
        with pytest.raises(ValueError): validate_service_observation(bad,c)
    bare={k:v for k,v in f.items() if k!='service_observation'}
    from src.control.bridge import validate_frame,ProtocolError
    validate_frame(bare,c,0)
    with pytest.raises(ProtocolError): validate_frame(bare,c,0,require_service_observation=True)


def test_history_no_offers_and_delayed_arrivals_not_clamped():
    h=ServiceHistory(PathServiceConfig()); c=config()
    assert h.update(frame(),c)['service_unknown']
    total=rx=0
    for step in range(1,4):
        f=frame(step,offers=10,received=12,old_offers=total,old_received=rx);total+=10;rx+=12
        result=h.update(f,c)
    assert result['recent_arrival_offer_ratio']==1.2 and not result['delivery_degraded']
    f=frame(4,old_offers=total,old_received=rx); result=h.update(f,c)
    assert result['service_unknown'] and result['recent_arrival_offer_ratio'] is None


def test_three_window_degradation_restoration_and_route_asymmetry():
    h=ServiceHistory(PathServiceConfig()); c=config(); h.update(frame(),c)
    for step in range(1,4):
        result=h.update(frame(step,offers=10,received=5,old_offers=(step-1)*10,old_received=(step-1)*5),c)
    assert result['restoration'] and result['partial_delivery_proxy']
    for step in range(4,7):
        result=h.update(frame(step,offers=10,received=10,old_offers=(step-1)*10,old_received=15+(step-4)*10),c)
    assert result['recent_service_evidence'] and not result['restoration']
    f=frame(); f['service_observation']['route_probe_hops'][0]['rx_ratio_reverse']=.1
    result=ServiceHistory(PathServiceConfig()).update(f,c)
    assert result['route_degraded'] and result['probe_inconsistent'] and not result['restoration']


def test_observed_path_union_score_and_dijkstra_fallback():
    c=config(); m=PathServiceConfig(); f=frame()
    risk=assess_risk(f,None,c,Risk(value=.1),m)
    assert risk['forecast_path']==[0,1,2,3,4] and risk['scores'][3]==.4
    assert risk['crossing_horizon_s']==3
    f=frame(route=[0],success=False)
    risk=assess_risk(f,None,c,Risk(value=.1),m)
    assert risk['forecast_path']==[0,4] and risk['path_origin']=='measured_min_sum_H20'
    f['rx_ratio']=[[0]*5 for _ in range(5)]
    assert assess_risk(f,None,c,Risk(),m)['crossing_horizon_s'] is None


def test_subset_plan_differs_from_old_all_relay_target_and_bounded_lookahead():
    c=config(); f=frame(); m=PathServiceConfig()
    p=enumerate_plans(f,c,m)['chosen']
    assert p and len(p['chain'])<5 and p['chain'][0]==0 and p['chain'][-1]==4
    action,_=one_step_action(f,c,p)
    assert not np.array_equal(action,target_plan(f,c)[0])
    assert p['lookahead_s']<=20 and p['arrival_time_s']<=120
    end=frame(119); end['velocities'][4]=[100,0,0]
    a=enumerate_plans(end,c,m)
    if a['chosen']: assert a['chosen']['lookahead_s']<=1


def test_budget_ablation_changes_complete_feasibility_but_hard_clips():
    c=config(); f=frame(); f['travel']=[0,400,400,400,0]
    p=enumerate_plans(f,c,PathServiceConfig())
    assert p['chosen'] is None and p['rejection_counts']['complete_budget']>0
    p=enumerate_plans(f,c,PathServiceConfig(),'no-budget')['chosen']
    assert p
    action,diag=one_step_action(f,c,p)
    assert not np.any(action) and diag['budget_truncated_relays']


def test_nonselected_relay_swept_and_static_collision():
    positions=np.array([[0.,0,0],[10.,0,0],[5.,0,0]])
    targets=np.array([[10.,0,0],[0.,0,0],[5.,0,0]])
    assert not separated(positions,targets,8,1,range(3))
    f=frame(); f['positions'][2]=[175,200,100]
    assessment=enumerate_plans(f,config(),PathServiceConfig())
    if assessment['chosen']:
        targets=np.array(assessment['chosen']['targets'])
        assert separated(np.array(f['positions']),targets,8,5,config().relay_ids)


def test_one_step_screen_mirrors_existing_simulator_budget_scale():
    c=config();f=frame();f['travel'][1]=396
    targets=deepcopy(f['positions']);targets[1][1]-=4
    action,d=one_step_action(f,c,dict(chain=[0,1,4],targets=targets))
    assert np.linalg.norm(action[:3])==.5
    assert d['expected_static_anchor_one_step_motion_m']['1']==2.
    assert d['simulator_budget_scaled_relays']==[1]


def test_deadline_ablation_and_grouped_service_path_ablation():
    c=config(); f=frame()
    a,d=PathServiceController('path-service',c,Risk(20)).act(f)
    assert not np.any(a) and d['risk']['crossing_horizon_s']==20
    a,d=PathServiceController('path-service-no-deadline',c,Risk(20)).act(f)
    assert np.any(a) and d['activation_context']=='forecast_prevention'
    a,d=PathServiceController('path-service-no-service',c,Risk(3)).act(f)
    assert d['risk']['path_origin']=='global_bridge_ablation' and d['risk']['crossing_horizon_s'] is None
    assert d['planner']['candidate_count']==1
    a,d=PathServiceController('path-service',c,Risk(3)).act(f)
    assert np.any(a)


def test_causal_future_metadata_and_old_features_actions_identical():
    c=config(); f=frame(); alternative=deepcopy(f); alternative['scenario_future']={'mission_turn':-999,'outcome':0}
    left=PathServiceController('path-service',c,Risk()).act(f)
    right=PathServiceController('path-service',c,Risk()).act(alternative)
    np.testing.assert_array_equal(left[0],right[0]); assert left[1]==right[1]
    bare={k:v for k,v in f.items() if k!='service_observation'}
    for p in ('hold','reactive','geometry','predictive-simple','predictive-learned'):
        np.testing.assert_array_equal(PreventionController(p,c,predictor=Risk()).act(f)[0],PreventionController(p,c,predictor=Risk()).act(bare)[0])
    for h in (3,5,10,20): np.testing.assert_array_equal(edge_features(f,None,(0,1),240,h),edge_features(bare,None,(0,1),240,h))


def test_exact_discrete_deadline_boundary(monkeypatch):
    import src.control.path_service.controller as module
    real=module.enumerate_plans
    def fixed(*args):
        p=real(*args); p['chosen']['arrival_time_s']=1.; return p
    monkeypatch.setattr(module,'enumerate_plans',fixed)
    a,d=PathServiceController('path-service',config(),Risk(5)).act(frame())
    assert np.any(a) and d['required_time_s']==4. and d['action_window_reached']


def test_frozen_protocol_cardinalities_and_hashes():
    from pathlib import Path
    from src.control.path_service.study import validate_protocol
    from src.control.prevention.prediction import strict_json
    root=Path('.superpowers/sdd/2026-11-11-routing-service/task-3-frozen')
    for phase,count in [('pilot',18),('formal',720)]:
        protocol=validate_protocol(strict_json(root/f'{phase}.json'),root/f'{phase}.json',phase)
        assert protocol['expected_episodes']==count


def test_grouped_ablation_planner_does_not_use_observed_route():
    f=frame(); missing=deepcopy(f)
    missing['service_observation']['route']=dict(nodes=[0],success=False,reason='missing_entry',failing_node=0)
    missing['service_observation']['route_probe_hops']=[]
    a=enumerate_plans(f,config(),PathServiceConfig(),'no-service')
    b=enumerate_plans(missing,config(),PathServiceConfig(),'no-service')
    assert a==b


def test_injected_collector_uses_real_loop_and_preserves_failure(tmp_path):
    from src.control.prevention.evaluation import run_grid
    from src.control.path_service.study import controller_factory
    from src.control.path_service.config import POLICIES
    class Predictor(Risk):
        artifact={'comm_range':240}
        def validate_test_scenarios(self,scenarios):pass
    class Env:
        def __init__(self,config,output_root,**kw):
            self.episode_output_dir=output_root/'episode';self.episode_output_dir.mkdir()
            self.frames=[]
        def reset(self,**kw):self.last_frame=frame();self.frames.append(self.last_frame)
        def step(self,action):
            assert np.isfinite(action).all() and len(action)==9
            self.last_frame=frame(1);self.frames.append(self.last_frame)
            return None,0,False,True,{}
        def close(self):
            (self.episode_output_dir/'transitions.jsonl').write_text('\n'.join(json.dumps(dict(frame=f)) for f in self.frames))
            (self.episode_output_dir/'closed_loop_metrics.csv').write_text('pdr,rx_packets,mean_delay_ms,offered_packets,tx_packets\n0,0,nan,0,0\n')
    result=run_grid(config=config(steps=1),launcher=['unused'],output=tmp_path/'ok',policies=['path-service'],scenarios=[(1,101)],predictor=Predictor(),
                    env_factory=Env,controller_factory=controller_factory,supported_policies=POLICIES)
    assert result['paired_valid'] and result['episode_count']==1
    class Broken(Env):
        def reset(self,**kw):raise RuntimeError('preserve me')
    result=run_grid(config=config(steps=1),launcher=['unused'],output=tmp_path/'failed',policies=['path-service'],scenarios=[(1,101)],predictor=Predictor(),
                    env_factory=Broken,controller_factory=controller_factory,supported_policies=POLICIES)
    assert not result['paired_valid'] and 'preserve me' in result['errors'][0]['error']


def test_evaluator_pairs_pdr_motion_outage_and_reports_failure_bounds():
    from scripts.analysis.evaluate_path_service import evaluate,METRICS
    from src.control.path_service.config import POLICIES
    scenarios=[dict(regime='fixed',layout_seed=i,radio_seed=i+100) for i in (1,2)]
    protocol=dict(policies=list(POLICIES),scenarios=scenarios,evaluation=dict(rng_seed=91011,bootstrap_resamples=10000))
    rows=[dict(policy=p,**s,status='ok',**{k:(.7 if p=='path-service' else .6) for k in METRICS}) for s in scenarios for p in POLICIES]
    result=evaluate(rows,protocol)
    assert result['complete_primary_grid']
    assert len(result['scenario_differences'])==16
    m=result['comparisons']['hold']['regimes']['fixed']['actual_udp_pdr']
    assert m['mean_difference']==pytest.approx(.1) and m['complete_pairs']==2
    assert m['scenario_bootstrap_95pct_ci']==pytest.approx([.1,.1])
    rows[0]['status']='failed'
    result=evaluate(rows,protocol)
    assert not result['complete_primary_grid'] and len(result['failures'])==1
    assert result['comparisons']['hold']['regimes']['fixed']['actual_udp_pdr']['complete_pairs']==1
    assert result['failure_sensitivity']['hold']['pooled_equal_regime_mean_difference_bounds']==pytest.approx([-.1,.4])
    with pytest.raises(ValueError): evaluate(rows[:-1],protocol)
