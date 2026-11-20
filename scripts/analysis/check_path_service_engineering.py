"""Sequential matched observer check and one new-controller engineering smoke.

Historical scenario replay is engineering evidence, not fresh validation. This
command does not launch pilot/formal study cells and never tunes method values.
"""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys
import time
import numpy as np

if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.control.bridge import make_launcher
from src.control.prevention.config import PreventionConfig
from src.control.prevention.controller import PreventionController
from src.control.prevention.environment import Ns3PreventionEnv
from src.control.prevention.features import edge_features,measured_edges
from src.control.prevention.prediction import LearnedPredictor,sha256
from src.control.path_service.study import reconcile_episode,controller_factory
from src.control.evaluation import verify_build_manifest,runtime_versions


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);p.add_argument('--wsl');p.add_argument('--model',type=Path,default=Path('reports/uav_prevention_pilot_2026-11-11/predictor.json'));args=p.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=False)
    model=LearnedPredictor.load(args.model);launcher=make_launcher(args.binary.resolve(),args.wsl)
    c=PreventionConfig(); runs=[]
    for observed in (False,True):
        env=Ns3PreventionEnv(c,launcher,args.output/('observer_on' if observed else 'observer_off'),service_observation=observed)
        frames=[]
        try:
            env.reset(options=dict(layout_seed=31001,radio_seed=131001));frames.append(env.last_frame)
            ctrl=PreventionController('hold',c)
            for _ in range(c.steps): env.step(ctrl.act(env.last_frame)[0]);frames.append(env.last_frame)
        finally: env.close()
        runs.append((env.episode_output_dir,frames))
    stripped=[{k:v for k,v in f.items() if k!='service_observation'} for f in runs[1][1]]
    assert runs[0][1]==stripped,'Original frame fields changed with observer'
    scientific=('nodes.csv','edges.csv','traffic_log.csv','recovery_log.csv','recovery_actions.csv','closed_loop_metrics.csv')
    equality={name:(runs[0][0]/name).read_bytes()==(runs[1][0]/name).read_bytes() for name in scientific}
    assert all(equality.values()),equality
    features=0
    for index,(bare,extended) in enumerate(zip(runs[0][1],runs[1][1])):
        for edge in measured_edges(bare):
            for h in (3,5,10,20):
                left=edge_features(bare,runs[0][1][index-1] if index else None,edge,c.comm_range,h)
                right=edge_features(extended,runs[1][1][index-1] if index else None,edge,c.comm_range,h)
                assert left.tobytes()==right.tobytes();features+=1
    baseline_actions={}
    for policy in ('hold','reactive','geometry','predictive-simple','predictive-learned'):
        a=controller_factory(policy,c,model);b=controller_factory(policy,c,model)
        for bare,extended in zip(runs[0][1][:-1],runs[1][1][:-1]):
            left,ld=a.act(bare);right,rd=b.act(extended)
            assert left.tobytes()==right.tobytes() and ld==rd
        baseline_actions[policy]='exact action bytes and diagnostics equal on all matched frames'
    accounting=reconcile_episode(runs[1][0],c)
    # Compact historical replay tests action feasibility, not performance selection.
    compact=replace(c,steps=30,initial_spacing_fraction=.30)
    env=Ns3PreventionEnv(compact,launcher,args.output/'new_policy_smoke',service_observation=True)
    decisions=[];times=[]
    try:
        env.reset(options=dict(layout_seed=31001,radio_seed=131001));ctrl=controller_factory('path-service',compact,model,model_hash=sha256(args.model))
        for _ in range(compact.steps):
            start=time.perf_counter();action,d=ctrl.act(env.last_frame);times.append(time.perf_counter()-start)
            decisions.append(d);env.step(action)
    finally:env.close()
    smoke_accounting=reconcile_episode(env.episode_output_dir,compact)
    tight=PreventionConfig(num_uavs=3,destination=2,steps=20,initial_spacing_fraction=.15)
    # Additional unrelated UDP flow uses another port; service callbacks and
    # matching FlowMonitor snapshots must retain only the configured flow.
    env=Ns3PreventionEnv(tight,launcher+['--backgroundFlows=1','--backgroundRateKbps=8'],args.output/'positive_delivery_filter',service_observation=True)
    try:
        env.reset(options=dict(layout_seed=31001,radio_seed=131001))
        for _ in range(tight.steps): env.step(np.zeros(3*len(tight.relay_ids)))
    finally:env.close()
    positive=reconcile_episode(env.episode_output_dir,tight)
    assert positive['application_received_packets']>0,'Positive-delivery fixture must exercise actual sink callbacks'
    with (args.output/'new_policy_decisions.jsonl').open('x',encoding='utf-8') as stream:
        for d in decisions: stream.write(json.dumps(d,allow_nan=False)+'\n')
    result=dict(schema='path-service-engineering-evidence-v1',scope='historical replay engineering only; no fresh validation or tuning',
                original_frames_equal=True,scientific_byte_equal=equality,matched_scientific_hashes={name:sha256(runs[0][0]/name) for name in scientific},
                feature_arrays_compared=features,feature_bytes_equal=True,baseline_actions=baseline_actions,
                accounting=accounting,new_policy_smoke=smoke_accounting,new_policy_commanded_motion_decisions=sum(d['commanded_motion'] for d in decisions),
                positive_delivery_with_unrelated_flow=positive,
                decision_seconds_mean=float(np.mean(times)),decision_seconds_max=max(times),
                model_sha256=sha256(args.model),build_manifest=verify_build_manifest(args.binary.resolve()),versions=runtime_versions())
    with (args.output/'evidence.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('original_frames_equal','scientific_byte_equal','feature_arrays_compared','new_policy_commanded_motion_decisions','decision_seconds_max')}))

if __name__=='__main__':main()
