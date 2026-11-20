"""Reuse prevention collection with explicit factory injection and strict accounting."""
from dataclasses import asdict,replace
from functools import partial
import json
from pathlib import Path
import numpy as np
from .config import POLICIES,PathServiceConfig
from .controller import PathServiceController
from ..prevention.config import PreventionConfig
from ..prevention.controller import PreventionController,target_plan
from ..prevention.environment import Ns3PreventionEnv
from ..prevention.evaluation import run_grid
from ..prevention.prediction import sha256,strict_json


def controller_factory(policy,config,predictor,threshold=.35,routing_delay_s=3.,model_hash=None,method_hash=None):
    if policy.startswith('path-service'):
        return PathServiceController(policy,config,predictor,model_hash=model_hash,method_hash=method_hash)
    return PreventionController(policy,config,predictor=predictor,
                                threshold=.5 if policy=='predictive-simple' else .35,
                                routing_delay_s=1. if policy=='predictive-simple' else 3.)


def reconcile_episode(episode,config):
    """Exact same-boundary counters first; late arrivals are explicitly retained."""
    episode=Path(episode)
    records=[json.loads(l) for l in (episode/'transitions.jsonl').read_text().splitlines()]
    frames=[r['frame'] for r in records]
    from .observation import validate_service_observation
    previous=None
    for frame in frames: validate_service_observation(frame,config,previous); previous=frame
    final=strict_json(episode/'service_final_accounting.json')
    import csv
    raw=list(csv.DictReader((episode/'closed_loop_metrics.csv').open(newline='',encoding='utf-8')))
    if len(raw)!=1: raise ValueError('Expected exactly one terminal FlowMonitor metrics row')
    raw=raw[0]
    for left,right in [('offered_packets','offered_packets'),('source_tx_packets','tx_packets'),('flowmonitor_rx_packets','rx_packets')]:
        if final[left]!=int(raw[right]): raise ValueError('Terminal actual/application/FlowMonitor count mismatch: '+left)
    if final['sink_rx_packets']!=final['flowmonitor_rx_packets']: raise ValueError('Destination application/FlowMonitor mismatch')
    live=frames[-1]['service_observation']
    if final['last_live_frame_time_s']!=live['sim_time_s']: raise ValueError('Final/live boundary mismatch')
    for key,late in [('offered_packets','offered_after_last_frame'),('source_tx_packets','source_tx_after_last_frame'),('sink_rx_packets','late_sink_rx_after_last_frame')]:
        if final[key]-live['cumulative'][key]!=final[late] or final[late]<0: raise ValueError('Late accounting mismatch')
    received,offered=final['sink_rx_packets'],final['offered_packets']
    if received>offered: raise ValueError('Sink receives exceed actual application offers')
    return dict(application_offered_packets=offered,application_received_packets=received,
                actual_udp_pdr=received/offered if offered else None,final_service_accounting=final,
                original_flowmonitor_pdr=float(raw['pdr']),sink_minus_flowmonitor_received=received-int(raw['rx_packets']),
                per_relay_travel_m={str(i):frames[-1]['travel'][i] for i in config.relay_ids},
                traversal_success_fraction=float(np.mean([f['service_observation']['route']['success'] for f in frames[1:]])))


def collect_case(*,scenario,policy,protocol,launcher,predictor,output,provenance):
    config=replace(PreventionConfig(**protocol['config']),num_uavs=scenario['num_uavs'],initial_spacing_fraction=scenario['initial_spacing_fraction'])
    summary=run_grid(config=config,launcher=launcher,output=output,policies=[policy],
                     scenarios=[(scenario['layout_seed'],scenario['radio_seed'])],predictor=predictor,
                     threshold=.35,routing_delay_s=3.,provenance=provenance,
                     env_factory=partial(Ns3PreventionEnv,service_observation=True),
                     controller_factory=partial(controller_factory,model_hash=protocol['model_sha256'],method_hash=protocol['method_sha256']),supported_policies=POLICIES)
    row=strict_json(Path(output)/'episodes.json')[0]
    row['regime']=scenario['regime']
    if row['status']=='ok':
        try:
            row.update(reconcile_episode(row['episode_dir'],config))
            decisions=[json.loads(l) for l in next((Path(output)/'episodes').rglob('decisions.jsonl')).read_text().splitlines()]
            if policy.startswith('path-service'):
                chosen=[d for d in decisions if d['planner']['chosen']]
                row.update(no_feasible_plan_fraction=float(np.mean([d['planner']['chosen'] is None for d in decisions])),
                           actual_planned_path_mismatch_fraction=float(np.mean([d['actual_planned_path_mismatch'] for d in chosen])) if chosen else None,
                           restoration_starts=sum(d['intervention_started'] and d['activation_context']=='restoration' for d in decisions),
                           forecast_starts=sum(d['intervention_started'] and d['activation_context']=='forecast_prevention' for d in decisions),
                           one_step_safety_rejections=sum(d['one_step_safety_rejected'] for d in decisions),
                           budget_truncation_decisions=sum(bool(d['budget_truncated_relays'] or d['simulator_budget_scaled_relays']) for d in decisions))
        except Exception as exc: row.update(status='technical_error',error=f'Accounting/decision validation: {type(exc).__name__}: {exc}')
    return row


def validate_protocol(protocol,path,phase):
    if protocol.get('schema')!='path-service-study-v1' or protocol.get('phase')!=phase: raise ValueError('Protocol phase/schema mismatch')
    if protocol['policies']!=list(POLICIES) or protocol['expected_episodes']!=len(protocol['scenarios'])*9: raise ValueError('Protocol cardinality/policy mismatch')
    if not protocol['service_observation'] or not protocol['full_horizon'] or protocol['config']!=asdict(PreventionConfig()): raise ValueError('Frozen observer/horizon/default contract changed')
    method=Path(path).parent/'method.json'
    if sha256(method)!=protocol['method_sha256'] or strict_json(method)['config']!=asdict(PathServiceConfig()): raise ValueError('Frozen numerical method mismatch')
    freeze=strict_json(Path(path).parent/'freeze.json')
    for name,digest in freeze['files'].items():
        if sha256(Path(path).parent/name)!=digest: raise ValueError('Frozen protocol file changed: '+name)
    from scripts.analysis.freeze_path_service_protocol import MODEL_HASH
    if protocol['model_sha256']!=MODEL_HASH: raise ValueError('Wrong frozen model identity')
    return protocol
