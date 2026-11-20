"""Causal receding-horizon path/subset controller, no future scenario inputs."""
from dataclasses import asdict
import hashlib
import json
import numpy as np
from .config import PathServiceConfig, POLICIES
from .observation import ServiceHistory
from .risk import assess_risk
from .planner import enumerate_plans, one_step_action


class PathServiceController:
    def __init__(self,policy,config,predictor,method=None,model_hash=None,method_hash=None):
        if policy not in POLICIES[5:] or predictor is None: raise ValueError('New policy and frozen predictor required')
        self.policy,self.config,self.predictor=policy,config,predictor
        self.method=method or PathServiceConfig(); self.history=ServiceHistory(self.method)
        self.previous=None; self.moving=False; self.first_action_step=None
        self.method_config_hash=hashlib.sha256(json.dumps(asdict(self.method),sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.method_hash=method_hash or self.method_config_hash
        self.model_hash=model_hash

    def act(self,frame):
        state=self.history.update(frame,self.config)
        ablation=self.policy.removeprefix('path-service-') if self.policy!='path-service' else None
        risk=assess_risk(frame,self.previous,self.config,self.predictor,self.method,ablation=='no-service')
        assessment=enumerate_plans(frame,self.config,self.method,ablation); chosen=assessment['chosen']
        horizon=risk['crossing_horizon_s']
        required=chosen['arrival_time_s']+self.method.routing_delay_s if chosen else None
        window=horizon is not None and required is not None and horizon <= required+1.
        restoration=frame['components']>1 if ablation=='no-service' else state['restoration']
        active=restoration or (horizon is not None and (ablation=='no-deadline' or window))
        reason='restoration' if restoration else 'forecast_prevention' if active else 'risk_outside_action_window' if horizon else 'no_path_risk_crossing'
        proposed,safety=one_step_action(frame,self.config,chosen)
        action=proposed if active else np.zeros_like(proposed)
        moving=bool(np.linalg.norm(action)>1e-8)
        if moving and self.first_action_step is None: self.first_action_step=frame['step']
        route=frame['service_observation']['route']
        d=dict(schema='path-service-decision-v1',step=frame['step'],sim_time_s=frame['service_observation']['sim_time_s'],
               policy=self.policy,method_hash=self.method_hash,method_config_hash=self.method_config_hash,model_hash=self.model_hash,
               observation=frame['service_observation'],service_state=state,risk=risk,planner=assessment,
               required_time_s=required,planning_slack_s=horizon-required if horizon is not None and required is not None else None,
               action_window_reached=window,reason=reason if chosen else assessment['reason'],activation_context=reason,
               observed_path=route['nodes'],actual_path_hops=len(route['nodes'])-1 if route['success'] else None,
               planned_path_hops=len(chosen['chain'])-1 if chosen else None,
               actual_planned_path_mismatch=route['nodes']!=chosen['chain'] if chosen else None,
               feasibility_type='static-target kinematic hypothesis; no delivery guarantee',
               proposed_action=proposed.tolist(),sent_action=action.tolist(),
               observed_executed_motion_since_previous_m=[frame['travel'][i]-self.previous['travel'][i] if self.previous else 0. for i in self.config.relay_ids],
               previous_simulator_safety_rejected=frame.get('safety_rejected',False),
               intervention_started=moving and not self.moving,intervention_stopped=self.moving and not moving,
               first_action_step=self.first_action_step,commanded_motion=moving,**safety)
        self.previous,self.moving=frame,moving
        return action,d
