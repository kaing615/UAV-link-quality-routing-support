"""Bounded subset enumeration, cheap ranking then lazy exact swept screening."""
from collections import Counter
from itertools import combinations
import numpy as np


def closest(a, va, b, vb, duration):
    r, v = a-b, va-vb
    vv = float(v@v)
    t = min(duration,max(0.,-float(r@v)/vv)) if vv else 0.
    return float(np.linalg.norm(r+t*v))


def separated(positions, targets, speed, minimum, nodes):
    """Exact piecewise synchronous travel; each node stops at its own arrival."""
    delta = targets-positions
    distances = np.linalg.norm(delta,axis=1)
    arrivals = distances/speed
    velocities = delta/np.maximum(arrivals[:,None],1e-15)
    for a,b in combinations(nodes,2):
        times=sorted(set((0.,float(arrivals[a]),float(arrivals[b]))))
        for start,end in zip(times,times[1:]):
            pa=positions[a]+velocities[a]*min(start,arrivals[a]); pb=positions[b]+velocities[b]*min(start,arrivals[b])
            va=velocities[a] if start < arrivals[a] else np.zeros(3)
            vb=velocities[b] if start < arrivals[b] else np.zeros(3)
            if closest(pa,va,pb,vb,end-start) < minimum-1e-8: return False
        if np.linalg.norm(targets[a]-targets[b]) < minimum-1e-8: return False
    return True


def enumerate_plans(frame, config, method, ablation=None):
    positions=np.asarray(frame['positions'],float); velocities=np.asarray(frame['velocities'],float)
    relays=config.relay_ids; source,dest=config.source,config.destination
    remaining=max(0.,config.steps-frame['step']); budget=np.maximum(0.,config.budget-np.asarray(frame['travel'],float))
    axis=positions[dest]-positions[source]
    ordered=tuple(sorted(relays,key=lambda i:(float((positions[i]-positions[source])@axis),i)))
    if ablation == 'no-service': chains=[(source,*ordered,dest)]
    else:
        chains=[(source,*subset,dest) for size in range(1,len(relays)+1) for subset in combinations(ordered,size)]
        route=frame['service_observation']['route']
        if route['success'] and len(set(route['nodes'])) == len(route['nodes']) and len(route['nodes']) > 2:
            actual=tuple(route['nodes'])
            if actual not in chains: chains.append(actual)
    rejected=Counter(); candidates=[]
    actual=(dict(success=frame['components']==1,nodes=[source,*ordered,dest])
            if ablation=='no-service' else frame['service_observation']['route'])
    for chain in chains:
        selected=chain[1:-1]; links=len(chain)-1
        fractions=np.arange(1,links)/links
        t0=max((float(np.linalg.norm(positions[source]+f*axis-positions[i]))/config.max_speed for i,f in zip(selected,fractions)),default=0.)
        lookahead=min(method.lookahead_cap_s,t0+method.routing_delay_s,remaining)
        raw=np.asarray([positions[i]+lookahead*velocities[i] for i in (source,dest)])
        endpoints=np.clip(raw,[0,0,config.z_min],[config.x_max,config.y_max,config.z_max])
        targets=positions.copy()
        for i,f in zip(selected,fractions): targets[i]=endpoints[0]+f*(endpoints[1]-endpoints[0])
        distances=np.linalg.norm(targets-positions,axis=1)
        travel=max((float(distances[i])/config.max_speed for i in selected),default=0.)
        if travel > remaining+1e-9: rejected['episode_time']+=1; continue
        if ablation != 'no-budget' and any(distances[i] > budget[i]+1e-6 for i in selected): rejected['complete_budget']+=1; continue
        goals=targets[list(chain)].copy(); goals[0],goals[-1]=endpoints
        g=float(np.linalg.norm(np.diff(goals,axis=0),axis=1).max()/config.comm_range)
        gap=round(max(0.,g-method.rho),6)
        current=positions[list(chain)].copy(); current[0],current[-1]=endpoints
        current_gap=round(max(0.,float(np.linalg.norm(np.diff(current,axis=0),axis=1).max()/config.comm_range)-method.rho),6)
        movement=max((float(distances[i]) for i in selected),default=0.)
        reason='geometry_improvement'
        improves=current_gap-gap >= method.gap_improvement
        improves |= gap==0 and actual['success'] and links < len(actual['nodes'])-1
        if not actual['success'] and movement >= method.min_motion_m: improves=True; reason='route_restoration_geometry_hypothesis'
        if movement < method.min_motion_m: rejected['no_actionable_geometric_change']+=1; continue
        if not improves: rejected['no_geometric_improvement']+=1; continue
        denom={i:config.budget if ablation=='no-budget' else budget[i] for i in selected}
        utilization=max((float(distances[i])/max(denom[i],1e-6) for i in selected),default=0.)
        rank=(gap,links,utilization,float(sum(distances[i] for i in selected)),chain)
        candidates.append((rank,dict(chain=list(chain),targets=targets.tolist(),distances_m=distances.tolist(),
                         remaining_budget_m=budget.tolist(),G=g,score=list(rank[:-1])+[list(chain)],arrival_time_s=travel,
                         lookahead_s=lookahead,endpoint_clipped=bool(np.any(raw!=endpoints)),
                         endpoint_estimates=endpoints.tolist(),reason=reason),targets))
    candidates.sort(key=lambda x:x[0]); chosen=None; checked=0
    for _, candidate,targets in candidates:
        checked+=1
        if not separated(positions,targets,config.max_speed,config.min_separation,relays): rejected['full_relay_swept_separation']+=1; continue
        # Anchors extrapolation is a screening hypothesis, not guaranteed motion.
        heuristic_targets=targets.copy(); heuristic_targets[[source,dest]]=candidate['endpoint_estimates']
        if not separated(positions,heuristic_targets,config.max_speed,config.min_separation,range(config.num_uavs)):
            rejected['anchor_extrapolation_separation']+=1; continue
        chosen=candidate; break
    return dict(candidate_count=len(chains),hold_candidate=True,cheap_eligible_count=len(candidates),
                collision_assessed_count=checked,collision_unevaluated_count=len(candidates)-checked,
                rejection_counts=dict(rejected),chosen=chosen,
                reason=chosen['reason'] if chosen else ('no_affordable_complete_plan' if rejected['complete_budget'] else 'no_actionable_geometric_change'))


def one_step_action(frame,config,plan):
    action=np.zeros((config.num_uavs,3)); positions=np.asarray(frame['positions'],float)
    clipped=[]
    if plan:
        for i in plan['chain'][1:-1]:
            delta=np.asarray(plan['targets'][i])-positions[i]; distance=float(np.linalg.norm(delta))
            budget=max(0.,config.budget-frame['travel'][i])
            movement=min(config.max_speed,distance,budget)
            if min(config.max_speed,distance) > budget+1e-9: clipped.append(i)
            if distance: action[i]=delta/distance*movement/config.max_speed
    # Mirror one-step simulator screening, including stationary anchors (the
    # simulator separately evaluates actual anchor motion and is authoritative).
    budgets=np.maximum(0.,config.budget-np.asarray(frame['travel'],float))
    scales=np.minimum(config.max_speed,budgets)/np.maximum(1.,np.linalg.norm(action,axis=1))
    next_positions=np.clip(positions+action*scales[:,None],[0,0,config.z_min],[config.x_max,config.y_max,config.z_max])
    rejected=any(closest(positions[a],next_positions[a]-positions[a],positions[b],next_positions[b]-positions[b],1.) < config.min_separation-1e-8
                 for a,b in combinations(range(config.num_uavs),2))
    if rejected: action[:]=0
    simulator_scaled=[i for i in config.relay_ids if budgets[i]<config.max_speed and np.linalg.norm(action[i])>0]
    return action[list(config.relay_ids)].ravel(),dict(one_step_safety_rejected=rejected,budget_truncated_relays=clipped,
          simulator_budget_scaled_relays=simulator_scaled,
          expected_static_anchor_one_step_motion_m={str(i):float(np.linalg.norm(next_positions[i]-positions[i])) if not rejected else 0. for i in config.relay_ids})
