"""Scenario-paired endpoints and healthy HOLD forecasting, without fitting.

The unit of resampling is a complete layout/radio scenario block. Null values
remain null. Treated warning absence is never interpreted as a false warning.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import math

import numpy as np

from .config import POLICIES, ServiceV2Config
from .protocol import strict_json, strict_loads, write_json, reserve, sha256, load_phase, safe_path
from .observation import ServiceHistory
from .prediction import _episode_states, _decline_label

BOOTSTRAP_SAMPLES=10000
BOOTSTRAP_SEED=20261112


def _mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _ci(values):
    return [float(v) for v in np.quantile(values,[.025,.975])]


def _observed(row, metric):
    return row is not None and row.get('status')=='ok' and row.get(metric) is not None


def _scenario_bootstrap(rows):
    """One draw per context/regime, retaining all eight role rows per block.

    An unavailable role excludes its whole scenario from complete-block point
    estimates and intervals. Every planned pair stays in the metadata and in
    the comparator-specific PDR sensitivity calculation below.
    """
    lookup={}; groups={}
    for r in rows:
        if r['policy'] not in POLICIES: raise ValueError('Unknown paired role')
        key=(r['context'],r['regime'],r['layout_seed'],r['radio_seed'])
        if (key,r['policy']) in lookup: raise ValueError('Duplicate paired cell')
        lookup[key,r['policy']]=r
        groups.setdefault((r['context'],r['regime']),set()).add(key)
    rng=np.random.default_rng(BOOTSTRAP_SEED); strata={}
    for context in ('static','gradual'):
        for regime in ('n4','n5'):
            keys=sorted(groups.get((context,regime),()))
            complete=[]; incomplete=[]
            for key in keys:
                unavailable=[role for role in POLICIES if not all(_observed(lookup.get((key,role)),metric)
                    for metric in ('actual_udp_pdr','total_relay_travel_m'))]
                if unavailable:
                    incomplete.append(dict(layout_seed=key[2],radio_seed=key[3],unavailable_roles=unavailable))
                else: complete.append(key)
            indices=rng.integers(0,len(complete),size=(BOOTSTRAP_SAMPLES,len(complete))) if complete else None
            metadata=dict(roles_kept_together=list(POLICIES),
                complete_scenario_pairs=[dict(layout_seed=key[2],radio_seed=key[3]) for key in complete],
                incomplete_blocks=incomplete,draw_shape=[BOOTSTRAP_SAMPLES,len(complete)],
                draw_indices_sha256=hashlib.sha256(indices.astype('<i8',copy=False).tobytes()).hexdigest() if indices is not None else None)
            strata[context,regime]=dict(keys=keys,complete=complete,indices=indices,metadata=metadata)
    return lookup,strata


def paired_effects(rows, comparator):
    return _paired_effects(_scenario_bootstrap(rows),comparator)


def _paired_effects(blocks, comparator):
    if comparator not in POLICIES or comparator=='service-v2': raise ValueError('Comparator role')
    lookup,strata=blocks
    if any(not strata['gradual',r]['keys'] for r in ('n4','n5')): raise ValueError('Both fixed regimes required')
    regimes={}; draws={}; bounds=[]
    for regime in ('n4','n5'):
        group=strata['gradual',regime]; keys=group['keys']; lower=[]; upper=[]; points=[]; complete=[]
        for key in keys:
            a=lookup.get((key,'service-v2')); b=lookup.get((key,comparator))
            av=a['actual_udp_pdr'] if _observed(a,'actual_udp_pdr') else None
            bv=b['actual_udp_pdr'] if _observed(b,'actual_udp_pdr') else None
            lower.append((av if av is not None else 0)-(bv if bv is not None else 1))
            upper.append((av if av is not None else 1)-(bv if bv is not None else 0))
            if key in group['complete']:
                delta=(av-bv,a['total_relay_travel_m']-b['total_relay_travel_m'])
                complete.append(delta)
                points.append(dict(layout_seed=key[2],radio_seed=key[3],pdr_difference=delta[0],motion_difference_m=delta[1]))
        if complete:
            values=np.asarray(complete)
            sample=values[group['indices']].mean(axis=1); draws[regime]=sample
            pdr,motion=values.mean(axis=0)
            interval=_ci(sample[:,0]); motion_ci=_ci(sample[:,1])
        else: pdr=motion=interval=motion_ci=None
        sensitivity=[float(np.mean(lower)),float(np.mean(upper))]; bounds.append(sensitivity)
        regimes[regime]=dict(planned_pairs=len(keys),complete_pairs=len(complete),pdr_difference=None if pdr is None else float(pdr),
            pdr_ci95=interval,motion_difference_m=None if motion is None else float(motion),motion_ci95_m=motion_ci,
            missing_pdr_sensitivity=sensitivity,paired_points=points,block_bootstrap=group['metadata'],
            practical_delta_at_least_point01=None if pdr is None else bool(pdr>=.01))
    if len(draws)==2:
        pooled=(draws['n4']+draws['n5'])/2
        point=lambda name:float(np.mean([regimes[r][name] for r in ('n4','n5')]))
        aggregate=dict(pdr_difference=point('pdr_difference'),pdr_ci95=_ci(pooled[:,0]),
            motion_difference_m=point('motion_difference_m'),motion_ci95_m=_ci(pooled[:,1]))
    else: aggregate=dict(pdr_difference=None,pdr_ci95=None,motion_difference_m=None,motion_ci95_m=None)
    aggregate['missing_pdr_sensitivity']=np.mean(bounds,axis=0).tolist()
    return dict(comparator=comparator,regimes=regimes,pooled=aggregate,bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,unit='scenario seed pair, all policy outcomes retained together',
        pooling='equal-weight regime means',interval_interpretation='principal v2/no-forecast; other contrasts descriptive')


def episode_endpoints(episode, decisions=()):
    _,states=_episode_states(episode); sequence=list(states.values()); final=sequence[-1]
    events={}
    for event in ('degraded','severe','outage','route_failure'):
        onsets=list(final.event_onsets[event]); confirmations=list(final.event_confirmations[event])
        onset=onsets[0] if onsets else None; confirmation=confirmations[0] if confirmations else None
        restorations=[]
        for onset_i,confirm in zip(onsets,confirmations):
            recovered=next((s.step for s in sequence if s.step>confirm and s.recovered),None)
            restorations.append(dict(onset_step=onset_i,confirmation_step=confirm,recovered_step=recovered,
                duration_seconds=(recovered if recovered is not None else episode.config.steps)-confirm,
                right_censored=recovered is None))
        events[event]=dict(onset_step=onset,confirmation_step=confirmation,event_count=len(onsets),
            incidence_numerator=int(bool(onsets)),incidence_denominator=1,right_censored=onset is None,
            restricted_event_free_seconds=episode.config.steps if onset is None else max(0,onset-1),
            restorations=restorations,recovered_by_end_numerator=sum(not r['right_censored'] for r in restorations),
            confirmed_episode_denominator=len(restorations))
    ratios=[s.ratios['A5'] for s in sequence[1:]]
    actual_seconds=sum(any(np.linalg.norm(np.array(b['positions'][i])-np.array(a['positions'][i]))>1e-6 for i in episode.config.relay_ids)
        for a,b in zip(episode.frames,episode.frames[1:]))
    armed=final.first_armed_step
    moving_before=sum(any(np.linalg.norm(np.array(b['positions'][i])-np.array(a['positions'][i]))>1e-6 for i in episode.config.relay_ids)
        and (armed is None or a['step']<armed) for a,b in zip(episode.frames,episode.frames[1:]))
    starts=[d for d in decisions if d.get('start',d.get('intervention_started',False))]
    context=lambda d:d.get('context',d.get('activation_context'))
    forecast_context=lambda d:context(d) in ('forecast','forecast_prevention')
    reasons={}
    for d in decisions:
        reason=d.get('stop_reason',d.get('reason'))
        if reason is not None: reasons[reason]=reasons.get(reason,0)+1
    startup=any(s.activation_context=='startup_restoration' for s in sequence)
    leads=[]
    for start in starts:
        if not forecast_context(start): continue
        t=start['step']
        match=next(((onset,confirm) for onset,confirm in zip(final.event_onsets['degraded'],final.event_confirmations['degraded']) if confirm>=t),None)
        next_start=next((d['step'] for d in starts if d['step']>t),episode.config.steps)
        executed=next((a['step'] for a,b in zip(episode.frames,episode.frames[1:]) if t<=a['step']<next_start and
            any(np.linalg.norm(np.array(b['positions'][i])-np.array(a['positions'][i]))>1e-6 for i in episode.config.relay_ids)),None)
        onset=match[0] if match else None
        leads.append(dict(warning_step=t,matched_ordinary_onset_step=onset,first_executed_motion_step=executed,
            warning_to_onset_seconds=onset-t if onset is not None else None,
            executed_motion_to_onset_seconds=onset-executed if onset is not None and executed is not None else None,
            treated_absence_is_not_false_warning=True,event_right_censored=onset is None))
    return dict(events=events,first_armed_step=armed,armed_by_step20_numerator=int(armed is not None and armed<=20),
        healthy_start_denominator=1,never_armed_numerator=int(armed is None),startup_restoration_numerator=int(startup),
        time_in_confirmed_degradation_seconds=sum(s.degraded_confirmed or (s.mode=='RESTORE' and s.activation_context=='delivery_restoration') for s in sequence[1:]),
        below_point90_window_numerator=sum(v is not None and v<.90 for v in ratios),
        evaluable_A5_window_denominator=sum(v is not None for v in ratios),unknown_A5_windows=sum(v is None for v in ratios),
        forecast_starts=sum(forecast_context(d) for d in starts),
        restoration_starts=sum(context(d) in ('delivery_restoration','routing_restoration','startup_restoration','restoration') for d in starts),
        commanded_moving_seconds=sum(bool(np.any(d['sent_action'])) if 'sent_action' in d else bool(d.get('commanded_motion')) for d in decisions),actual_moving_seconds=actual_seconds,
        actual_moving_seconds_before_arming=moving_before,
        forecast_starts_before_arming=sum(forecast_context(d) and (armed is None or d['step']<armed) for d in starts),
        simulator_safety_rejections=sum(bool(f['safety_rejected']) for f in episode.frames[1:]),
        budget_rejections=sum(bool(d.get('budget_truncated_relays') or d.get('simulator_budget_scaled_relays') or d.get('reason')=='no_movable_budget') for d in decisions),
        burst_durations_seconds=[sum(1 for d in decisions if start['step']<=d['step']<next((other['step'] for other in starts if other['step']>start['step']),episode.config.steps) and d.get('motion_mode')=='BURST') for start in starts],
        cooldown_seconds=sum(d.get('motion_mode')=='COOLDOWN' for d in decisions),stop_reasons=reasons,
        intervention_leads=leads)


def secondary_effects(rows):
    """Descriptive contrasts, complete scenario blocks with every planned role."""
    if not any(r['policy']=='service-v2' for r in rows): return {}
    return _secondary_effects(_scenario_bootstrap(rows))


def _secondary_effects(blocks):
    lookup,strata=blocks; roles=list(POLICIES); output={}
    for context in ('static','gradual'):
        output[context]={}
        for regime in ('n4','n5'):
            group=strata[context,regime]; complete=group['complete']
            names=['actual_udp_pdr','total_relay_travel_m','time_in_confirmed_degradation_seconds','forecast_starts','restoration_starts',
                'commanded_moving_seconds','actual_moving_seconds','actual_moving_seconds_before_arming','cooldown_seconds',
                'armed_by_step20_numerator','never_armed_numerator','startup_restoration_numerator']
            names += [event+'_'+kind for event in ('degraded','severe','outage','route_failure') for kind in ('restricted_event_free_seconds','incidence_numerator')]
            def metric(row,name):
                if name in row: return row[name]
                if name in row['endpoints']: return row['endpoints'][name]
                for event in ('degraded','severe','outage','route_failure'):
                    if name.startswith(event+'_'): return row['endpoints']['events'][event][name[len(event)+1:]]
                raise ValueError('Unknown endpoint')
            contrasts={}
            ix=group['indices']
            for role in roles:
                if role=='service-v2': continue
                metrics={}
                for name in names:
                    delta=np.array([metric(lookup[(key,'service-v2')],name)-metric(lookup[(key,role)],name) for key in complete])
                    metrics[name]=dict(mean_paired_difference=float(delta.mean()) if len(delta) else None,
                        descriptive_ci95=_ci(delta[ix].mean(axis=1)) if len(delta) else None,
                        individual_differences=delta.tolist())
                contrasts[role]=metrics
            output[context][regime]=dict(planned_pairs=len(group['keys']),all_role_complete_pairs=len(complete),
                roles_kept_together=roles,block_bootstrap=group['metadata'],contrasts=contrasts,
                interpretation='descriptive secondary intervals; binary endpoints are paired risk differences with raw counts in rows')
    return output


def _effect_tables(rows):
    """Production path: generate all primary/secondary contrasts together."""
    if not any(r['policy']=='service-v2' for r in rows): return {},{}
    blocks=_scenario_bootstrap(rows)
    return ({role:_paired_effects(blocks,role) for role in POLICIES if role!='service-v2'},
            _secondary_effects(blocks))


def binary_scores(labels, probabilities):
    y=np.asarray(labels,float); p=np.asarray(probabilities,float)
    if y.shape!=p.shape or y.ndim!=1 or not np.isfinite(p).all() or not np.isin(y,[0,1]).all() or (p<0).any() or (p>1).any(): raise ValueError('Invalid score rows')
    if not len(y):
        return dict(rows=0,prevalence=None,brier=None,log_loss=None,precision=None,recall=None,pr_auc=None,reliability=[])
    threshold=p>=.5; tp=int(np.sum(threshold*(y==1))); fp=int(np.sum(threshold*(y==0)))
    positives=int(y.sum()); clipped=np.clip(p,1e-15,1-1e-15)
    reliability=[]
    for index in range(5):
        mask=(p>=index*.2)&((p<(index+1)*.2) if index<4 else (p<=1))
        reliability.append(dict(lower=index*.2,upper=(index+1)*.2,count=int(mask.sum()),
            mean_probability=float(p[mask].mean()) if mask.any() else None,event_fraction=float(y[mask].mean()) if mask.any() else None))
    auc=None
    if 0<positives<len(y):
        # Trapezoidal precision-recall area, tied scores enter together.
        precision=[1.]; recall=[0.]
        for threshold_value in sorted(set(p),reverse=True):
            mask=p>=threshold_value; true=float(y[mask].sum())
            precision.append(true/int(mask.sum())); recall.append(true/positives)
        auc=float(np.trapezoid(precision,recall))
    return dict(rows=len(y),positives=positives,prevalence=float(y.mean()),brier=float(np.mean((p-y)**2)),
        log_loss=float(-np.mean(y*np.log(clipped)+(1-y)*np.log(1-clipped))),
        true_positive=tp,false_positive=fp,predicted_positive=tp+fp,precision=tp/(tp+fp) if tp+fp else None,
        recall=tp/positives if positives else None,pr_auc=auc,pr_auc_definition='trapezoid with tied scores grouped',reliability=reliability)


def forecast_scores(episodes, predictor):
    from .controller import ServiceV2Controller
    from .features import service_features
    from .prediction import _predict_details
    selected=[e for e in episodes if e.policy=='hold']
    if not selected: raise ValueError('Heldout HOLD trajectories required')
    predictor.validate_scenarios([dict(layout_seed=e.layout_seed,radio_seed=e.radio_seed,config=asdict(e.config)) for e in selected])
    heads={h:dict(labels=[],probabilities=[],censored=0,eligible=0,episodes=[],origins=[]) for h in (10,20)}
    warnings=[]; events_in_risk=set(); detected=set(); all_events=set(); eligible_seconds=0; shadow=[]
    for e in selected:
        frames,states=_episode_states(e); series=[]; active=False; earliest={}; episode_head={h:([],[]) for h in heads}
        all_events.update((e.identity,onset) for onset in states[max(states)].event_onsets['degraded'])
        controller=ServiceV2Controller('service-v2',e.config,predictor,ServiceV2Config(),'a'*64,'b'*64)
        for f in e.frames:
            t=f['step']; state=states[t]
            # The shadow controller is read-only: proposed actions are discarded.
            _,decision=controller.act(f)
            if decision.get('start') and decision.get('context')=='forecast':
                shadow.append(dict(episode=e.identity,step=t,kind='shadow_forecast_opportunity',decision=decision))
            if not state.forecast_eligible: active=False; continue
            eligible_seconds+=1
            details=_predict_details(service_features([frames[k] for k in frames if k<=t],e.config),predictor.artifact)
            risks={h:details[str(h)]['projected_risk'] for h in heads}; labels={}
            for h,head in heads.items():
                label=_decline_label(e,frames,states,t,h); labels[h]=label; head['eligible']+=1
                if label.value is None: head['censored']+=1
                else:
                    head['labels'].append(label.value); head['probabilities'].append(risks[h])
                    episode_head[h][0].append(label.value); episode_head[h][1].append(risks[h])
                    head['origins'].append(dict(episode=e.identity,step=t,label=label.value,**details[str(h)]))
                    if label.value==1 and label.ordinary_onset_step>t: events_in_risk.add((e.identity,label.ordinary_onset_step))
            crossings=[h for h in heads if risks[h]>=.50]
            if crossings and not active:
                h=min(crossings); label=labels[h]; onset=label.ordinary_onset_step
                key=(e.identity,onset); redundant=label.value==1 and key in earliest
                lead=onset-t if label.value==1 else None
                warning=dict(episode=e.identity,step=t,horizon=h,label=label.value,censor_reason=label.reason,
                    ordinary_onset_step=onset,lead_seconds=lead,advance=lead is not None and lead>0,redundant_positive=redundant)
                warnings.append(warning)
                if label.value==1:
                    earliest.setdefault(key,warning)
                    if lead>0: detected.add(key)
            active=bool(crossings)
        for h in heads: heads[h]['episodes'].append(episode_head[h])
    output={}
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    for h,head in heads.items():
        score=binary_scores(head['labels'],head['probabilities']); clusters=head['episodes']
        # Episode cluster uncertainty; never resample individual windows.
        totals=np.array([[len(y),sum((p_i-y_i)**2 for y_i,p_i in zip(y,p)),
            sum(-y_i*math.log(max(1e-15,p_i))-(1-y_i)*math.log(max(1e-15,1-p_i)) for y_i,p_i in zip(y,p))] for y,p in clusters])
        ix=rng.integers(0,len(clusters),size=(BOOTSTRAP_SAMPLES,len(clusters))); sums=totals[ix].sum(axis=1); valid=sums[:,0]>0
        score.update(eligible_rows=head['eligible'],censored_rows=head['censored'],
            independent_seed_pairs=len({(e.layout_seed,e.radio_seed) for e in selected}),
            brier_episode_cluster_ci95=_ci(sums[valid,1]/sums[valid,0]) if valid.any() else None,
            log_loss_episode_cluster_ci95=_ci(sums[valid,2]/sums[valid,0]) if valid.any() else None,
            origins=head['origins'])
        output[str(h)]=score
    false=sum(w['label']==0 for w in warnings); evaluated=sum(w['label'] is not None for w in warnings)
    return dict(schema='service-v2-forecast-scores-v1',population='heldout eligible healthy armed HOLD only',horizons=output,
        warnings=warnings,false_warning_count=false,evaluable_warning_count=evaluated,
        censored_warning_count=sum(w['label'] is None for w in warnings),eligible_hold_seconds=eligible_seconds,
        false_warnings_per100_eligible_minutes=false/(eligible_seconds/60)*100 if eligible_seconds else None,
        false_warning_fraction=false/evaluated if evaluated else None,
        event_sensitivity=len(detected&events_in_risk)/len(events_in_risk) if events_in_risk else None,
        detected_event_numerator=len(detected&events_in_risk),eligible_event_denominator=len(events_in_risk),
        all_ordinary_events=len(all_events),ordinary_events_outside_risk_set=len(all_events-events_in_risk),
        redundant_positive_warnings=sum(w['redundant_positive'] for w in warnings),shadow_opportunities=shadow,
        bootstrap_seed=BOOTSTRAP_SEED,bootstrap_samples=BOOTSTRAP_SAMPLES,
        warning_lead_seconds=[w['lead_seconds'] for w in warnings if w['label']==1 and not w['redundant_positive']],
        first_shadow_opportunity_lead_seconds=[dict(episode=w['episode'],lead_seconds=w['ordinary_onset_step']-next((s['step'] for s in shadow if s['episode']==w['episode'] and w['step']<=s['step']<w['ordinary_onset_step']),w['ordinary_onset_step']))
            for w in warnings if w['label']==1 and w['advance'] and not w['redundant_positive']])


def _descriptive(rows):
    output={}
    for context in ('static','gradual'):
        output[context]={}
        for regime in ('n4','n5'):
            output[context][regime]={}
            for role in sorted({r['policy'] for r in rows}):
                planned=[r for r in rows if (r['context'],r['regime'],r['policy'])==(context,regime,role)]
                ok=[r for r in planned if r['status']=='ok']
                endpoint=[r['endpoints'] for r in ok]
                incidence={event:dict(numerator=sum(e['events'][event]['incidence_numerator'] for e in endpoint),
                    observed_denominator=len(ok),planned_denominator=len(planned),
                    restricted_event_free_mean_seconds=_mean([e['events'][event]['restricted_event_free_seconds'] for e in endpoint]))
                    for event in ('degraded','severe','outage','route_failure')}
                output[context][regime][role]=dict(planned=len(planned),observed=len(ok),
                    mean_application_pdr=_mean([r['actual_udp_pdr'] for r in ok]),
                    mean_total_relay_meters=_mean([r['total_relay_travel_m'] for r in ok]),
                    received=sum(r['application_received_packets'] for r in ok),offered=sum(r['application_offered_packets'] for r in ok),
                    armed_by_step20=dict(numerator=sum(e['armed_by_step20_numerator'] for e in endpoint),denominator=len(ok),planned_denominator=len(planned)),
                    never_armed=dict(numerator=sum(e['never_armed_numerator'] for e in endpoint),denominator=len(ok)),
                    startup_restoration=dict(numerator=sum(e['startup_restoration_numerator'] for e in endpoint),denominator=len(ok)),events=incidence)
    return output


def analyze_collection(protocol_path, run_root, output):
    from .study import validate_collection, read_frames
    from .prediction import ServiceEpisode, ServicePredictor
    from ..prevention.config import PreventionConfig
    phase=load_phase(protocol_path); validation=validate_collection(protocol_path,run_root)
    out=reserve(output); root=Path(run_root).resolve(); package=Path(phase['_path']).parent
    rows=validation['rows']; episodes=[]; mechanism=[]; timings=[]; component_times={}
    for row in rows:
        if row['status']!='ok': continue
        attempt=root/row['cell_id']/f"attempt-{row['attempt']}"; episode=safe_path(attempt,row['episode_path'])
        frames=read_frames(episode,PreventionConfig(**row['config']))
        e=ServiceEpisode(row['cell_id'],phase['phase'],row['policy'],row['regime'],row['context'],row['layout_seed'],row['radio_seed'],
            PreventionConfig(**row['config']),frames,{},row['trace_sha256']); episodes.append(e)
        decisions=[strict_loads(line) for line in next((attempt/'collector/episodes').rglob('decisions.jsonl')).read_text(encoding='utf-8').splitlines()]
        if len(decisions)!=240 or [d['step'] for d in decisions]!=list(range(240)): raise ValueError('Full decision trace required')
        row['endpoints']=episode_endpoints(e,decisions)
        for d in decisions:
            for name,value in d.get('timing_seconds',{}).items():
                if type(value) not in (int,float) or not math.isfinite(value) or value<0: raise ValueError('Invalid monotonic decision timing')
                component_times.setdefault(name,[]).append(value)
            if d.get('start',d.get('intervention_started',False)):
                observed=frames[d['step']]['service_observation']['route']['nodes']
                plan=d.get('plan') or {}; chain=plan.get('chain')
                mechanism.append(dict(cell_id=row['cell_id'],decision=d,observed_route_before=observed,
                    planned_chain=chain,actual_planned_path_mismatch=chain is not None and list(chain)!=observed,
                    following5s=[f for f in frames if d['step']<f['step']<=d['step']+5],
                    following5s_decisions=[other for other in decisions if d['step']<other['step']<=d['step']+5]))
        samples=strict_json(attempt/'timings.json')
        for sample in samples:
            timings.append(dict(cell_id=row['cell_id'],**sample))
    contrasts,secondary=_effect_tables(rows)
    hold_armed={(r['regime'],r['layout_seed'],r['radio_seed'],r['context']) for r in rows if r['policy']=='hold' and r['status']=='ok' and r['endpoints']['armed_by_step20_numerator']}
    subgroup=[r for r in rows if (r['regime'],r['layout_seed'],r['radio_seed'],r['context']) in hold_armed]
    forecast=forecast_scores(episodes,ServicePredictor.load(package/'new-model.json')) if phase['new_model'] else None
    values=[t['decision_seconds'] for t in timings]
    report=dict(schema='service-v2-analysis-v1',phase=phase['phase'],freeze_sha256=phase['_freeze_sha256'],
        complete=validation['complete'],planned_cells=validation['planned_cells'],ok_cells=validation['ok_cells'],
        rows=rows,contrasts=contrasts,secondary_contrasts=secondary,descriptive=_descriptive(rows),
        hold_armed_by20_subgroup=dict(definition='descriptive fixed-reference HOLD armed by20; primary remains all seeds',descriptive=_descriptive(subgroup)),
        forecasts=forecast,forecast_availability_reason=None if forecast is not None else
            'Collection-only package freezes new_model=null. Development diagnostics use the public forecast_scores API with an independently hash-pinned fitted model and fresh immutable artifact; no earlier package is modified.',mechanism_traces=mechanism,
        timing=dict(decision_seconds=dict(median=float(np.median(values)) if values else None,p95=float(np.quantile(values,.95)) if values else None,max=max(values) if values else None),
            components_seconds={name:dict(count=len(v),median=float(np.median(v)),p95=float(np.quantile(v,.95)),max=max(v)) for name,v in component_times.items()},
            subprocess_wall_seconds=[dict(cell_id=r['cell_id'],attempt=a['attempt'],seconds=a['wall_seconds']) for r in rows for a in r['attempt_records']],
            synchronous_bridge='simulator blocks while deciding; processing is not modeled control-plane delay'),
        inference_limits=['Scenario pairs are units; shared seeds do not match packet-level randomness.',
            'Secondary intervals descriptive; static four-pair uncertainty is imprecise.',
            'Distance is not energy, endurance or battery usage.',
            '3s routing allowance is a hypothesis, not measured transport latency.',
            'Sink/FlowMonitor differences retained; drop reasons overlap.',
            'No artificial drain, efficacy exclusion, superiority guarantee or action-benefit probability.'])
    write_json(out/'analysis.json',report)
    write_json(out/'validation.json',validation)
    inputs=dict(protocol_sha256=sha256(protocol_path),freeze_sha256=phase['_freeze_sha256'],collection_sha256=sha256(root/'collection.json'),
        original_protocol_path=str(Path(protocol_path).resolve()),original_run_root=str(root),original_analysis_root=str(out.resolve()),
        relocation='Manifest file names are relative to each artifact root. Copy complete roots byte-for-byte; pass downloaded analysis and engineering roots to report. Absolute roots inside data remain historical provenance and are never rewritten or used to resolve report input files.')
    write_json(out/'inputs.json',inputs)
    write_json(out/'manifest.json',dict(schema='service-v2-analysis-manifest-v1',files={p.name:sha256(p) for p in out.iterdir() if p.is_file()}))
    return report
