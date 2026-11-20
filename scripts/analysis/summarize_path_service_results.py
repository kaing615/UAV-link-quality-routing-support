"""Descriptive final formal summaries, mechanism records and scientific figures.

Consumes approved evaluator/coincidence outputs and retained raw records. No
simulation, tuning, fitting or additional bootstrap. All four regimes and nine
roles are retained. Output must be new; source/runtime/raw hashes stay separate.
"""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.control.prevention.prediction import strict_json,sha256


def write_json(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def write_csv(path,rows):
    names=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=names);writer.writeheader();writer.writerows(rows)


def mean(rows,key):
    values=[r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','evaluation','coincidence','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(argv)
    if args.output.exists():raise ValueError('Fresh final summary output required')
    rows=strict_json(args.run/'manifest.json');protocol=strict_json(args.run/'protocol.json')
    evaluation=strict_json(args.evaluation);coincidence=strict_json(args.coincidence)
    if len(rows)!=720 or not evaluation['complete_primary_grid'] or not coincidence['complete_grid']:raise ValueError('Expected complete formal grid')
    if sha256(args.run/'manifest.json')!=evaluation['input_hashes']['manifest.json']:raise ValueError('Evaluation/raw manifest mismatch')
    if sha256(args.run/'manifest.json')!=coincidence['input_hashes']['manifest.json']:raise ValueError('Coincidence/raw manifest mismatch')
    policies=protocol['policies'];regimes=list(dict.fromkeys(s['regime'] for s in protocol['scenarios']))
    groups=defaultdict(list)
    for row in rows:
        if row['status']!='ok':raise ValueError('Non-ok case')
        groups[(row['regime'],row['policy'])].append(row)
    metric_names=('actual_udp_pdr','total_relay_travel','max_relay_travel','mean_delay_ms','source_dest_disconnected_fraction',
        'disconnected_fraction','olsr_unusable_fraction','traversal_success_fraction','rejected_actions','intervention_starts','first_action_step')
    summaries=[]
    for regime in [*regimes,'pooled']:
        for policy in policies:
            sample=groups[(regime,policy)] if regime!='pooled' else [r for r in rows if r['policy']==policy]
            if len(sample)!=(20 if regime!='pooled' else 80):raise ValueError('Unequal planned group')
            summaries.append(dict(regime=regime,policy=policy,episodes=len(sample),zero_received_episodes=sum(r['application_received_packets']==0 for r in sample),
                valid_delay_episodes=sum(r['mean_delay_ms'] is not None for r in sample),
                offered_packets=sum(r['application_offered_packets'] for r in sample),received_packets=sum(r['application_received_packets'] for r in sample),
                source_tx_packets=sum(r['tx_packets'] for r in sample),
                **{key:mean(sample,key) for key in metric_names}))
    episodes=[];relay_rows=[];first_actions=[];decision_rows=[];detailed={};signals=defaultdict(Counter)
    for row in rows:
        episodes.append({k:row.get(k) for k in ('regime','policy','layout_seed','radio_seed','status',*metric_names,
            'application_offered_packets','application_received_packets','original_flowmonitor_pdr','first_partition_step')})
        for relay,distance in row['per_relay_travel_m'].items():relay_rows.append(dict(regime=row['regime'],policy=row['policy'],layout_seed=row['layout_seed'],radio_seed=row['radio_seed'],relay=int(relay),travel_m=distance))
        if not row['policy'].startswith('path-service'):continue
        case=args.run/row['case_dir'];decisions=[json.loads(l) for l in next(case.rglob('decisions.jsonl')).read_text().splitlines()]
        frames=[json.loads(l)['frame'] for l in next(case.rglob('transitions.jsonl')).read_text().splitlines()]
        if len(decisions)!=120 or len(frames)!=121:raise ValueError('Incomplete mechanism input')
        key=(row['regime'],row['policy']);count=signals[key];detailed[(row['policy'],row['layout_seed'],row['radio_seed'])]=(decisions,frames)
        count['episodes']+=1;count['decisions']+=120
        first=next((d for d in decisions if d['intervention_started']),None)
        for index,d in enumerate(decisions):
            chosen=d['planner']['chosen'];moving=d['commanded_motion'];service=d['service_state']
            count['commanded_motion_frames']+=moving;count['chosen_plan_frames']+=chosen is not None
            count['no_plan_frames']+=chosen is None;count['one_step_safety_rejected_frames']+=d['one_step_safety_rejected']
            count['simulator_rejected_intervals']+=bool(frames[index+1]['safety_rejected'])
            count['forecast_starts']+=d['intervention_started'] and d['activation_context']=='forecast_prevention'
            count['restoration_starts']+=d['intervention_started'] and d['activation_context']=='restoration'
            count['no_forecast_crossing_frames']+=d['risk']['crossing_horizon_s'] is None
            count['forecast_crossing_frames']+=d['risk']['crossing_horizon_s'] is not None
            count['restoration_state_frames']+=service['restoration']
            count['restoration_with_forecast_crossing_frames']+=service['restoration'] and d['risk']['crossing_horizon_s'] is not None
            count['risk_outside_action_window_frames']+=d['risk']['crossing_horizon_s'] is not None and not d['action_window_reached']
            count['outside_window_without_restoration_frames']+=d['risk']['crossing_horizon_s'] is not None and not d['action_window_reached'] and not service['restoration']
            count['budget_scaled_sent_frames']+=moving and bool(d['simulator_budget_scaled_relays'])
            count['route_missing_frames']+=service['route_missing'];count['service_unknown_frames']+=service['service_unknown']
            count['delivery_degraded_frames']+=service['delivery_degraded']
            count['probe_inconsistent_frames']+=service['probe_inconsistent']
            count['recent_arrival_ratio_gt1_frames']+=service['recent_arrival_offer_ratio'] is not None and service['recent_arrival_offer_ratio']>1
            count['path_origin:'+d['risk']['path_origin']]+=1
            if chosen:
                count['strict_subset_plan_frames']+=len(chosen['chain'])-2<row['num_uavs']-2
                count['actual_planned_path_mismatch_frames']+=bool(d['actual_planned_path_mismatch'])
                count['planned_hops_sum']+=len(chosen['chain'])-1
                if d['actual_path_hops'] is not None:
                    count['observed_success_on_chosen_plan_frames']+=1;count['observed_hops_sum']+=d['actual_path_hops']
            for reason,value in d['planner']['rejection_counts'].items():count['candidate_rejection:'+reason]+=value
            if chosen is None:count['hold_reason:'+d['planner']['reason']]+=1
        if first:
            step=first['step'];currently_out=not frames[step]['source_dest_connected']
            next_out=next((f['step'] for f in frames[step+1:] if not f['source_dest_connected']),None)
            first_actions.append(dict(regime=row['regime'],policy=row['policy'],layout_seed=row['layout_seed'],radio_seed=row['radio_seed'],
                first_command_start_step=step,sim_time_s=first['sim_time_s'],context=first['activation_context'],
                observed_route_degraded=first['service_state']['route_degraded'],observed_delivery_degraded=first['service_state']['delivery_degraded'],
                current_source_destination_outage=currently_out,next_observed_source_destination_outage_step=next_out,
                lead_to_next_observed_outage_s=next_out-step if next_out is not None and not currently_out else None,
                interpretation='descriptive command/event timing; not prevented-outage counterfactual evidence'))
    identical=[]
    for scenario in protocol['scenarios']:
        seed=(scenario['layout_seed'],scenario['radio_seed'])
        full,ff=detailed[('path-service',*seed)];ablated,af=detailed[('path-service-no-deadline',*seed)]
        same_actions=all(a['sent_action']==b['sent_action'] for a,b in zip(full,ablated))
        same_frames=ff==af
        identical.append(dict(**scenario,sent_actions_exactly_equal=same_actions,all_frame_fields_exactly_equal=same_frames))
    mechanics=[dict(regime=r,policy=p,**dict(signals[(r,p)])) for r in regimes for p in policies if p.startswith('path-service')]
    coincidences=[]
    for regime in regimes:
        for policy in policies:
            if not policy.startswith('path-service'):continue
            items=[x for x in coincidence['summaries'] if x['regime']==regime and x['policy']==policy]
            count=Counter()
            for item in items:count.update(item['counts'])
            coincidences.append(dict(regime=regime,policy=policy,**dict(count)))
    args.output.mkdir(parents=True,exist_ok=False)
    write_csv(args.output/'policy-regime-means.csv',summaries);write_csv(args.output/'episodes.csv',episodes)
    write_csv(args.output/'per-relay-motion.csv',relay_rows);write_csv(args.output/'first-action-context.csv',first_actions)
    write_csv(args.output/'mechanism-counts.csv',mechanics);write_csv(args.output/'planner-coincidence-counts.csv',coincidences)
    write_csv(args.output/'paired-scenario-differences.csv',[{k:v for k,v in x.items() if k!='differences'}|x['differences'] for x in evaluation['scenario_differences']])
    cirows=[]
    for reference,value in evaluation['comparisons'].items():
        for regime,values in [*value['regimes'].items(),('pooled_equal_regime',value['pooled_equal_regime'])]:
            for metric,v in values.items():
                ci=v['scenario_bootstrap_95pct_ci'];cirows.append(dict(reference=reference,regime=regime,metric=metric,mean_difference=v['mean_difference'],
                    ci_low=ci[0] if ci else None,ci_high=ci[1] if ci else None,complete_pairs=v.get('complete_pairs'),
                    expected_pairs=v.get('expected_pairs'),represented_regimes=v.get('represented_regimes'),expected_regimes=v.get('expected_regimes')))
    write_csv(args.output/'all-paired-confidence-intervals.csv',cirows)
    result=dict(schema='path-service-final-descriptive-summary-v1',scope='formal-v2 only; all predefined regimes and roles; no tuning',
        summaries=summaries,mechanisms=mechanics,coincidences=coincidences,first_actions=first_actions,no_deadline_equivalence=identical,
        no_deadline_actions_equal_scenarios=sum(x['sent_actions_exactly_equal'] for x in identical),
        no_deadline_all_frames_equal_scenarios=sum(x['all_frame_fields_exactly_equal'] for x in identical),
        input_hashes={'manifest':sha256(args.run/'manifest.json'),'evaluation':sha256(args.evaluation),'coincidence':sha256(args.coincidence)},
        code_sha256=sha256(Path(__file__)))
    write_json(args.output/'summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    refs=[x for x in policies if x!='path-service'];ys=np.arange(len(refs))
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
    for ax,metric,factor,label in [(axes[0],'actual_udp_pdr',100,'PDR difference (percentage points)'),(axes[1],'total_relay_travel',1,'Relay motion difference (m)')]:
        for y,reference in zip(ys,refs):
            value=evaluation['comparisons'][reference]['pooled_equal_regime'][metric];center=value['mean_difference']*factor
            lo,hi=np.asarray(value['scenario_bootstrap_95pct_ci'])*factor
            ax.errorbar(center,y,xerr=[[center-lo],[hi-center]],fmt='o',color='#245d78',capsize=3)
        ax.axvline(0,color='0.5',lw=.8);ax.set_xlabel(label);ax.grid(axis='x',alpha=.2)
    axes[0].set_yticks(ys,refs);axes[0].invert_yaxis();fig.suptitle('Path-service minus reference: paired scenario bootstrap 95% intervals')
    fig.tight_layout();fig.savefig(args.output/'paired-pdr-motion.png',dpi=200);fig.savefig(args.output/'paired-pdr-motion.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    matrix=np.asarray([[next(x for x in summaries if x['regime']==regime and x['policy']==policy)['actual_udp_pdr']*100 for regime in regimes] for policy in policies])
    image=axes[0].imshow(matrix,aspect='auto',cmap='Blues',vmin=0)
    axes[0].set_xticks(range(4),['n8 / .55','n8 / .30','n12 / .55','n12 / .30']);axes[0].set_yticks(range(9),policies)
    for i in range(9):
        for j in range(4):axes[0].text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if matrix[i,j]>matrix.max()*.6 else 'black')
    fig.colorbar(image,ax=axes[0],label='Mean application UDP PDR (%)',shrink=.7)
    axes[1].scatter([x['source_dest_disconnected_fraction']*100 for x in summaries if x['regime']!='pooled'],
                    [x['actual_udp_pdr']*100 for x in summaries if x['regime']!='pooled'],color='#245d78',s=22)
    axes[1].set_xlabel('Source–destination probe outage (%)');axes[1].set_ylabel('Mean application UDP PDR (%)')
    axes[1].set_title('36 prespecified policy × regime means');axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(args.output/'regime-pdr-service-gap.png',dpi=200);fig.savefig(args.output/'regime-pdr-service-gap.pdf');plt.close(fig)
    print(json.dumps(dict(cases=len(rows),summaries=len(summaries),decision_frames=sum(x['decisions'] for x in mechanics),
        no_deadline_actions_equal=result['no_deadline_actions_equal_scenarios'],no_deadline_all_frames_equal=result['no_deadline_all_frames_equal_scenarios'])))
    return result

if __name__=='__main__':main()
