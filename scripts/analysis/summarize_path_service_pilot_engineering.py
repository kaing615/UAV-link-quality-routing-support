"""Offline same-frame planner coincidence for frozen pilot or full formal runs.

Never launches a simulator or changes the frozen collection snapshot. Compare
the old all-relay planner on the SAME retained pre-action frame, rather than
comparing different policy trajectories. Tolerances are reporting precision,
not selected controller settings: absolute 1e-6 m targets, 1e-8 actions, rtol=0.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.control.prevention.config import PreventionConfig
from src.control.prevention.controller import target_plan
from src.control.prevention.prediction import sha256,strict_json


def matched(left,right,tolerance):
    return bool(np.allclose(np.asarray(left,float),np.asarray(right,float),atol=tolerance,rtol=0))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.output.exists(): raise ValueError('Fresh analysis output required')
    root=args.run.resolve(); rows=strict_json(root/'manifest.json'); protocol=strict_json(root/'protocol.json')
    expected=Counter((p,s['layout_seed'],s['radio_seed']) for p in protocol['policies'] for s in protocol['scenarios'])
    if Counter((r['policy'],r['layout_seed'],r['radio_seed']) for r in rows)!=expected: raise ValueError('Run grid differs from protocol')
    phase=protocol['phase']
    expected_count={'pilot':18,'formal':720}.get(phase)
    if expected_count is None or len(rows)!=expected_count: raise ValueError('Expected frozen 18-pilot or 720-formal grid')
    if any(r['status'] not in ('ok','failed') for r in rows): raise ValueError('Collection has nonterminal cells; wait for whole run')
    if phase=='pilot' and any(r['status']!='ok' for r in rows): raise ValueError('Pilot engineering requires complete successful pilot')
    failures=[{k:r.get(k) for k in ('policy','regime','layout_seed','radio_seed','status','error','case_dir')} for r in rows if r['status']!='ok']
    verified=0
    for row in rows:
        if row['status']=='ok' and not row.get('artifact_hashes'): raise ValueError('Successful case lacks raw artifact hashes')
        for name,digest in row.get('artifact_hashes',{}).items():
            path=(root/name).resolve()
            if not path.is_relative_to(root) or sha256(path)!=digest: raise ValueError('Raw hash mismatch: '+name)
            verified+=1
    summaries=[]; comparisons=[]
    for row in rows:
        if not row['policy'].startswith('path-service'): continue
        if row['status']!='ok': continue
        case=(root/row['case_dir']).resolve()
        if not case.is_relative_to(root): raise ValueError('Case escapes run')
        traces=list(case.rglob('transitions.jsonl')); logs=list(case.rglob('decisions.jsonl'))
        if len(traces)!=1 or len(logs)!=1: raise ValueError('Ambiguous raw episode')
        frames=[json.loads(line)['frame'] for line in traces[0].read_text().splitlines()]
        decisions=[json.loads(line) for line in logs[0].read_text().splitlines()]
        config=PreventionConfig(**strict_json(traces[0].with_name('config.json'))['config'])
        if [f['step'] for f in frames]!=list(range(121)) or [d['step'] for d in decisions]!=list(range(120)): raise ValueError('Incomplete horizon')
        counts=Counter(); counts['all_decision_frames']=len(decisions)
        for frame,decision in zip(frames,decisions):
            old_action,_=target_plan(frame,config)
            chosen=decision['planner']['chosen']; moving=decision['commanded_motion']
            item=dict(policy=row['policy'],regime=row['regime'],layout_seed=row['layout_seed'],radio_seed=row['radio_seed'],step=frame['step'],
                      has_chosen_plan=chosen is not None,commanded_motion=moving,
                      target_coincidence=None,nominal_action_coincidence=None,
                      protected_proposal_coincidence=matched(decision['proposed_action'],old_action,1e-8),
                      sent_vs_old_geometry_action_coincidence=matched(decision['sent_action'],old_action,1e-8),
                      simulator_budget_scaled=bool(decision['simulator_budget_scaled_relays']),
                      one_step_safety_rejected=bool(decision['one_step_safety_rejected']))
            counts['protected_proposal_coincidences']+=item['protected_proposal_coincidence']
            counts['sent_vs_old_geometry_action_coincidences']+=item['sent_vs_old_geometry_action_coincidence']
            counts['commanded_motion_frames']+=moving
            counts['active_sent_coincidences']+=moving and item['sent_vs_old_geometry_action_coincidence']
            counts['budget_scaled_proposal_frames']+=item['simulator_budget_scaled']
            counts['budget_scaled_sent_frames']+=moving and item['simulator_budget_scaled']
            if chosen:
                counts['chosen_plan_frames']+=1
                positions=np.asarray(frame['positions'],float); start=positions[config.source]; axis=positions[config.destination]-start
                ordered=sorted(config.relay_ids,key=lambda i:(float((positions[i]-start)@axis),i))
                old_targets={i:start+(rank+1)/(len(ordered)+1)*axis for rank,i in enumerate(ordered)}
                new_targets=np.asarray(chosen['targets'],float)
                item['target_coincidence']=matched(new_targets[list(config.relay_ids)],[old_targets[i] for i in config.relay_ids],1e-6)
                nominal=[]
                for i in config.relay_ids:
                    delta=new_targets[i]-positions[i];nominal.extend(delta/max(config.max_speed,float(np.linalg.norm(delta))))
                item['nominal_action_coincidence']=matched(nominal,old_action,1e-8)
                item['selected_relays']=chosen['chain'][1:-1]
                item['target_max_abs_difference_m']=float(np.max(np.abs(new_targets[list(config.relay_ids)]-np.asarray([old_targets[i] for i in config.relay_ids]))))
                item['nominal_action_max_abs_difference']=float(np.max(np.abs(np.asarray(nominal)-old_action)))
                counts['target_coincidences']+=item['target_coincidence']
                counts['nominal_action_coincidences']+=item['nominal_action_coincidence']
                counts['strict_subset_plan_frames']+=len(chosen['chain'][1:-1])<len(config.relay_ids)
            else: counts['no_chosen_plan_frames']+=1
            comparisons.append(item)
        def measure(numerator,denominator):
            n,d=counts[numerator],counts[denominator]
            return dict(numerator=n,denominator=d,fraction=n/d if d else None)
        summaries.append(dict(policy=row['policy'],regime=row['regime'],layout_seed=row['layout_seed'],radio_seed=row['radio_seed'],counts=dict(counts),
            target_coincidence_on_chosen_plans=measure('target_coincidences','chosen_plan_frames'),
            nominal_action_coincidence_on_chosen_plans=measure('nominal_action_coincidences','chosen_plan_frames'),
            protected_proposal_coincidence_all_frames=measure('protected_proposal_coincidences','all_decision_frames'),
            sent_action_coincidence_all_frames=measure('sent_vs_old_geometry_action_coincidences','all_decision_frames'),
            sent_action_coincidence_when_commanded=measure('active_sent_coincidences','commanded_motion_frames'),
            no_plan_fraction=measure('no_chosen_plan_frames','all_decision_frames'),
            strict_subset_fraction_on_chosen_plans=measure('strict_subset_plan_frames','chosen_plan_frames')))
    args.output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='path-service-planner-coincidence-v1',phase=phase,
        scope='engineering description only; two scenarios do not support efficacy comparison or selection' if phase=='pilot' else 'descriptive same-frame planner coincidence; not an efficacy endpoint or counterfactual policy comparison',
        expected_cases=expected_count,successful_cases=sum(r['status']=='ok' for r in rows),failures=failures,complete_grid=not failures,
        hashes_verified=verified,run=str(root),input_hashes={n:sha256(root/n) for n in ('manifest.json','protocol.json','provenance.json','status.json')},
        analysis_sha256=sha256(Path(__file__)),target_atol_m=1e-6,action_atol=1e-8,rtol=0,
        definitions=dict(reference='Old target_plan evaluated on each SAME new-policy pre-action frame, source/destination current positions; all relays projection ordered, equal fractions; no trigger/budget/safety in reference.',
            target='Compare complete relay target vectors: chosen subset targets and stationary nonselected relays versus old all-relay targets. Chosen-plan frames only; no-plan excluded explicitly.',
            nominal_action='Normalize each chosen target displacement by max(config.max_speed,distance), before one-step hard-budget/safety protection; compare with old target_plan. Chosen-plan denominator.',
            protected_proposal='Retained proposed_action after new one-step protection versus old unprotected normalized geometry action; all 120 frames, zero proposals included.',
            sent_action='Retained sent_action after activation gating versus old geometry action evaluated on same frame. All frames and commanded-motion subset reported separately; this is not the historical baseline trajectory.',
            tolerance='Absolute componentwise tolerance, zero relative tolerance; reporting precision fixed before calculation, not a policy setting.'),
        summaries=summaries,comparisons=comparisons,
        timing_limit='T=max(d_i/V) is nominal static-target constant-speed planning time. Existing simulator remaining-budget scaling can make realized steps smaller; endpoint changes/safety rejection also invalidate completion guarantees. No changes made.')
    filename='pilot-engineering.json' if phase=='pilot' else 'formal-coincidence.json'
    with (args.output/filename).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(hashes_verified=verified,new_policy_cases=len(summaries),decision_comparisons=len(comparisons))))
    return result

if __name__=='__main__':main()
