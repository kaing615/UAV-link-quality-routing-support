"""Create immutable accepted numerical-method and paired-study protocols, once."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.control.path_service.config import PathServiceConfig,POLICIES
from src.control.prevention.config import PreventionConfig
from src.control.prevention.prediction import sha256

ROOT=Path(__file__).resolve().parents[2]
SPEC=ROOT/'.superpowers/sdd/2026-11-11-routing-service'
MODEL='reports/uav_prevention_pilot_2026-11-11/predictor.json'
MODEL_HASH='3b74301ccb0e659ba6d2b593250decabd0f317386adfa4cc5d5b4e5174bacae8'


def write(path,value):
    with path.open('x',encoding='utf-8') as stream: json.dump(value,stream,indent=2,allow_nan=False)


def main():
    out=SPEC/'task-3-frozen'; out.mkdir(exist_ok=False)
    method=dict(schema='path-service-method-v1',config=asdict(PathServiceConfig()),
                algorithm='task-3-design-proposal.md with task-3-design-accepted.md clarifications',
                proposal_sha256=sha256(SPEC/'task-3-design-proposal.md'),accepted_sha256=sha256(SPEC/'task-3-design-accepted.md'),
                ablations={'no-service':'Remove observed route/delivery and path selection from trigger and planner; measured global bridge maximum forecast, current global components restoration, fixed all-relay projection order. Retain endpoint lookahead, deadline and budget rules. Grouped ablation.',
                           'no-deadline':'Keep endpoint lookahead and travel feasibility; start on any crossing; remove horizon versus required gating only.',
                           'no-budget':'Remove complete remaining-budget rejection; replace relative budget denominator with original configured budget; retain episode feasibility, exact hard one-step budget and safety.'},
                hypotheses=['rho .50 and service .95 are fixed untuned conventions','path risk sum is a heuristic, not calibrated probability or action counterfactual','candidate chain cannot install routes; nonselected relays remain radio-active','compact geometry and induced trajectory covariate shift; no automatic calibration transfer'],
                observer_origin='simulation time zero; frame0 at warmup+1 before application start warmup+1.01',
                terminal_ordering='live terminal snapshot then UAVQUIT stops simulator at that boundary; final accounting after existing FlowMonitor metrics; preserve any tail/late counters separately')
    write(out/'method.json',method)
    pilot=[dict(regime='n8_spacing030',layout_seed=80001,radio_seed=180001,num_uavs=8,initial_spacing_fraction=.30),dict(regime='n12_spacing055',layout_seed=80002,radio_seed=180002,num_uavs=12,initial_spacing_fraction=.55)]
    formal=[]
    for n,spacing,start in [(8,.55,81001),(8,.30,82001),(12,.55,83001),(12,.30,84001)]:
        formal += [dict(regime=f'n{n}_spacing{int(spacing*100):03}',layout_seed=s,radio_seed=s+100000,num_uavs=n,initial_spacing_fraction=spacing) for s in range(start,start+20)]
    for phase,scenarios in [('pilot',pilot),('formal',formal)]:
        write(out/f'{phase}.json',dict(schema='path-service-study-v1',phase=phase,policies=POLICIES,scenarios=scenarios,
              config=asdict(PreventionConfig()),method_sha256=sha256(out/'method.json'),model=MODEL,model_sha256=MODEL_HASH,
              baseline_settings={'predictive-simple':{'threshold':.5,'routing_delay_s':1.},'predictive-learned':{'threshold':.35,'routing_delay_s':3.}},
              service_observation=True,expected_episodes=len(scenarios)*len(POLICIES),max_workers=4,full_horizon=True,
              tuning_search_count=0,pilot_use='engineering only; no PDR-driven selection; technical fixes preserve originals and rerun affected paired grid in new root',
              endpoints=['actual UDP full-horizon sink received/application offered','conditional delay of received packets or null','per-relay and total motion','source-destination probe outage','whole-graph disconnection','OLSR usable fraction and traversal success','safety rejections','action starts/feasibility'],
              evaluation={'bootstrap_resamples':10000,'rng_seed':91011,'confidence_percent':95,'unit':'whole paired scenario','regime_pool':'equal regime stratified','comparisons':'path-service minus each of eight other roles','failure_handling':'primary incomplete with any failed case; complete pairs and attempted-grid failures; sensitivity bounds separate','multiplicity':'descriptive intervals; no multiple-comparison superiority claim'}))
    write(out/'freeze.json',dict(schema='path-service-freeze-v1',files={p.name:sha256(p) for p in sorted(out.glob('*.json'))},frozen_before_collection=True))
    print(out)

if __name__=='__main__': main()
