"""One-cell legacy collector injection and strict v2 collection loading.

Run layout: collection.json, <cell_id>/attempt-{0,1}/, retained result.json,
artifacts.json and status revisions. No model fitting or seed selection here.
"""
from dataclasses import asdict
from functools import partial
import csv
import json
from math import isfinite
from pathlib import Path
import time

from .config import POLICIES, ServiceV2Config
from .protocol import (ROOT, REFERENCE_SHA256, load_phase, verify_runtime, strict_json, strict_loads,
                       write_json, sha256, safe_path, reserve, publish_status)


def _collector_decision(action, decision):
    """Add the unchanged collector's required fields without changing v2 truth.

    Collector intervention counts alias logical v2 starts, including a fresh
    per-frame plan in the ungated ablation. Commanded motion is an emitted
    nonzero action, never inferred execution; frames/previous_command retain
    measured movement and safety rejection separately.
    """
    start=decision['start']
    if type(start) is not bool: raise ValueError('V2 start must be an explicit boolean')
    if decision['sent_action']!=action.tolist(): raise ValueError('V2 sent action disagrees with emitted action')
    if {'intervention_started','commanded_motion','collector_compatibility'} & decision.keys():
        raise ValueError('V2 decision already defines collector compatibility fields')
    return dict(decision,intervention_started=start,commanded_motion=bool(any(value!=0 for value in action)),
        collector_compatibility=dict(schema='service-v2-collector-decision-v1',
            intervention_started='alias_of_v2_start',commanded_motion='any_nonzero_emitted_action',
            executed_motion='measured_frames_and_previous_command_only'))


class _CollectorCompatibleV2:
    def __init__(self, controller): self.controller=controller
    def act(self, frame):
        action,decision=self.controller.act(frame)
        return action,_collector_decision(action,decision)


def make_controller_factory(new_predictor, old_predictor, method, model_hash, method_hash):
    from ..prevention.controller import PreventionController
    from ..path_service.controller import PathServiceController
    from ..path_service.config import PathServiceConfig
    from .controller import ServiceV2Controller
    if old_predictor is None or old_predictor.artifact['comm_range']!=240: raise ValueError('Validated reference predictor required')
    if method!=ServiceV2Config(): raise ValueError('Adopted method required')
    def factory(policy, config, predictor=None, threshold=.5, routing_delay_s=3.):
        if policy not in POLICIES: raise ValueError('Unknown role')
        if config.comm_range!=240: raise ValueError('Reference/v2 range mismatch')
        if policy in ('hold','reactive','geometry'):
            return PreventionController(policy,config,predictor=None,threshold=.5,routing_delay_s=3.)
        if policy=='path-service-v1-reference':
            return PathServiceController('path-service',config,old_predictor,method=PathServiceConfig(),
                                         model_hash=REFERENCE_SHA256)
        return _CollectorCompatibleV2(ServiceV2Controller(policy,config,new_predictor,method,model_hash,method_hash))
    return factory


def read_frames(episode_dir, config):
    from ..bridge import validate_frame
    from .observation import service_observation
    frames=[]
    for line in (Path(episode_dir)/'transitions.jsonl').read_text(encoding='utf-8').splitlines():
        row=strict_loads(line)
        f=row['frame']
        validate_frame(f,config,len(frames))
        service_observation(f,config,frames[-1] if frames else None)
        frames.append(f)
    if len(frames)!=config.steps+1 or [f['step'] for f in frames]!=list(range(config.steps+1)):
        raise ValueError('Full frames 0..240 required')
    if not frames[-1]['terminal'] or any(f['terminal'] for f in frames[:-1]): raise ValueError('Terminal frame boundary')
    return frames


def reconcile_episode(episode_dir, config):
    root=Path(episode_dir); frames=read_frames(root,config)
    final=strict_json(root/'service_final_accounting.json')
    if (final.get('schema')!='service-final-accounting-v1' or final.get('counting_origin_s')!=0
            or final.get('sim_time_s')!=config.warmup+config.steps+1): raise ValueError('Terminal accounting schema/stop boundary')
    with (root/'closed_loop_metrics.csv').open(newline='',encoding='utf-8') as stream: rows=list(csv.DictReader(stream))
    if len(rows)!=1: raise ValueError('One terminal FlowMonitor row required')
    raw=rows[0]
    required=('offered_packets','source_tx_packets','sink_rx_packets','flowmonitor_rx_packets',
              'offered_after_last_frame','source_tx_after_last_frame','late_sink_rx_after_last_frame')
    if any(type(final.get(k)) is not int or final[k]<0 for k in required): raise ValueError('Terminal counter schema')
    for left,right in [('offered_packets','offered_packets'),('source_tx_packets','tx_packets'),('flowmonitor_rx_packets','rx_packets')]:
        if final[left]!=int(raw[right]): raise ValueError('Same-layer terminal mismatch: '+left)
    pdr=float(raw['pdr'])
    if not isfinite(pdr) or pdr<0: raise ValueError('Nonfinite/negative FlowMonitor PDR')
    offered=final['offered_packets']; received=final['sink_rx_packets']
    if offered!=15000 or final['source_tx_packets']>offered or received>offered or final['flowmonitor_rx_packets']>final['source_tx_packets']:
        raise ValueError('Impossible terminal packet counts')
    if abs(final['flowmonitor_rx_packets']/offered-pdr)>1e-5: raise ValueError('FlowMonitor PDR mismatch')
    live=frames[-1]['service_observation']
    if final['last_live_frame_time_s']!=live['sim_time_s']: raise ValueError('Live/final boundary mismatch')
    for key,late in [('offered_packets','offered_after_last_frame'),('source_tx_packets','source_tx_after_last_frame'),('sink_rx_packets','late_sink_rx_after_last_frame')]:
        if final[key]-live['cumulative'][key]!=final[late]: raise ValueError('Late accounting mismatch')
    travel={str(i):frames[-1]['travel'][i] for i in config.relay_ids}
    if any(v>config.budget+1e-6 for v in travel.values()): raise ValueError('Travel budget violation')
    return dict(application_offered_packets=offered,application_received_packets=received,
        application_source_tx_packets=final['source_tx_packets'],actual_udp_pdr=received/offered,
        flowmonitor_received_packets=final['flowmonitor_rx_packets'],flowmonitor_pdr=pdr,
        sink_minus_flowmonitor_received=received-final['flowmonitor_rx_packets'],
        per_relay_travel_m=travel,total_relay_travel_m=sum(travel.values()),
        meters_per_offered_1000=sum(travel.values())/offered*1000,
        budget_fraction_per_relay={k:v/config.budget for k,v in travel.items()},
        final_service_accounting=final)


def _collection_manifest(root, phase):
    value=dict(schema='service-v2-collection-v1',phase=phase['phase'],protocol=phase.get('protocol_path_at_freeze',phase['_path']),
        freeze_sha256=phase['_freeze_sha256'],cells=[c['cell_id'] for c in phase['cells']])
    path=root/'collection.json'
    if not path.exists():
        try: write_json(path,value)
        except FileExistsError: pass
    if strict_json(path)!=value: raise ValueError('Collection protocol mismatch')


def _record_process_evidence(attempt, result, process_started, capture_errors):
    """Retain capture failures explicitly; absence never means an empty log."""
    actual_exit=result['actual_exit']
    outcome=('completed' if actual_exit is not None else 'not_started' if process_started is False
             else 'exit_unavailable' if process_started is True else 'process_unavailable')
    error=result.get('error') or '; '.join(capture_errors) or 'Direct process creation/exit was not observed.'
    streams={}
    for stream in ('stdout','stderr'):
        paths=list(attempt.rglob(f'simulator.{stream}.log'))
        if len(paths)==1:
            path=paths[0]
            entry=dict(status='error' if capture_errors else 'captured',path=path.relative_to(attempt).as_posix(),sha256=sha256(path))
            if capture_errors: entry.update(reason='capture_error',error='; '.join(capture_errors))
        else:
            entry=dict(status='unavailable',reason='process_not_started' if process_started is False else 'capture_unavailable',
                error=error if not paths else 'Multiple ambiguous simulator log paths: '+str(paths))
        streams[stream]=entry
    evidence=dict(schema='service-v2-process-evidence-v1',process_started=process_started,
        actual_exit=actual_exit,outcome=outcome,error=result.get('error') if outcome=='completed' else error,streams=streams)
    write_json(attempt/'process-evidence.json',evidence)
    result['process_evidence_path']='process-evidence.json'


def _validate_process_evidence(attempt, result, artifacts):
    """Every final attempt has both streams or bound unavailable/error evidence."""
    if result['status'] not in ('ok','technical_error','interrupted'): raise ValueError('Unknown final process status')
    actual_exit=result['actual_exit']
    if actual_exit is not None and type(actual_exit) is not int: raise ValueError('Invalid direct process outcome')
    if result['status']!='ok' and (not isinstance(result.get('error'),str) or not result['error']):
        raise ValueError('Failed process outcome requires retained error')
    logs={stream:[name for name in artifacts if Path(name).name==f'simulator.{stream}.log'] for stream in ('stdout','stderr')}
    evidence_path=result.get('process_evidence_path')
    if evidence_path is None:
        if any(len(paths)!=1 for paths in logs.values()): raise ValueError('Missing retained simulator log or process evidence')
        if actual_exit is None: raise ValueError('Missing retained process evidence for unavailable direct process outcome')
        return
    if evidence_path not in artifacts: raise ValueError('Unlisted process evidence')
    evidence=strict_json(safe_path(attempt,evidence_path))
    required={'schema','process_started','actual_exit','outcome','error','streams'}
    if not isinstance(evidence,dict) or not required<=evidence.keys(): raise ValueError('Incomplete process evidence')
    if (evidence.get('schema')!='service-v2-process-evidence-v1' or evidence.get('actual_exit')!=actual_exit
            or type(evidence.get('actual_exit')) is not type(actual_exit)):
        raise ValueError('Direct process outcome disagrees with process evidence')
    started=evidence.get('process_started')
    if started is not None and type(started) is not bool: raise ValueError('Invalid process creation evidence')
    expected=('completed' if actual_exit is not None else 'not_started' if started is False
              else 'exit_unavailable' if started is True else 'process_unavailable')
    if evidence.get('outcome')!=expected or (actual_exit is not None and started is not True):
        raise ValueError('Contradictory direct process outcome evidence')
    if expected!='completed' and (not isinstance(evidence.get('error'),str) or not evidence['error']):
        raise ValueError('Unavailable process outcome requires error evidence')
    streams=evidence.get('streams')
    if not isinstance(streams,dict) or set(streams)!={'stdout','stderr'}: raise ValueError('Both simulator streams require process evidence')
    for stream,entry in streams.items():
        if not isinstance(entry,dict): raise ValueError('Invalid simulator stream evidence')
        status=entry.get('status')
        if status not in ('captured','unavailable','error'): raise ValueError('Invalid simulator stream capture evidence')
        path=entry.get('path')
        if path is not None:
            if path not in logs[stream] or entry.get('sha256')!=artifacts[path]: raise ValueError('Simulator log/process evidence mismatch')
            safe_path(attempt,path)
        elif status=='captured' or logs[stream]: raise ValueError('Missing simulator log capture path')
        if status!='captured' and any(not isinstance(entry.get(k),str) or not entry[k] for k in ('reason','error')):
            raise ValueError('Unavailable simulator log requires explicit error evidence')
        if result['status']=='ok' and status!='captured': raise ValueError('Successful process requires both full simulator logs')


def collect_case(protocol_path, cell_id, attempt, output):
    phase=load_phase(protocol_path)
    cells=[c for c in phase['cells'] if c['cell_id']==cell_id]
    if len(cells)!=1 or type(attempt) is not int or attempt not in (0,1): raise ValueError('Unknown cell/attempt')
    cell=cells[0]; output=Path(output).resolve()
    if output.name!=f'attempt-{attempt}' or output.parent.name!=cell_id: raise ValueError('Output must be run-root/cell-id/attempt-N')
    verify_runtime(phase,ROOT)
    if ROOT.resolve()!=Path(phase['working_directory']).resolve() or Path.cwd().resolve()!=ROOT.resolve():
        raise ValueError('Execute from declared frozen source working directory')
    from ..prevention.config import PreventionConfig
    from ..prevention.prediction import LearnedPredictor
    from .prediction import ServicePredictor
    config=PreventionConfig(**cell['config']); package=Path(phase['_path']).parent
    old=LearnedPredictor.load(package/'reference-model.json')
    old.validate_test_scenarios([(cell['layout_seed'],cell['radio_seed'])])
    new=ServicePredictor.load(package/'new-model.json') if phase['new_model'] else None
    if new: new.validate_scenarios([cell])
    if attempt==1:
        prior=output.parent/'attempt-0'
        s=strict_json(prior/'status.json')
        if s['status'] not in ('technical_error','interrupted'): raise ValueError('Retry only after documented infrastructure/validation failure')
        r=strict_json(prior/'result.json')
        if r['cell']!=cell or r['freeze_sha256']!=phase['_freeze_sha256']: raise ValueError('Retry changed frozen cell')
    out=reserve(output); root=out.parent.parent
    _collection_manifest(root,phase)
    started=time.time(); start=time.monotonic()
    publish_status(out,dict(status='planned',actual_exit=None,cell_id=cell_id,attempt=attempt))
    from ..prevention.environment import Ns3PreventionEnv
    from ..prevention.evaluation import run_grid
    exits=[]; environments=[]; processes=[]; capture_errors=[]
    class ObservedEnv(Ns3PreventionEnv):
        def __init__(self, **kwargs):
            super().__init__(**kwargs); environments.append(self)
        def close(self):
            bridge=self.bridge
            if bridge is not None and bridge.process is not None: processes.append(bridge.process)
            try: super().close()
            except BaseException as exc:
                capture_errors.append(f'{type(exc).__name__}: {exc}')
                raise
            finally:
                if bridge is not None and bridge.process is not None:
                    exits.append(bridge.process.returncode)
    factory=make_controller_factory(new,old,ServiceV2Config(),phase['new_model'],phase['method_sha256'])
    timings=[]
    class Timed:
        def __init__(self, controller): self.controller=controller
        def act(self, frame):
            before=time.monotonic(); action,decision=self.controller.act(frame)
            timings.append(dict(step=frame['step'],decision_seconds=time.monotonic()-before))
            return action,decision
    def timed_factory(*args,**kwargs): return Timed(factory(*args,**kwargs))
    result=dict(cell=cell,attempt=attempt,freeze_sha256=phase['_freeze_sha256'],status='technical_error',
                actual_exit=None,error=None,started_unix=started)
    publish_status(out,dict(status='running',actual_exit=None,started_unix=started))
    interrupted=False
    try:
        run_grid(config=config,launcher=phase['runtime']['declared']['launcher'],output=out/'collector',
            policies=[cell['policy']],scenarios=[(cell['layout_seed'],cell['radio_seed'])],
            predictor=None,threshold=.5,routing_delay_s=3.,split=phase['phase'] if phase['phase'] in ('train','calibration') else 'test',
            provenance=dict(freeze_sha256=phase['_freeze_sha256'],cell_id=cell_id),
            env_factory=partial(ObservedEnv,service_observation=True),controller_factory=timed_factory,supported_policies=POLICIES)
        records=strict_json(out/'collector/episodes.json')
        if len(records)!=1 or records[0]['status']!='ok': raise ValueError('Collector failed: '+str(records))
        episode=Path(records[0]['episode_dir']).resolve()
        if not episode.is_relative_to(out): raise ValueError('Episode path escape')
        if exits!=[0]: raise ValueError('Direct simulator exit unavailable/nonzero: '+str(exits))
        row=reconcile_episode(episode,config)
        for name in ('simulator.stdout.log','simulator.stderr.log','config.json','policy_metadata.json'):
            if not (episode/name).is_file(): raise ValueError('Missing terminal provenance/log: '+name)
        result.update(row,status='ok',episode_path=episode.relative_to(out).as_posix(),trace_sha256=sha256(episode/'transitions.jsonl'))
    except BaseException as exc:
        interrupted=isinstance(exc,(KeyboardInterrupt,SystemExit))
        result.update(status='interrupted' if interrupted else 'technical_error',error=f'{type(exc).__name__}: {exc}')
    finally:
        for env in environments:
            try: env.close()
            except Exception as exc: result.update(status='technical_error',error=f'Close failure: {type(exc).__name__}: {exc}')
        result.update(actual_exit=exits[-1] if exits else None,finished_unix=time.time(),wall_seconds=time.monotonic()-start)
        if environments and environments[-1].episode_output_dir is not None:
            episode=Path(environments[-1].episode_output_dir).resolve()
            if not episode.is_relative_to(out): raise ValueError('Episode path escape')
            result.setdefault('episode_path',episode.relative_to(out).as_posix())
        process_started=True if processes else False if not environments else None
        _record_process_evidence(out,result,process_started,capture_errors)
        write_json(out/'timings.json',timings)
        write_json(out/'result.json',result)
        artifacts={p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and 'status-history' not in p.parts and p.name!='status.json'}
        write_json(out/'artifacts.json',artifacts)
        publish_status(out,dict(status=result['status'],actual_exit=result['actual_exit'],validated=result['status']=='ok',
            finished_unix=result['finished_unix'],wall_seconds=result['wall_seconds'],error=result['error'],
            result_sha256=sha256(out/'result.json'),artifacts_sha256=sha256(out/'artifacts.json')))
    if interrupted: raise KeyboardInterrupt(result['error'])
    return result


def validate_collection(protocol_path, run_root):
    phase=load_phase(protocol_path); root=Path(run_root).resolve()
    manifest=strict_json(root/'collection.json')
    expected=dict(schema='service-v2-collection-v1',phase=phase['phase'],protocol=phase.get('protocol_path_at_freeze',phase['_path']),
        freeze_sha256=phase['_freeze_sha256'],cells=[c['cell_id'] for c in phase['cells']])
    if manifest!=expected: raise ValueError('Whole collection manifest mismatch')
    names=set(expected['cells'])
    if any(p.name not in names for p in root.iterdir() if p.is_dir()): raise ValueError('Unlisted cell directory')
    rows=[]; listed_traces=set()
    for cell in phase['cells']:
        directory=root/cell['cell_id']; attempts=[]
        if directory.exists():
            if any(p.name not in ('attempt-0','attempt-1') for p in directory.iterdir()): raise ValueError('Unlisted cell attempt/file')
            for number in (0,1):
                a=directory/f'attempt-{number}'
                if not a.exists(): continue
                if number==1 and (not attempts or attempts[0]['status']=='ok'): raise ValueError('Invalid retry')
                status=strict_json(a/'status.json')
                history=sorted((a/'status-history').glob('*.json'))
                if not history or strict_json(history[-1])!=status or [p.stem for p in history]!=[f'{i:06d}' for i in range(1,len(history)+1)]: raise ValueError('Status revision history')
                for index,h in enumerate(history):
                    if strict_json(h).get('previous_revision_sha256')!=(sha256(history[index-1]) if index else None): raise ValueError('Status revision hash chain')
                if status['status'] in ('planned','running'): raise ValueError('Incomplete running attempt')
                result=strict_json(a/'result.json'); artifacts=strict_json(a/'artifacts.json')
                if status['result_sha256']!=sha256(a/'result.json') or status['artifacts_sha256']!=sha256(a/'artifacts.json'): raise ValueError('Attempt manifest mismatch')
                actual={p.relative_to(a).as_posix() for p in a.rglob('*') if p.is_file() and 'status-history' not in p.parts and p.name not in ('status.json','artifacts.json')}
                if actual!=set(artifacts): raise ValueError('Unlisted/missing attempt data')
                for name,digest in artifacts.items():
                    if sha256(safe_path(a,name))!=digest: raise ValueError('Attempt data hash mismatch')
                    if name.endswith('/transitions.jsonl'): listed_traces.add(safe_path(a,name))
                if (result['cell']!=cell or result['attempt']!=number or result['freeze_sha256']!=phase['_freeze_sha256']
                        or result['status']!=status['status'] or result['actual_exit']!=status['actual_exit']): raise ValueError('Attempt cell/source mismatch')
                _validate_process_evidence(a,result,artifacts)
                if result['status']=='ok':
                    if result['actual_exit']!=0 or status.get('validated') is not True: raise ValueError('Nonzero/incomplete success')
                    from ..prevention.config import PreventionConfig
                    episode=safe_path(a,result['episode_path']); config=PreventionConfig(**cell['config'])
                    saved=strict_json(episode/'config.json'); metadata=strict_json(episode/'policy_metadata.json')
                    if (saved['config']!=cell['config'] or saved['layout_seed']!=cell['layout_seed'] or saved['radio_seed']!=cell['radio_seed']
                            or metadata['collection_policy']!=cell['policy'] or metadata['layout_seed']!=cell['layout_seed'] or metadata['radio_seed']!=cell['radio_seed']): raise ValueError('Trace config/policy/seeds mismatch')
                    if metadata['split']!=(phase['phase'] if phase['phase'] in ('train','calibration') else 'test'): raise ValueError('Legacy split mismatch')
                    for name in ('simulator.stdout.log','simulator.stderr.log'):
                        if not (episode/name).is_file(): raise ValueError('Missing retained simulator log')
                    measured=reconcile_episode(episode,config)
                    if any(result.get(k)!=v for k,v in measured.items()) or result['trace_sha256']!=sha256(episode/'transitions.jsonl'): raise ValueError('Stored result disagrees with trace')
                attempts.append(result)
        selected=next((r for r in reversed(attempts) if r['status']=='ok'),None)
        rows.append(dict(cell,**(selected or dict(status='missing' if not attempts else attempts[-1]['status'],actual_udp_pdr=None,total_relay_travel_m=None)),attempt_records=attempts))
    if {p.resolve() for p in root.rglob('transitions.jsonl')}!=listed_traces: raise ValueError('Unlisted trace')
    return dict(schema='service-v2-validation-v1',phase=phase['phase'],protocol=phase['_path'],freeze_sha256=phase['_freeze_sha256'],
        complete=all(r['status']=='ok' for r in rows),planned_cells=len(rows),ok_cells=sum(r['status']=='ok' for r in rows),rows=rows)


def load_service_episodes(run_root, expected_phase, policies=None):
    from .prediction import ServiceEpisode
    from ..prevention.config import PreventionConfig
    root=Path(run_root).resolve(); manifest=strict_json(root/'collection.json')
    report=validate_collection(manifest['protocol'],root)
    if report['phase']!=expected_phase or not report['complete']: raise ValueError('Incomplete/wrong phase collection; survivor selection forbidden')
    phase=load_phase(manifest['protocol']); source=strict_json(Path(phase['_path']).parent/'source-manifest.json')
    hashes={k:v['executed_sha256'] for k,v in source.items()}
    hashes['runtime/launcher']=phase['runtime']['launcher_sha256']
    if policies is not None and (not policies or len(set(policies))!=len(policies) or set(policies)-set(phase['policies'])): raise ValueError('Invalid policy selection')
    episodes=[]
    for row in report['rows']:
        if policies is not None and row['policy'] not in policies: continue
        a=root/row['cell_id']/f"attempt-{row['attempt']}"; episode=safe_path(a,row['episode_path'])
        config=PreventionConfig(**row['config'])
        episodes.append(ServiceEpisode(row['cell_id'],expected_phase,row['policy'],row['regime'],row['context'],
            row['layout_seed'],row['radio_seed'],config,read_frames(episode,config),hashes,row['trace_sha256']))
    return episodes
