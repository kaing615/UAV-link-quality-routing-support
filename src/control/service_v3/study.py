"""v3 arms on the unchanged v2 collector, trace loading and the tuning selection step."""
from pathlib import Path

from ..service_v2.config import ServiceV2Config
from ..service_v2.protocol import reserve, safe_path, sha256, strict_json, strict_loads, write_json
from ..service_v2.study import (_CollectorCompatibleV2, collect_case, make_controller_factory, read_frames,
                                validate_collection)
from . import protocol as v3


def make_factory(phase, new, old):
    from ..service_v2.controller import ServiceV2Controller, threshold_method
    from .actionvalue import GainGate
    base = make_controller_factory(new, old, ServiceV2Config(), phase['new_model'], phase['method_sha256'])
    package = Path(phase['_path']).parent
    selection = strict_json(package/'selection.json') if phase['selection_sha256'] else None

    def factory(policy, config, **kwargs):
        if not policy.startswith('service-v3-'):
            return base(policy, config, **kwargs)
        if selection is None: raise ValueError('v3 arm without a frozen tuning selection')
        method = threshold_method(selection['risk_on'], selection['risk_off'])
        gate = GainGate(selection) if policy == 'service-v3-gate' else None
        # Logged role stays 'service-v2' (same controller); the cell policy identifies the arm.
        return _CollectorCompatibleV2(ServiceV2Controller('service-v2', config, new, method, phase['new_model'],
                                                          phase['selection_sha256'], action_gate=gate))
    return factory


def run_case(protocol_path, cell_id, attempt, output):
    return collect_case(protocol_path, cell_id, attempt, output, loader=v3.load, factory_builder=make_factory)


def load_traces(protocol_path, run_root, loader=v3.load):
    """Complete collection only (no survivor selection): rows with frames and decisions."""
    from ..prevention.config import PreventionConfig
    phase = loader(protocol_path)
    report = validate_collection(protocol_path, run_root, loader=loader)
    if not report['complete']: raise ValueError('Incomplete collection; survivor selection forbidden')
    root = Path(run_root).resolve(); rows = []
    for row in report['rows']:
        attempt = root/row['cell_id']/f"attempt-{row['attempt']}"
        config = PreventionConfig(**row['config'])
        decisions = [strict_loads(line) for line in next((attempt/'collector/episodes').rglob('decisions.jsonl'))
                     .read_text(encoding='utf-8').splitlines()]
        if [d['step'] for d in decisions] != list(range(config.steps)): raise ValueError('Full decision trace required')
        episode_dir = safe_path(attempt, row['episode_path'])
        rows.append(dict(row, frames=read_frames(episode_dir, config), decisions=decisions, episode_dir=str(episode_dir)))
    return phase, report, rows


def episode(row, phase_name):
    from ..prevention.config import PreventionConfig
    from ..service_v2.prediction import ServiceEpisode
    return ServiceEpisode(row['cell_id'], phase_name, row['policy'], row['regime'], row['context'], row['layout_seed'],
                          row['radio_seed'], PreventionConfig(**row['config']), row['frames'], {}, row['trace_sha256'])


def scenario_key(row):
    return row['regime'], row['context'], row['layout_seed'], row['radio_seed']


def labelled_bursts(rows, policies):
    from .actionvalue import burst_labels
    hold = {scenario_key(r): r for r in rows if r['policy'] == 'hold'}
    labels, excluded = [], {}
    for r in rows:
        if r['policy'] not in policies: continue
        found, skipped = burst_labels(r['frames'], r['decisions'], hold[scenario_key(r)]['frames'],
                                      (r['regime'], r['layout_seed'], r['radio_seed']))
        for item in found: item.update(policy=r['policy'], context=r['context'])
        labels += found
        for k, v in skipped.items(): excluded[k] = excluded.get(k, 0) + v
    return labels, excluded


def select(protocol_path, run_root, output):
    from ..service_v2.analysis import forecast_scores
    from ..service_v2.prediction import ServicePredictor
    from .actionvalue import build_selection
    phase, _, rows = load_traces(protocol_path, run_root)
    if phase['phase'] != 'tuning': raise ValueError('Selection uses tuning only')
    if Path(output).exists(): raise FileExistsError(str(output))
    model = Path(phase['_path']).parent/'new-model.json'
    scores = forecast_scores([episode(r, 'tuning') for r in rows if r['policy'] == 'hold'], ServicePredictor.load(model))
    labels, excluded = labelled_bursts(rows, ('service-v2',))
    selection = build_selection(scores, labels, excluded, tuning_freeze_sha256=phase['_freeze_sha256'],
                                predictor_sha256=sha256(model))
    write_json(output, selection)
    return selection
