"""Exact horizon accounting and complete paired-grid admission."""
import json

import pytest

from src.control.prevention.config import PreventionConfig
from src.control.prevention.evaluation import (
    validate_paired_grid, episode_metrics, require_empty_output, run_grid, reliability_metrics,
)


def record(policy='hold', layout=1, status='ok'):
    return dict(policy=policy, layout_seed=layout, radio_seed=layout+100, status=status)


def test_paired_grid_rejects_missing_duplicate_and_errors():
    rows = [record(), record('geometry')]
    validate_paired_grid(rows, ['hold', 'geometry'], [(1, 101)])
    for bad in [rows[:1], rows+rows[:1], [record(status='technical_error'), rows[1]]]:
        with pytest.raises(ValueError): validate_paired_grid(bad, ['hold', 'geometry'], [(1, 101)])


def test_full_horizon_metrics_exclude_anchors_and_read_simulator_pdr(tmp_path):
    frames = [dict(step=t, components=1 if t in (0, 2) else 2,
                   source_dest_connected=t in (0, 2), olsr_route_usable=t == 0,
                   safety_rejected=t == 2, travel=[100*t, 2*t, 100*t]) for t in range(4)]
    (tmp_path/'closed_loop_metrics.csv').write_text('pdr,rx_packets,mean_delay_ms\n0.25,0,0\n')
    c = PreventionConfig(num_uavs=3, destination=2, steps=3)
    m = episode_metrics(frames, c, tmp_path, [{'intervention_started': True}])
    assert m['disconnected_fraction'] == pytest.approx(2/3)
    assert m['partition_count'] == 2 and m['first_partition_step'] == 1
    assert m['source_dest_disconnected_fraction'] == pytest.approx(2/3)
    assert m['olsr_unusable_fraction'] == 1
    assert m['total_relay_travel'] == m['max_relay_travel'] == 6
    assert m['pdr'] == .25 and m['mean_delay_ms'] is None
    with pytest.raises(ValueError, match='horizon'): episode_metrics(frames[:-1], c, tmp_path, [])


def test_no_overwrite_before_any_launch(tmp_path):
    output = tmp_path/'existing'; output.mkdir(); (output/'sentinel').write_text('keep')
    calls = []
    with pytest.raises(ValueError, match='empty|new'):
        run_grid(config=PreventionConfig(), launcher=['never'], output=output,
                 policies=['hold'], scenarios=[(1, 101)], env_factory=lambda **kw: calls.append(kw))
    assert calls == [] and (output/'sentinel').read_text() == 'keep'
    path = tmp_path/'file'; path.write_text('keep')
    with pytest.raises(ValueError): require_empty_output(path)


def test_reliability_exact_and_empty_bin_null():
    result = reliability_metrics([0, 1], [.25, .75], threshold=.5, bins=5)
    assert result['brier'] == .0625 and result['recall'] == 1 and result['false_positive_rate'] == 0
    assert result['bins'][0]['observed_fraction'] is None


def test_cli_seed_grids_explicit_and_output_refusal(tmp_path):
    from scripts.analysis.evaluate_prevention import parse_seed_spec, main
    from scripts.train.train_prevention_predictor import main as train_main
    assert parse_seed_spec('11001:11003,11008') == [11001, 11002, 11003, 11008]
    for bad in ('1,1', '4:2', '-1', '1.5', ''):
        with pytest.raises(ValueError): parse_seed_spec(bad)
    output = tmp_path/'evidence'; output.mkdir(); (output/'sentinel').write_text('preserve')
    with pytest.raises(ValueError, match='empty|new'):
        main(['--output', str(output), '--split', 'train', '--policies', 'hold', '--layout-seeds', '1'])
    with pytest.raises(ValueError, match='empty|new'):
        train_main(['--output', str(output), '--train-root', 'missing', '--calibration-root', 'missing'])
    assert (output/'sentinel').read_text() == 'preserve'


def test_collection_error_persisted_and_no_complete_comparison(tmp_path):
    class Broken:
        episode_output_dir = None
        def __init__(self, **kw): pass
        def reset(self, **kw): raise RuntimeError('fixed seed setup failed')
        def close(self): pass
    summary = run_grid(config=PreventionConfig(), launcher=['unused'], output=tmp_path/'run',
                       policies=['hold', 'geometry'], scenarios=[(1, 101)], env_factory=Broken)
    assert not summary['paired_valid'] and summary['comparison'] is None
    rows = json.loads((tmp_path/'run/episodes.json').read_text())
    assert len(rows) == 2 and all(r['status'] == 'setup_error' for r in rows)


def test_offline_reliability_stratifies_behavior_and_refuses_overlap():
    from src.control.prevention.evaluation import evaluate_predictor
    from src.control.prevention.prediction import EpisodeTrace
    class Predictor:
        def validate_test_scenarios(self, scenarios):
            if any(layout == 1 for layout, radio in scenarios): raise ValueError('overlap')
        def predict(self, current, previous, comm_range):
            return {h: {(0, 1): .25} for h in (3, 5, 10, 20)}
    frames = [dict(step=t, positions=[[0, 0, 0], [1, 0, 0]], rx_ratio=[[0, 1], [1, 0]]) for t in range(22)]
    hold = EpisodeTrace('hold', 2, 102, 240, frames, {}, collection_policy='hold')
    reactive = EpisodeTrace('reactive', 2, 102, 240, frames, {}, collection_policy='reactive')
    result = evaluate_predictor(Predictor(), [hold, reactive])
    assert set(result['by_collection_policy']) == {'hold', 'reactive'}
    hold.layout_seed = 1
    with pytest.raises(ValueError, match='overlap'): evaluate_predictor(Predictor(), [hold])


def test_paired_runner_binds_simple_and_learned_independently(tmp_path, monkeypatch):
    import numpy as np
    from src.control.prevention import evaluation
    from src.control.prevention.prediction import SimplePredictor
    from src.control.prevention.controller import PreventionController
    bindings = []
    class Predictor:
        artifact = {'comm_range': 240}
        def validate_test_scenarios(self, scenarios): pass
        def predict(self, current, previous, comm_range):
            return {h: {(0, 1): 1., (1, 2): 1.} for h in (3, 5, 10, 20)}
    learned = Predictor()
    class RecordingController(PreventionController):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw); bindings.append((self.policy, self.predictor))
    class Env:
        def __init__(self, config, output_root, **kw):
            self.config = config; self.episode_output_dir = output_root/'episode'
            self.episode_output_dir.mkdir()
        def frame(self, step):
            return dict(step=step, positions=[[0, 100, 100], [100, 200, 100], [200, 100, 100]],
                        velocities=[[0, 0, 0]]*3, travel=[0, 0, 0], components=1,
                        rx_ratio=[[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                        rssi=[[0]*3 for _ in range(3)], rssi_mask=[[False]*3 for _ in range(3)])
        def reset(self, **kw): self.last_frame = self.frame(0)
        def step(self, action):
            self.last_frame = self.frame(1)
            return None, 0, False, True, {}
        def close(self):
            (self.episode_output_dir/'transitions.jsonl').write_text('{}')
    monkeypatch.setattr(evaluation, 'PreventionController', RecordingController)
    monkeypatch.setattr(evaluation, 'episode_metrics', lambda *a: {key: 0 for key in (
        'disconnected_fraction', 'source_dest_disconnected_fraction', 'olsr_unusable_fraction',
        'partition_count', 'pdr', 'mean_delay_ms', 'total_relay_travel', 'max_relay_travel',
        'rejected_actions', 'intervention_starts')})
    policies = ['predictive-simple', 'predictive-learned', 'learned-no-criticality', 'learned-no-deadline']
    summary = run_grid(config=PreventionConfig(num_uavs=3, destination=2, steps=1), launcher=['unused'],
                       output=tmp_path/'run', policies=policies, scenarios=[(1, 101)], predictor=learned, env_factory=Env)
    assert summary['paired_valid']
    assert all(isinstance(p, SimplePredictor) if name == 'predictive-simple' else p is learned for name, p in bindings)
