from copy import deepcopy
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from src.control.service_v2.features import service_features
from src.control.service_v2.prediction import (future_decline_label, build_service_samples,
    validate_partitions, fit_service_predictor, ServicePredictor)
from tests.service_v2_support import (fixture_config, service_frames, episode_from_frames,
    fail_route, synthetic_partition, set_window_counts)


def test_hand_calculated_ordered_features():
    fs = service_frames([60] * 7)
    want = [1, 1, 0, 1, 0, 1, 1/3, 0, 0, .45, 0, .15,
            .45, 0, 1, -.7, 0, 2/3, 1, 0, 1, 1, 1, 1, 1]
    np.testing.assert_allclose(service_features(fs, fixture_config()), want, atol=1e-12)


def test_hand_calculated_features_with_motion_relay_order_and_measured_route():
    fs = service_frames([60, 60, 60, 60, 60, 30, 90])
    set_window_counts(fs, source=[60, 60, 60, 60, 30, 60, 60])
    for f in fs[-2:]:
        f['service_observation']['route']['nodes'] = [0, 1, 3]
        f['service_observation']['route_probe_hops'] = [
            dict(**{'from': a, 'to': b}, rx_ratio_forward=forward, rx_ratio_reverse=reverse)
            for a, b, forward, reverse in [(0, 1, .8, .7), (1, 3, .6, .9)]]
    f = fs[-1]
    f['positions'] = [[300., 600., 100.], [360., 620., 100.], [340., 650., 100.], [420., 600., 100.]]
    f['velocities'] = [[0., -1., 0.], [.6, .8, 0.], [0., 0., 2.], [2., 1., 0.]]
    f['travel'] = [0., 100., 300., 0.]
    f['rssi'][0][1], f['rssi'][1][0] = -50., -70.
    f['rssi'][1][3], f['rssi'][3][1] = -80., -100.
    f['rx_ratio'] = [[0.] * 4 for _ in range(4)]
    for a, b, forward, reverse in [(0, 1, .8, .7), (1, 3, .6, .9)]:
        f['rx_ratio'][a][b], f['rx_ratio'][b][a] = forward, reverse
    want = [1.5, 1., .1, .9, 0., 1., 2/3, 0., .25, .5, .25, np.sqrt(4100)/240,
            np.sqrt(4000)/240, 80/np.sqrt(4000)/8, .6, -.9, -2., 2/3, .25, 1.5/8,
            1., 1., 1., 1., 1.]
    np.testing.assert_allclose(service_features(fs, fixture_config()), want, atol=1e-12)


def test_rssi_masks_clipping_and_same_route_delta():
    fs = service_frames([60] * 7)
    fs[-1]['rssi'][0][3], fs[-1]['rssi'][3][0] = -200., 0.
    x = service_features(fs, fixture_config())
    assert x[15] == -.7  # mean(-120,-20)/100
    fs[-1]['rssi_mask'][3][0] = False
    x = service_features(fs, fixture_config())
    assert x[15] == -1.2 and x[16] == -5.  # (-120 - -70)/10
    assert x[22] == 1 and x[23] == 1
    fs[-2]['service_observation']['route']['nodes'] = [0, 1, 3]
    fs[-2]['service_observation']['route_probe_hops'] = [
        dict(**{'from': a, 'to': b}, rx_ratio_forward=1., rx_ratio_reverse=1.) for a, b in [(0, 1), (1, 3)]]
    x = service_features(fs, fixture_config())
    assert x[16] == 0 and x[23] == 0
    assert x[8] == .5  # the penultimate route change and return
    fs[-1]['rssi_mask'][0][3] = False
    x = service_features(fs, fixture_config())
    assert x[15] == 0 and x[22] == 0
    fail_route(fs[-1])
    x = service_features(fs, fixture_config())
    assert list(x[12:17]) == [0] * 5 and x[21] == 0


def test_features_ignore_future_and_mission_metadata_and_keep_endpoint_connectivity():
    fs = service_frames([60] * 20)
    before = service_features(fs[:8], fixture_config())
    fs[8]['positions'][0][0] = 9999
    changed = replace(fixture_config(), mission_start_step=999, mission_speed=7)
    np.testing.assert_equal(service_features(fs[:8], changed), before)
    fs[7]['rx_ratio'][1] = [0.] * 4
    for row in fs[7]['rx_ratio']:
        row[1] = 0.
    assert service_features(fs[:8], changed)[20] == 1


def test_feature_history_masks_and_invalid_geometry():
    fs = service_frames([60] * 5)
    x = service_features(fs, fixture_config())
    assert x[24] == 0
    fs[-1]['positions'][0][0] = float('nan')
    with pytest.raises(ValueError, match='finite|geometry'):
        service_features(fs, fixture_config())


def test_future_only_target_and_uniform_censoring():
    ep = episode_from_frames(service_frames([60] * 7 + [0] * 20))
    label = future_decline_label(ep, 7, 10)
    assert (label.value, label.target_confirmation_step, label.ordinary_onset_step) == (1, 14, 8)
    assert future_decline_label(replace(ep, frames=ep.frames[:17]), 7, 10).value is None
    assert future_decline_label(ep, 8, 10).value is None
    invalid = deepcopy(ep)
    invalid.frames[17]['service_observation']['window']['offered_packets'] = 29
    assert future_decline_label(invalid, 7, 10).value is None


@pytest.mark.parametrize('zero_start,expected', [(15, 1), (16, 0)])
def test_confirmation_exactly_at_horizon_boundary(zero_start, expected):
    ep = episode_from_frames(service_frames([60] * (zero_start - 1) + [0] * 30))
    label = future_decline_label(ep, 7, 10)
    assert label.value == expected
    assert label.target_confirmation_step == (17 if expected else None)


def test_route_failure_alone_is_not_service_target():
    ep = episode_from_frames(service_frames([60] * 30))
    for f in ep.frames[8:]:
        fail_route(f)
    assert future_decline_label(ep, 7, 20).value == 0


def test_rows_equal_trajectory_weights_and_keep_eligibility_counts():
    first = episode_from_frames(service_frames([60] * 30))
    second = replace(episode_from_frames(service_frames([60] * 40)), identity='other')
    empty = replace(first, identity='never-armed', frames=service_frames([0] * 40))
    rows = build_service_samples([first, second, empty], 10)
    assert rows.x.shape == (38, 25)  # origins7..20 and7..30
    assert rows.origin_step.tolist() == list(range(7, 21)) + list(range(7, 31))
    assert np.sum(rows.weight[np.asarray(rows.episode_id) == first.identity]) == pytest.approx(1.)
    assert np.sum(rows.weight[np.asarray(rows.episode_id) == second.identity]) == pytest.approx(1.)
    assert rows.counts['zero_eligible_episodes'] == ['never-armed']
    assert rows.counts['right_censored'] == 20
    assert rows.counts['negative_rows'] == 38


def test_radio_collision_is_not_fixed_by_different_layout():
    first = episode_from_frames(service_frames([60] * 30), layout_seed=11, radio_seed=22)
    second = replace(first, identity='other', phase='calibration', layout_seed=33)
    with pytest.raises(ValueError, match='radio_seed'):
        validate_partitions({'train': [first], 'calibration': [second]})


@pytest.mark.parametrize('field,value', [('layout_seed', 11), ('identity', 'fixture')])
def test_layout_and_episode_collisions(field, value):
    first = episode_from_frames(service_frames([60] * 30))
    second = replace(first, identity='other', phase='calibration', layout_seed=33, radio_seed=44)
    second = replace(second, **{field: value})
    with pytest.raises(ValueError, match=field):
        validate_partitions({'train': [first], 'calibration': [second]})


def test_phase_and_within_split_pairing_are_validated():
    ep = episode_from_frames(service_frames([60] * 30))
    with pytest.raises(ValueError, match='phase'):
        validate_partitions({'calibration': [ep]})
    with pytest.raises(ValueError, match='Duplicate'):
        validate_partitions({'train': [ep, replace(ep, identity='duplicate-cell')]})
    with pytest.raises(ValueError, match='seed'):
        validate_partitions({'train': [ep, replace(ep, identity='unpaired', radio_seed=99)]})


@pytest.mark.parametrize('phase', ['pilot', 'formal'])
def test_pilot_and_formal_refuse_seed_pair_reuse_across_contexts(phase):
    first = episode_from_frames(service_frames([60] * 30), phase=phase, context='static')
    second = replace(first, identity='gradual-cell', context='gradual')
    with pytest.raises(ValueError, match='context|seed'):
        validate_partitions({phase: [first, second]})


@pytest.mark.parametrize('phase', ['train', 'calibration', 'development'])
def test_collection_phases_allow_context_and_policy_pairing(phase):
    first = episode_from_frames(service_frames([60] * 30), phase=phase)
    cells = [replace(first, identity=f'{phase}-{context}-{policy}', context=context, policy=policy)
             for context in ('static', 'gradual') for policy in ('hold', 'geometry')]
    validate_partitions({phase: cells})


@pytest.mark.parametrize('phase', ['pilot', 'formal'])
def test_pilot_and_formal_allow_policy_pairing_with_distinct_context_seeds(phase):
    first = episode_from_frames(service_frames([60] * 30), phase=phase, context='static')
    cells = [first, replace(first, identity='static-geometry', policy='geometry'),
        replace(first, identity='gradual-hold', context='gradual', layout_seed=33, radio_seed=44),
        replace(first, identity='gradual-geometry', context='gradual', policy='geometry', layout_seed=33, radio_seed=44)]
    validate_partitions({phase: cells})


@pytest.fixture(scope='module')
def fit_inputs():
    return synthetic_partition('train', 6), synthetic_partition('calibration', 3)


def test_fit_class_deficiency_and_hold_only_calibration(fit_inputs):
    train, calibration = fit_inputs
    with pytest.raises(ValueError, match='train.*counts'):
        fit_service_predictor(train[:4], calibration)
    with pytest.raises(ValueError, match='calibration.*counts'):
        fit_service_predictor(train, calibration[:8])
    healthy_hold = [replace(e, frames=service_frames([60] * 115)) if e.policy == 'hold' else e for e in calibration]
    with pytest.raises(ValueError, match='calibration.*counts'):
        fit_service_predictor(train, healthy_hold)


def test_fit_refuses_fewer_than_fifty_positive_rows_even_with_six_groups(fit_inputs):
    _, calibration = fit_inputs
    train = [episode_from_frames(service_frames([60] * 7 + [0] * 25),
        identity=f'fixture-short-{i}', layout_seed=10+i, radio_seed=30+i) for i in range(6)]
    with pytest.raises(ValueError, match='train.*counts'):
        fit_service_predictor(train, calibration)


def test_real_fit_produces_models(fit_inputs):
    predictor = fit_service_predictor(*fit_inputs)
    assert 'models' in predictor.artifact


def test_empty_artifact_is_not_a_predictor():
    with pytest.raises(ValueError, match='schema'):
        ServicePredictor({})


@pytest.fixture(scope='module')
def fitted(fit_inputs):
    return fit_service_predictor(*fit_inputs)


def test_finite_reproducible_fit_nested_risks_and_training_only_normalization(fit_inputs, fitted):
    again = fit_service_predictor(*fit_inputs)
    assert fitted.artifact == again.artifact
    for h in ('10', '20'):
        model = fitted.artifact['models'][h]
        assert np.isfinite(model['weights']).all()
        assert np.linalg.norm(model['weights']) > 0
        assert model['train_fit']['converged'] and model['calibration_fit']['converged']
        rows = build_service_samples(fit_inputs[0], int(h))
        expected = np.average(rows.x, axis=0, weights=rows.weight)
        np.testing.assert_allclose(model['mean'], expected)
    for arrivals in ([60] * 7, [60] * 30 + [0] * 3):
        risks = fitted.predict(service_frames(arrivals), fixture_config())
        assert set(risks) == {10, 20}
        assert 0 <= risks[10] <= risks[20] <= 1
    frames = service_frames([60] * 7)
    original = fitted.predict(frames, fixture_config())
    frames[-1]['future_mission'] = {'start': -999, 'speed': 10000}
    assert fitted.predict(frames, replace(fixture_config(), mission_start_step=999, mission_speed=7.)) == original
    diagnostics = fitted.artifact['calibration_diagnostics']
    assert {row['policy'] for row in diagnostics['rows']} == {'hold', 'geometry'}
    assert all({'raw_logit', 'mapped_risk', 'projected_risk'} <= set(row['predictions']['10'])
               for row in diagnostics['rows'])
    assert all(e['policy'] == 'hold' for e in fitted.artifact['splits']['calibration']
               if e['identity'] in fitted.artifact['models']['10']['calibration_counts']['per_episode'])


@pytest.mark.parametrize('mutation', ['schema', 'names', 'range', 'horizons', 'nonfinite',
    'scale', 'transforms', 'trace', 'overlap', 'solver', 'counts', 'missing'])
def test_strict_artifact_refuses_contract_and_provenance_changes(fitted, mutation):
    a = fitted.artifact
    if mutation == 'schema': a['schema'] = 'old'
    elif mutation == 'names': a['feature_names'][0:2] = reversed(a['feature_names'][0:2])
    elif mutation == 'range': a['comm_range'] = 243
    elif mutation == 'horizons': a['horizons'] = [10]
    elif mutation == 'nonfinite': a['models']['10']['weights'][0] = float('nan')
    elif mutation == 'scale': a['models']['10']['scale'][0] = 0
    elif mutation == 'transforms': a['feature_transforms'][0] = 'different'
    elif mutation == 'trace': a['splits']['train'][0]['trace_sha256'] = 'unknown'
    elif mutation == 'overlap': a['splits']['calibration'][0]['radio_seed'] = a['splits']['train'][0]['radio_seed']
    elif mutation == 'solver': a['solver']['max_iterations'] = 99
    elif mutation == 'counts': a['models']['10']['calibration_counts']['positive_rows'] = 19
    else: del a['target']
    with pytest.raises(ValueError):
        ServicePredictor(a)


def test_exclusive_save_strict_load_scenario_overlap_and_defensive_artifact(fitted, tmp_path):
    path = tmp_path / 'predictor.json'
    fitted.save_exclusive(path)
    assert ServicePredictor.load(path).artifact == fitted.artifact
    with pytest.raises(ValueError, match='exists'):
        fitted.save_exclusive(path)
    before = path.read_bytes()
    a = fitted.artifact
    a['models']['10']['weights'][0] = 999
    assert path.read_bytes() == before and fitted.artifact['models']['10']['weights'][0] != 999
    fitted.validate_scenarios([dict(layout_seed=888, radio_seed=999, config=asdict(fixture_config()))])
    with pytest.raises(ValueError, match='radio_seed'):
        fitted.validate_scenarios([dict(layout_seed=888, radio_seed=200, config=asdict(fixture_config()))])
    with pytest.raises(ValueError, match='range|config'):
        fitted.predict(service_frames([60] * 7), replace(fixture_config(), comm_range=243))
    bad = tmp_path / 'invalid.json'
    bad.write_text('{"schema": NaN}', encoding='utf-8')
    with pytest.raises(ValueError):
        ServicePredictor.load(bad)
    bad.write_text('{"schema": "a", "schema": "b"}', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate'):
        ServicePredictor.load(bad)


def test_real_solver_nonconvergence_is_not_silently_accepted():
    from src.control.service_v2.prediction import _fit_logistic
    # Real Newton iterations, limited to one to exercise the exhausted gate.
    x = np.asarray([[-2.], [-1.], [1.], [2.]])
    with pytest.raises(ValueError, match='converge'):
        _fit_logistic(x, np.asarray([0., 0., 1., 1.]), np.ones(4), .01, max_iterations=1)


@pytest.mark.parametrize('phase', ['train', 'calibration'])
def test_twenty_second_class_gate_is_independent_of_ten_second_gate(fit_inputs, phase):
    train, calibration = fit_inputs
    original = train if phase == 'train' else calibration
    # Initial eligible origins are all positive. The recovered block has
    # six complete10s negative origins, but no complete20s negative origins.
    changed = [replace(e, frames=service_frames([60] * 13 + [0] * 20 + [60] * 22)) for e in original]
    ten = build_service_samples(changed if phase == 'train' else [e for e in changed if e.policy == 'hold'], 10)
    assert ten.counts['positive_rows'] >= (50 if phase == 'train' else 20)
    assert ten.counts['negative_rows'] >= (50 if phase == 'train' else 20)
    with pytest.raises(ValueError, match=phase + '.*"horizon": 20'):
        fit_service_predictor(changed if phase == 'train' else train,
                              changed if phase == 'calibration' else calibration)


@pytest.mark.parametrize('value', ['1.0', True])
def test_feature_geometry_refuses_coercible_non_numbers(value):
    fs = service_frames([60] * 7)
    fs[-1]['positions'][0][0] = value
    with pytest.raises(ValueError, match='geometry|numeric'):
        service_features(fs, fixture_config())


@pytest.mark.parametrize('key,value', [('weights', '0.0'), ('scale', True), ('calibration', '0.0')])
def test_artifact_refuses_coercible_non_numeric_coefficients(fitted, key, value):
    a = fitted.artifact
    a['models']['10'][key][0] = value
    with pytest.raises(ValueError, match='numeric|vector|calibration'):
        ServicePredictor(a)


@pytest.mark.parametrize('horizon', ['10', '20'])
def test_loader_refuses_hold_calibration_counts_contradicting_stored_labels(fitted, tmp_path, horizon):
    artifact = fitted.artifact
    counts = artifact['models'][horizon]['calibration_counts']
    for ec in counts['per_episode'].values():
        assert ec['negative_rows'] > 1 and ec['positive_rows'] > 0
        ec['positive_rows'] += 1
        ec['negative_rows'] -= 1
    counts['positive_rows'] += counts['episodes']
    counts['negative_rows'] -= counts['episodes']
    path = tmp_path / 'contradictory-hold-counts.json'
    path.write_text(json.dumps(artifact, allow_nan=False), encoding='utf-8')
    with pytest.raises(ValueError, match='calibration.*counts|HOLD.*counts'):
        ServicePredictor.load(path)


@pytest.mark.parametrize('policy', ['hold', 'geometry'])
@pytest.mark.parametrize('horizon', ['10', '20'])
def test_loader_refuses_projected_metric_row_count_mismatch(fitted, tmp_path, policy, horizon):
    artifact = fitted.artifact
    artifact['calibration_diagnostics']['metrics_after_projection'][f'{policy}-{horizon}']['rows'] = 999999
    path = tmp_path / 'contradictory-metric-count.json'
    path.write_text(json.dumps(artifact, allow_nan=False), encoding='utf-8')
    with pytest.raises(ValueError, match='metric'):
        ServicePredictor.load(path)


@pytest.mark.parametrize('policy', ['hold', 'geometry'])
@pytest.mark.parametrize('horizon', ['10', '20'])
@pytest.mark.parametrize('metric', ['brier', 'log_loss', 'prevalence'])
def test_loader_refuses_projected_weighted_metric_value_mismatch(fitted, tmp_path, policy, horizon, metric):
    artifact = fitted.artifact
    metrics = artifact['calibration_diagnostics']['metrics_after_projection'][f'{policy}-{horizon}']
    original = metrics[metric]
    metrics[metric] = original + .01 if original < .98 or metric == 'log_loss' else original - .01
    path = tmp_path / 'contradictory-metric-value.json'
    path.write_text(json.dumps(artifact, allow_nan=False), encoding='utf-8')
    with pytest.raises(ValueError, match='metric'):
        ServicePredictor.load(path)


def test_loader_uses_equal_trajectory_metric_weights_with_unequal_calibration_lengths(fit_inputs, tmp_path):
    train, calibration = fit_inputs
    changed = [replace(e, frames=e.frames[:90]) if e.context == 'static' else e for e in calibration]
    artifact = fit_service_predictor(train, changed).artifact
    diagnostics = artifact['calibration_diagnostics']
    for policy in ('hold', 'geometry'):
        for horizon in ('10', '20'):
            per_episode = {}
            for row in diagnostics['rows']:
                if row['policy'] == policy and horizon in row['labels']:
                    label = row['labels'][horizon]
                    risk = row['predictions'][horizon]['projected_risk']
                    p = min(1 - 1e-15, max(1e-15, risk))
                    per_episode.setdefault(row['identity'], []).append(
                        [(risk - label) ** 2, -label * np.log(p) - (1 - label) * np.log1p(-p), label])
            assert len({len(rows) for rows in per_episode.values()}) == 2
            expected = np.mean([np.mean(rows, axis=0) for rows in per_episode.values()], axis=0)
            actual = diagnostics['metrics_after_projection'][f'{policy}-{horizon}']
            np.testing.assert_allclose([actual['brier'], actual['log_loss'], actual['prevalence']], expected,
                                       rtol=0, atol=1e-12)
    path = tmp_path / 'unequal-calibration.json'
    path.write_text(json.dumps(artifact, allow_nan=False), encoding='utf-8')
    ServicePredictor.load(path)
    labels = [r['labels']['10'] for r in diagnostics['rows'] if r['policy'] == 'hold' and '10' in r['labels']]
    row_average = float(np.mean(labels))
    assert abs(row_average - diagnostics['metrics_after_projection']['hold-10']['prevalence']) > 1e-6
    diagnostics['metrics_after_projection']['hold-10']['prevalence'] = row_average
    path.write_text(json.dumps(artifact, allow_nan=False), encoding='utf-8')
    with pytest.raises(ValueError, match='metric'):
        ServicePredictor.load(path)


def test_platt_fit_is_independent_of_calibration_geometry(fit_inputs, fitted):
    train, calibration = fit_inputs
    changed = [replace(e, frames=service_frames([60] * 115)) if e.policy == 'geometry' else e for e in calibration]
    predictor = fit_service_predictor(train, changed)
    for h in ('10', '20'):
        assert predictor.artifact['models'][h] == fitted.artifact['models'][h]


def test_unequal_trajectory_lengths_have_equal_fit_weight(fit_inputs):
    train, calibration = fit_inputs
    changed = [replace(e, frames=e.frames[:90]) if e.context == 'static' else e for e in train]
    rows = build_service_samples(changed, 10)
    predictor = fit_service_predictor(changed, calibration)
    model = predictor.artifact['models']['10']
    np.testing.assert_allclose(model['mean'], np.average(rows.x, axis=0, weights=rows.weight))
    assert not np.allclose(model['mean'], np.mean(rows.x, axis=0))
