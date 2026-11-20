"""Hand-checked causal features, censoring and safe learned-model contracts."""
from copy import deepcopy
import json
import hashlib

import numpy as np
import pytest

from src.control.prevention.features import FEATURE_NAMES, edge_features, measured_edges, bridge_edges
from src.control.prevention.prediction import (
    HORIZONS, EpisodeTrace, LearnedPredictor, build_samples, fit_predictor, validate_partitions,
    load_traces,
)


def graph_frame(step=0, triangle=False):
    rx = [[0, 1, int(triangle)], [1, 0, 1], [int(triangle), 1, 0]]
    return dict(step=step, positions=[[0, 0, 100], [100, 0, 100], [200, 0, 100]],
                velocities=[[0, 0, 0], [2, 0, 0], [0, 0, 0]], rx_ratio=rx,
                rssi=[[-999]*3 for _ in range(3)], rssi_mask=[[False]*3 for _ in range(3)])


def trace(name, seed, failure=True):
    frames = [graph_frame(t) for t in range(26)]
    if failure:
        for f in frames[12:]: f['rx_ratio'][0][1] = f['rx_ratio'][1][0] = 0
    return EpisodeTrace(name, seed, seed+100000, 240, frames, {'transitions': hashlib.sha256(name.encode()).hexdigest()})


def test_line_bridges_and_triangle_no_bridges():
    assert measured_edges(graph_frame()) == [(0, 1), (1, 2)]
    assert bridge_edges(graph_frame()) == {(0, 1), (1, 2)}
    assert bridge_edges(graph_frame(triangle=True)) == set()


def test_features_only_current_and_previous_missing_rssi_mask():
    current, previous = graph_frame(1), graph_frame(0)
    x = edge_features(current, previous, (0, 1), 240, 5)
    assert x[FEATURE_NAMES.index('distance_range')] == pytest.approx(100/240)
    assert x[FEATURE_NAMES.index('radial_speed')] == 2
    assert x[FEATURE_NAMES.index('rssi')] == x[FEATURE_NAMES.index('rssi_present')] == 0
    other = deepcopy(current); other['rssi'][0][1] = -20
    np.testing.assert_equal(x, edge_features(other, previous, (0, 1), 240, 5))
    future = graph_frame(2); future['positions'][1][0] = 9999
    np.testing.assert_equal(x, edge_features(current, previous, (0, 1), 240, 5))
    current['rssi_mask'][0][1] = True; current['rssi'][0][1] = -65
    x = edge_features(current, previous, (0, 1), 240, 5)
    assert x[FEATURE_NAMES.index('rssi')] == -65
    assert x[FEATURE_NAMES.index('rssi_delta_present')] == 0


def test_future_tail_censored_and_failure_label_exact():
    episode = trace('a', 10)
    x, y, ids = build_samples([episode], 20)
    assert len(y) == 12  # t=0..5, two connected edges, no short-tail negatives
    assert sum(y) == 6
    assert set(ids) == {'a'}
    altered = deepcopy(episode); altered.frames[-1]['rx_ratio'][1][2] = 0
    np.testing.assert_equal(x, build_samples([altered], 20)[0])


def test_split_overlap_and_both_classes_required():
    with pytest.raises(ValueError, match='overlap'):
        validate_partitions([trace('a', 10)], [trace('b', 10)])
    with pytest.raises(ValueError, match='both classes'):
        fit_predictor([trace('a', 10, False)], [trace('b', 20, False)])


def test_model_json_roundtrip_finite_schema_and_test_overlap(tmp_path):
    model = fit_predictor([trace('a', 10)], [trace('b', 20)])
    path = tmp_path/'model.json'; model.save(path)
    loaded = LearnedPredictor.load(path)
    risks = loaded.predict(graph_frame(), None, 240)
    assert tuple(risks) == HORIZONS
    assert all(0 <= risk <= 1 for values in risks.values() for risk in values.values())
    loaded.validate_test_scenarios([(30, 100030)])
    with pytest.raises(ValueError, match='overlap'): loaded.validate_test_scenarios([(10, 100010)])
    with pytest.raises(ValueError, match='empty|exists'): model.save(path)
    data = json.loads(path.read_text()); data['feature_version'] = 'old-GNN'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='schema|feature'): LearnedPredictor.load(path)
    data = model.artifact; data['models']['3']['weights'][0] = float('nan')
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='finite|constant'): LearnedPredictor.load(path)


def write_trace(root, name, seed, split='train'):
    path = root/name; path.mkdir(parents=True)
    episode = trace(name, seed)
    from src.control.prevention.config import PreventionConfig
    from dataclasses import asdict
    c = PreventionConfig(num_uavs=3, destination=2, steps=25)
    (path/'config.json').write_text(json.dumps(dict(config=asdict(c), layout_seed=seed, radio_seed=seed+100000)))
    rows = []
    for f in episode.frames:
        f.update(version=1, travel=[0, 0, 0], components=1 if f['step'] < 12 else 2,
                 source_dest_connected=f['step'] < 12, olsr_route_usable=f['step'] < 12,
                 connected_streak=0, safety_rejected=False, terminal=f['step'] == 25)
        rows.append(dict(event='initial' if f['step'] == 0 else 'transition', frame=f, truncated=f['step'] == 25))
    (path/'transitions.jsonl').write_text('\n'.join(json.dumps(row) for row in rows))
    (path/'policy_metadata.json').write_text(json.dumps(dict(collection_policy='hold', split=split,
                                                         layout_seed=seed, radio_seed=seed+100000)))
    return path


def test_training_cli_end_to_end_and_partial_trace_refusal(tmp_path):
    from scripts.train.train_prevention_predictor import main
    train_root, calibration_root = tmp_path/'train', tmp_path/'calibration'
    write_trace(train_root, 'train_episode', 10)
    path = write_trace(calibration_root, 'calibration_episode', 20, 'calibration')
    output = tmp_path/'model'
    report = main(['--train-root', str(train_root), '--calibration-root', str(calibration_root), '--output', str(output)])
    assert report['train_episodes'] == report['calibration_episodes'] == 1
    assert LearnedPredictor.load(output/'predictor.json').artifact['splits']['train'][0]['collection_policy'] == 'hold'
    records = (path/'transitions.jsonl').read_text().splitlines()
    (path/'transitions.jsonl').write_text('\n'.join(records[:-1]))
    with pytest.raises(ValueError, match='horizon'): load_traces(calibration_root)


def collector_fixture(root, split='train'):
    from src.control.prevention.prediction import sha256
    paths = [write_trace(root/'episodes/hold', 'episode_a', 10, split),
             write_trace(root/'episodes/hold', 'episode_b', 11, split)]
    rows = [dict(status='ok', policy='hold', layout_seed=seed, radio_seed=seed+100000,
                 episode_dir=str(path.resolve()), trace_sha256=sha256(path/'transitions.jsonl'))
            for path, seed in zip(paths, (10, 11))]
    (root/'episodes.json').write_text(json.dumps(rows))
    (root/'run_config.json').write_text(json.dumps(dict(schema='prevention-grid-v1', split=split,
                                                      output_root=str(root.resolve()), policies=['hold'],
                                                      scenarios=[[10, 100010], [11, 100011]])))
    return paths, rows


@pytest.mark.parametrize('root_kind', ['run', 'subtree', 'shards'])
def test_manifest_missing_trace_refused_including_subtrees_and_shards(tmp_path, root_kind):
    root = tmp_path/'dataset'
    run = root/'shard-a' if root_kind == 'shards' else root
    paths, _ = collector_fixture(run)
    (paths[1]/'transitions.jsonl').unlink()
    selected = run/'episodes/hold' if root_kind == 'subtree' else root
    with pytest.raises(ValueError, match='missing|Missing|manifest|Manifest'):
        load_traces(selected)


@pytest.mark.parametrize('mutation', ['hash', 'seed', 'policy', 'duplicate_dir', 'unlisted'])
def test_manifest_reconciles_hash_identity_and_unlisted_traces(tmp_path, mutation):
    paths, rows = collector_fixture(tmp_path)
    if mutation == 'hash': rows[0]['trace_sha256'] = '0'*64
    elif mutation == 'seed':
        data = json.loads((paths[0]/'config.json').read_text()); data['layout_seed'] = 999
        (paths[0]/'config.json').write_text(json.dumps(data))
    elif mutation == 'policy':
        data = json.loads((paths[0]/'policy_metadata.json').read_text()); data['collection_policy'] = 'reactive'
        (paths[0]/'policy_metadata.json').write_text(json.dumps(data))
    elif mutation == 'duplicate_dir': rows[1]['episode_dir'] = rows[0]['episode_dir']
    else: write_trace(tmp_path/'episodes/hold', 'unlisted', 12)
    (tmp_path/'episodes.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError): load_traces(tmp_path/'episodes/hold')


@pytest.mark.parametrize('bad_role', ['train', 'calibration'])
def test_training_cli_refuses_known_test_in_either_input(tmp_path, bad_role):
    from scripts.train.train_prevention_predictor import main
    write_trace(tmp_path/'train', 'train_episode', 10, 'test' if bad_role == 'train' else 'train')
    write_trace(tmp_path/'calibration', 'calibration_episode', 20, 'test' if bad_role == 'calibration' else 'calibration')
    with pytest.raises(ValueError, match='split|partition'):
        main(['--train-root', str(tmp_path/'train'), '--calibration-root', str(tmp_path/'calibration'),
              '--output', str(tmp_path/'model')])
    assert not (tmp_path/'model').exists()


def test_run_episode_split_conflict_and_expected_partition_refused(tmp_path):
    paths, _ = collector_fixture(tmp_path, 'train')
    data = json.loads((paths[0]/'policy_metadata.json').read_text()); data['split'] = 'test'
    (paths[0]/'policy_metadata.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='split|partition'): load_traces(tmp_path)
    data['split'] = 'train'; (paths[0]/'policy_metadata.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='split|partition'): load_traces(tmp_path, expected_split='test')


def test_manifest_relocation_binds_recorded_tree_and_retains_split(tmp_path):
    import shutil
    original, downloaded = tmp_path/'original', tmp_path/'downloaded'
    collector_fixture(original, 'test')
    shutil.copytree(original, downloaded)
    episodes = load_traces(downloaded/'episodes/hold', expected_split='test')
    assert len(episodes) == 2 and {e.collection_split for e in episodes} == {'test'}
    assert all(str(downloaded) in key for e in episodes for key in e.source_hashes)
    # An absolute Linux output root also relocates to the same recorded tree.
    run_path = downloaded/'run_config.json'; rows_path = downloaded/'episodes.json'
    run = json.loads(run_path.read_text()); run['output_root'] = '/remote/pilot/test'
    rows = json.loads(rows_path.read_text())
    for row in rows:
        row['episode_dir'] = '/remote/pilot/test/episodes/hold/'+row['episode_dir'].replace('\\', '/').split('/')[-1]
    run_path.write_text(json.dumps(run)); rows_path.write_text(json.dumps(rows))
    assert len(load_traces(downloaded, expected_split='test')) == 2
    rows[0]['episode_dir'] = '/remote/other/episode_a'; rows_path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match='outside'): load_traces(downloaded)


def test_direct_fit_refuses_recorded_test_role():
    train, calibration = trace('a', 10), trace('b', 20)
    train.collection_split = 'test'
    with pytest.raises(ValueError, match='split|partition'): fit_predictor([train], [calibration])


def test_deleted_episode_manifest_is_not_admitted_as_unmanifested(tmp_path):
    collector_fixture(tmp_path)
    (tmp_path/'episodes.json').unlink()
    with pytest.raises(ValueError, match='manifest|Manifest'): load_traces(tmp_path/'episodes/hold')


def test_training_cli_conflict_and_scoring_cli_require_recorded_roles(tmp_path):
    from scripts.train.train_prevention_predictor import main as train_main
    from scripts.analysis.evaluate_prevention import main as score_main
    root = tmp_path/'train'
    paths, _ = collector_fixture(root)
    metadata_path = paths[0]/'policy_metadata.json'
    metadata = json.loads(metadata_path.read_text()); metadata['split'] = 'test'
    metadata_path.write_text(json.dumps(metadata))
    write_trace(tmp_path/'calibration', 'calibration_episode', 20, 'calibration')
    with pytest.raises(ValueError, match='split|partition'):
        train_main(['--train-root', str(root), '--calibration-root', str(tmp_path/'calibration'),
                    '--output', str(tmp_path/'model-output')])
    assert not (tmp_path/'model-output').exists()
    metadata['split'] = 'train'; metadata_path.write_text(json.dumps(metadata))
    fit_predictor([trace('model-train', 100)], [trace('model-calibration', 200)]).save(tmp_path/'model.json')
    with pytest.raises(ValueError, match='split|partition'):
        score_main(['--split', 'test', '--model', str(tmp_path/'model.json'), '--score-traces', str(root/'episodes/hold'),
                    '--output', str(tmp_path/'score-output')])
    assert not (tmp_path/'score-output').exists()
