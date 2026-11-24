"""Scientific-contract checks for causal features, deployment and paired evaluation."""
import copy

import numpy as np
import torch

from scripts.train.train_service_v8_gnn import ForecastNet
from src.control.service_v8.features import FEATURE_SCHEMA, Features
from src.control.service_v8.predictor import Predictor


def frame(n, step=0):
    return dict(step=step, positions=[[24 * i, 0, 100] for i in range(n)],
                velocities=[[0, 0, 0] for _ in range(n)], rx_ratio=(np.ones((n, n)) - np.eye(n)).tolist(),
                rssi=np.full((n, n), -70.).tolist(), rssi_mask=(~np.eye(n, dtype=bool)).tolist())


def artifact(model):
    return dict(feature_schema=FEATURE_SCHEMA, graph=model.graph,
                weights={key: value.detach().numpy().tolist() for key, value in model.state_dict().items()})


def test_numpy_export_matches_torch_for_variable_graph_sizes_and_empty_graph():
    torch.manual_seed(20261123)
    for graph in (True, False):
        model = ForecastNet(graph).eval()
        deployed = Predictor(artifact(model))
        for n in (4, 5, 7):
            for disconnected in (False, True):
                observation = frame(n)
                if disconnected:
                    observation["rx_ratio"] = np.zeros((n, n)).tolist()
                    observation["rssi_mask"] = np.zeros((n, n), bool).tolist()
                inputs = Features().update(observation)
                with torch.no_grad():
                    expected = model(*(torch.from_numpy(value[None]) for value in inputs))[0].numpy()
                actual = deployed.predict_inputs(*inputs)
                np.testing.assert_allclose(actual, expected, atol=1e-6)
                np.testing.assert_allclose(actual, actual.T)
                assert np.isfinite(actual).all() and np.all(np.diag(actual) == 0)


def test_features_ignore_mission_metadata_and_future_labels():
    original = frame(5)
    changed = copy.deepcopy(original)
    changed.update(mission_start=0, future_rx_ratio=np.zeros((5, 5)).tolist(), future_pdr=0,
                   service_observation={"cumulative": {"sink_rx_packets": 99999}})
    for a, b in zip(Features().update(original), Features().update(changed)):
        np.testing.assert_array_equal(a, b)


def test_features_are_translation_invariant_and_ewma_is_causal():
    first = frame(5)
    moved = copy.deepcopy(first)
    moved["positions"] = (np.asarray(first["positions"]) + [100, -200, 40]).tolist()
    for a, b in zip(Features().update(first), Features().update(moved)):
        np.testing.assert_allclose(a, b, atol=1e-6)
    features = Features()
    features.update(first)
    second = frame(5, 1)
    second["rx_ratio"] = (np.asarray(first["rx_ratio"]) * .5).tolist()
    _, edge, _ = features.update(second)
    assert edge[0, 1, 3] == .75
    assert edge[0, 1, 4] == -.5


def test_nonconsecutive_frames_fail():
    import pytest
    features = Features()
    features.update(frame(5))
    with pytest.raises(ValueError, match="consecutive"):
        features.update(frame(5, 2))


def test_planned_cells_are_disjoint_across_phases():
    from src.control.service_v3.protocol import build_phases, validate_phases
    from src.control.service_v8.study import V8
    phases = validate_phases(build_phases(V8), V8)
    assert len(phases["validation"]) * len(V8.policies["validation"]) == 80
    assert len(phases["test"]) * len(V8.policies["test"]) == 192
    assert set(s["layout_seed"] for s in phases["validation"]).isdisjoint(s["layout_seed"] for s in phases["test"])
