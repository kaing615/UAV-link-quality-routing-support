from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from src.serving import app as serving_app
from src.serving.schemas import PredictionRequest
from src.validation.data_quality import NODE_FEATURE_RANGES, check_tensor_quality


class _CapturingModel:
    def __init__(self) -> None:
        self.node_features: torch.Tensor | None = None

    def __call__(self, x, edge_index, edge_attr, edge_label_index, labeled_edge_attr):
        self.node_features = x.detach().clone()
        return torch.zeros(edge_label_index.shape[1])


class FeatureSchemaContractTests(unittest.TestCase):
    def test_data_quality_checks_degree_then_load(self):
        node_features = torch.tensor(
            [[0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 51.0, 1.25]],
            dtype=torch.float32,
        )

        issues = check_tensor_quality(node_features, "node_features", NODE_FEATURE_RANGES)
        violated_features = {issue["feature"] for issue in issues if issue["check"] == "range_violation"}

        self.assertEqual(violated_features, {"degree", "load"})

    def test_predict_uses_canonical_degree_then_load_node_order(self):
        payload = {
            "nodes": [
                {
                    "node_id": 10,
                    "x": 1.0,
                    "y": 2.0,
                    "z": 3.0,
                    "vx": 4.0,
                    "vy": 5.0,
                    "vz": 6.0,
                    "degree": 7,
                    "load": 0.25,
                },
                {
                    "node_id": 20,
                    "x": 11.0,
                    "y": 12.0,
                    "z": 13.0,
                    "vx": 14.0,
                    "vy": 15.0,
                    "vz": 16.0,
                    "degree": 17,
                    "load": 0.75,
                },
            ],
            "edges": [
                {
                    "src": 10,
                    "dst": 20,
                    "distance": 100.0,
                    "rssi": -60.0,
                    "snr": 30.0,
                    "delay": 2.0,
                    "packet_loss": 0.01,
                    "relative_speed": 3.0,
                    "throughput": 10.0,
                }
            ],
            "query_edges": [(10, 20)],
        }
        try:
            request = PredictionRequest(**payload)
        except ValueError as exc:
            self.fail(f"canonical node feature payload was rejected: {exc}")
        model = _CapturingModel()

        with (
            patch.object(serving_app, "_model", model),
            patch.object(serving_app, "_model_id", "edge-sage"),
            patch.object(serving_app, "_threshold", 0.5),
        ):
            serving_app.predict(request)

        self.assertIsNotNone(model.node_features)
        self.assertEqual(
            model.node_features.tolist(),
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.25],
                [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 0.75],
            ],
        )


if __name__ == "__main__":
    unittest.main()
