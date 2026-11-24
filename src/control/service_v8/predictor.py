"""Frozen Edge-SAGE/MLP inference in NumPy; Torch is only needed for training."""
import numpy as np

from .features import FEATURE_SCHEMA, Features


class Predictor:
    def __init__(self, artifact):
        if artifact["feature_schema"] != FEATURE_SCHEMA:
            raise ValueError("Feature schema mismatch")
        self.artifact = artifact
        self.graph = artifact["graph"]
        self.weights = {key: np.asarray(value, dtype=np.float32) for key, value in artifact["weights"].items()}
        if not all(np.isfinite(value).all() for value in self.weights.values()):
            raise ValueError("Non-finite model weights")
        self.features = Features()

    def linear(self, x, prefix):
        return x @ self.weights[prefix + ".weight"].T + self.weights[prefix + ".bias"]

    def predict_inputs(self, node, edge, adjacency):
        h = node
        norm = adjacency / np.maximum(adjacency.sum(axis=1, keepdims=True), 1)
        edge_mean = (norm[:, :, None] * edge).sum(axis=1)
        for i in range(2):
            updated = self.linear(h, f"self_layers.{i}")
            if self.graph:
                neighbor = np.concatenate((norm @ h, edge_mean), axis=-1)
                updated += self.linear(neighbor, f"message_layers.{i}")
            h = np.maximum(updated, 0)
        n = len(node)
        pair = np.concatenate((np.broadcast_to(h[:, None, :], (n, n, h.shape[-1])),
                               np.broadcast_to(h[None, :, :], (n, n, h.shape[-1])), edge), axis=-1)
        hidden = np.maximum(self.linear(pair, "decoder.0"), 0)
        logits = self.linear(hidden, "decoder.2")[..., 0]
        logits = (logits + logits.T) / 2
        value = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        np.fill_diagonal(value, 0)
        if not np.isfinite(value).all():
            raise ValueError("Non-finite GNN prediction")
        return value

    def predict(self, frame):
        return self.predict_inputs(*self.features.update(frame))
