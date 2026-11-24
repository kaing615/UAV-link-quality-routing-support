"""Fit fixed causal Edge-SAGE and graphless ablation on archived development data."""
import argparse
import copy
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn

from src.control.service_v8.features import FEATURE_SCHEMA, HORIZON, Features
from src.control.service_v8.predictor import Predictor

TRAIN_SEED = 20261123
MAX_EPOCHS = 60
PATIENCE = 10
BATCH_SIZE = 128
HIDDEN = 64


class ForecastNet(nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.graph = graph
        self.self_layers = nn.ModuleList([nn.Linear(8, HIDDEN), nn.Linear(HIDDEN, HIDDEN)])
        self.message_layers = nn.ModuleList([nn.Linear(8 + 7, HIDDEN), nn.Linear(HIDDEN + 7, HIDDEN)]) if graph else nn.ModuleList()
        self.decoder = nn.Sequential(nn.Linear(2 * HIDDEN + 7, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, 1))

    def forward(self, node, edge, adjacency):
        h = node
        norm = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1)
        edge_mean = (norm[..., None] * edge).sum(dim=2)
        for i in range(2):
            updated = self.self_layers[i](h)
            if self.graph:
                updated = updated + self.message_layers[i](torch.cat((norm @ h, edge_mean), dim=-1))
            h = torch.relu(updated)
        batch, n, width = h.shape
        pair = torch.cat((h[:, :, None, :].expand(batch, n, n, width),
                          h[:, None, :, :].expand(batch, n, n, width), edge), dim=-1)
        logits = self.decoder(pair)[..., 0]
        logits = (logits + logits.transpose(1, 2)) / 2
        mask = 1 - torch.eye(n, device=logits.device)
        return torch.sigmoid(logits) * mask


def load_data(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / "development.jsonl.gz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["data_sha256"]
    groups = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            episode = json.loads(line)
            frames = episode["frames"]
            features = Features()
            for i, frame in enumerate(frames):
                inputs = features.update(frame)
                if i + HORIZON >= len(frames):
                    continue
                target = np.mean([np.asarray(f["rx_ratio"], dtype=np.float32) for f in frames[i + 1:i + HORIZON + 1]], axis=0)
                key = (episode["split"], len(inputs[0]))
                groups.setdefault(key, []).append((*inputs, target))
    return {key: tuple(torch.from_numpy(np.stack([sample[i] for sample in samples])) for i in range(4))
            for key, samples in groups.items()}, manifest


def mse(predicted, target):
    n = target.shape[-1]
    return ((predicted - target) ** 2).sum(dim=(1, 2)).mean() / (n * (n - 1))


def evaluate(model, groups):
    values, persistence = [], []
    model.eval()
    with torch.no_grad():
        for n in (5, 7):
            x, e, a, y = groups[("early_stop", n)]
            total = 0.
            for start in range(0, len(x), BATCH_SIZE):
                pred = model(x[start:start + BATCH_SIZE], e[start:start + BATCH_SIZE], a[start:start + BATCH_SIZE])
                total += float(mse(pred, y[start:start + BATCH_SIZE])) * len(pred)
            values.append(total / len(x))
            persistence.append(float(mse(e[..., 3], y)))
    return sum(values) / 2, sum(persistence) / 2


def fit(graph, groups):
    torch.manual_seed(TRAIN_SEED)
    model = ForecastNet(graph)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    rng = np.random.default_rng(TRAIN_SEED)
    best_loss = float("inf")
    best_state = None
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batches = []
        # Equal number of samples in each regime; each epoch visits every graph once.
        for n in (5, 7):
            order = rng.permutation(len(groups[("train", n)][0]))
            batches.extend((n, order[start:start + BATCH_SIZE]) for start in range(0, len(order), BATCH_SIZE))
        rng.shuffle(batches)
        for n, indices in batches:
            x, e, a, y = (value[indices] for value in groups[("train", n)])
            optimizer.zero_grad()
            loss = mse(model(x, e, a), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        validation, persistence = evaluate(model, groups)
        history.append(dict(epoch=epoch, early_stop_mse=validation))
        if validation < best_loss:
            best_loss, best_epoch, best_state = validation, epoch, copy.deepcopy(model.state_dict())
        print(json.dumps(dict(model="gnn" if graph else "mlp", epoch=epoch, validation_mse=validation,
                              persistence_mse=persistence)), flush=True)
        if epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    artifact = dict(feature_schema=FEATURE_SCHEMA, graph=graph, horizon=HORIZON,
                    hidden_channels=HIDDEN, best_epoch=best_epoch, early_stop_mse=best_loss,
                    weights={key: value.detach().numpy().tolist() for key, value in model.state_dict().items()})
    exported = Predictor(artifact)
    error = 0.
    for n in (5, 7):
        x, e, a, y = groups[("early_stop", n)]
        for index in np.linspace(0, len(x) - 1, 20, dtype=int):
            with torch.no_grad():
                expected = model(x[index:index + 1], e[index:index + 1], a[index:index + 1])[0].numpy()
            actual = exported.predict_inputs(x[index].numpy(), e[index].numpy(), a[index].numpy())
            error = max(error, float(np.max(np.abs(actual - expected))))
    assert error < 1e-5, error
    artifact["export_parity_max_abs_error"] = error
    return artifact, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    groups, manifest = load_data(args.data)
    print(json.dumps({str(key): len(value[0]) for key, value in groups.items()}), flush=True)
    models, history = {}, {}
    for name, graph in (("gnn", True), ("mlp", False)):
        models[name], history[name] = fit(graph, groups)
    selection = dict(schema="service-v8-tuning-selection-v1",
                     predictor_sha256="719816f80b9df51c5544422522265e50bd454768869523036a8b8f4b1c657c10",
                     legacy_predictor_role="Unused compatibility artifact of the v2 collector",
                     feature_schema=FEATURE_SCHEMA, horizon=HORIZON, margin=1., hello_interval_s=.5,
                     training_seed=TRAIN_SEED, models=models,
                     development_data_sha256=manifest["data_sha256"],
                     development_manifest=manifest,
                     training=dict(max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BATCH_SIZE,
                                   learning_rate=.001, weight_decay=1e-5, grad_clip=5.,
                                   objective="mean future-five-second probe-delivery MSE, equal regimes",
                                   history=history),
                     runtime=dict(torch=torch.__version__, numpy=np.__version__, python=sys.version),
                     source_sha256={path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in [Path(__file__), ROOT / "src/control/service_v8/features.py",
                                                 ROOT / "src/control/service_v8/predictor.py"]})
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False) + "\n")
    print("MODEL_EXPORT_OK", {key: value["export_parity_max_abs_error"] for key, value in models.items()}, flush=True)


if __name__ == "__main__":
    main()
