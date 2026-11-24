"""Exploratory: training-seed variability of the v8 forecaster (offline MSE only).

Refits GNN and MLP with the locked v8 recipe for several seeds on the same
development data, then scores each fit on the same held-out v8 test
lq-route gradual traces as the locked forecast evaluation (balanced n5/n7).
The first seed is the locked one, so its row must reproduce the locked MSE.
No closed-loop PDR is implied; not part of the pre-registered analysis.

Usage: py scripts/train/multiseed_service_v8.py --data reports/uav_service_v8/development
       --traces <dir with runs/test/...lq-route.../transitions.jsonl> --output out.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_service_v8_gnn as train  # noqa: E402
from src.control.service_v8.features import HORIZON, Features  # noqa: E402
from src.control.service_v8.predictor import Predictor  # noqa: E402

SEEDS = (20261123, 1, 2, 3, 4)


def load_traces(root):
    out = []
    for path in sorted(Path(root).glob("runs/test/*gradual*__lq-route/attempt-0/**/transitions.jsonl")):
        regime = path.parts[path.parts.index("test") + 1].split("-")[1]
        out.append((regime, [json.loads(line)["frame"] for line in path.read_text().splitlines()]))
    assert len(out) == 40, len(out)
    return out


def score(artifacts, traces):
    """Same procedure as service_v8.analysis.forecast_evaluation, for many predictors at once."""
    per_regime = {}
    for regime, frames in traces:
        predictors = {name: Predictor(a) for name, a in artifacts.items()}
        errors = {name: [] for name in [*artifacts, "ewma"]}
        features = Features()
        for i, frame in enumerate(frames):
            inputs = features.update(frame)
            if i + HORIZON >= len(frames):
                continue
            target = np.mean([np.asarray(f["rx_ratio"]) for f in frames[i + 1:i + HORIZON + 1]], axis=0)
            mask = ~np.eye(len(target), dtype=bool)
            estimates = {name: p.predict_inputs(*inputs) for name, p in predictors.items()}
            estimates["ewma"] = features.ewma
            for name, predicted in estimates.items():
                errors[name].append(float(np.mean((predicted[mask] - target[mask]) ** 2)))
        for name, values in errors.items():
            per_regime.setdefault(name, {}).setdefault(regime, []).append(np.mean(values))
    return {name: float(np.mean([np.mean(v) for v in regimes.values()])) for name, regimes in per_regime.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--traces", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    groups, _ = train.load_data(args.data)
    traces = load_traces(args.traces)
    locked = json.loads((ROOT / "reports/uav_service_v8/model/selection.json").read_text())["models"]
    rows = [dict(seed="locked", best_epoch={k: v["best_epoch"] for k, v in locked.items()},
                 test_mse=score(locked, traces))]
    print(json.dumps(rows[-1]), flush=True)
    for seed in SEEDS:
        train.TRAIN_SEED = seed
        fitted = {name: train.fit(graph, groups)[0] for name, graph in (("gnn", True), ("mlp", False))}
        rows.append(dict(seed=seed, best_epoch={k: v["best_epoch"] for k, v in fitted.items()},
                         early_stop_mse={k: v["early_stop_mse"] for k, v in fitted.items()},
                         test_mse=score(fitted, traces)))
        print(json.dumps(rows[-1]), flush=True)
    fits = rows[1:]
    summary = {}
    for name in ("gnn", "mlp"):
        values = np.array([r["test_mse"][name] for r in fits])
        summary[name] = dict(mean=float(values.mean()), sd=float(values.std(ddof=1)),
                             min=float(values.min()), max=float(values.max()))
    summary["ewma"] = fits[0]["test_mse"]["ewma"]
    summary["gnn_below_mlp_seeds"] = sum(r["test_mse"]["gnn"] < r["test_mse"]["mlp"] for r in fits)
    summary["gnn_below_ewma_seeds"] = sum(r["test_mse"]["gnn"] < r["test_mse"]["ewma"] for r in fits)
    args.output.write_text(json.dumps(dict(note="exploratory; offline MSE only; first fitted seed is the locked one",
                                           seeds=list(SEEDS), rows=rows, summary=summary), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
