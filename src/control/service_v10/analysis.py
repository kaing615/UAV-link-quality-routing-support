"""v10 tuning selection and locked validation/test analysis (docs/service_v10_protocol.md)."""
import json
from pathlib import Path

import numpy as np

from ..service_v2.protocol import reserve, sha256, write_json
from ..service_v3.analysis import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, blocks, contrast
from ..service_v3.protocol import V2_MODEL_SHA256
from ..service_v3.study import load_traces
from ..service_v4.study import route_integrity
from ..service_v7.study import background, startup_blackout
from ..service_v8.features import HORIZON, Features
from ..service_v8.predictor import Predictor
from .study import ALPHAS, BETAS, TEST_ARMS, TUNE_ARMS, V10, V10T, load, load_tune

CHANGES = ("engage", "switch", "release", "replace_broken", "release_broken")


def choose(means, grid):
    """Highest balanced mean PDR; ties (< 1e-9) go to the larger coefficient (less smoothing)."""
    best = max(means[arm] for arm in grid)
    return max(grid[arm] for arm in grid if best - means[arm] < 1e-9)


def balanced_means(rows, arms, regimes):
    return {arm: float(np.mean([np.mean([r["actual_udp_pdr"] for r in rows if r["policy"] == arm
                                         and r["regime"] == g and r["context"] == "gradual"]) for g in regimes]))
            for arm in arms}


def forecast_ok(row, selection_sha256):
    if not row["policy"].startswith("gnn"):
        return True
    n = row["config"]["num_uavs"]
    for d in row["decisions"]:
        item = d.get("forecast_control", {})
        q = np.asarray(item.get("q", []))
        if (item.get("model_sha256") != selection_sha256 or item.get("origin_step") != d["step"]
                or q.shape != (n, n) or not np.isfinite(q).all() or np.any((q < 0) | (q > 1))):
            return False
    return True


def technical(phase, rows):
    return dict(route_integrity_ok=all(route_integrity(r)["ok"] for r in rows),
                forecast_traces_ok=all(forecast_ok(r, phase["selection_sha256"]) for r in rows),
                relays_stationary=all(r["total_relay_travel_m"] < 1e-6 for r in rows),
                startup_blackouts=[r["cell_id"] for r in rows if startup_blackout(r)])


def annotate(rows):
    for r in rows:
        r["route_changes"] = sum(d["route_control"]["reason"] in CHANGES for d in r["decisions"])
        r["disconnected_seconds"] = sum(not f["source_dest_connected"] for f in r["frames"][1:])
        bg = background(r)
        r["background_pdr"] = None if bg is None else bg[0] / bg[1]


def _strip(rows):
    return [{k: v for k, v in r.items() if k not in ("frames", "decisions")} for r in rows]


def analyze_tuning(protocol_path, run_root, output):
    phase, report, rows = load_traces(protocol_path, run_root, loader=load_tune)
    annotate(rows)
    out = reserve(output)
    means = balanced_means(rows, TUNE_ARMS, V10T.regimes)
    alpha, beta = choose(means, ALPHAS), choose(means, BETAS)
    v8 = json.loads((Path(phase["_path"]).parent / "selection.json").read_text())
    selection = dict(schema=V10.schema("tuning-selection"), predictor_sha256=V2_MODEL_SHA256,
                     alpha=alpha, beta=beta, models=v8["models"], feature_schema=v8["feature_schema"],
                     horizon=v8["horizon"], margin=1.,
                     tuning=dict(freeze_sha256=phase["_freeze_sha256"], balanced_gradual_pdr=means,
                                 rule="max balanced mean gradual PDR per family; ties -> larger coefficient"),
                     v8_models_from=v8.get("derived_from"))
    checks = technical(phase, rows)
    write_json(out / "tuning-selection.json", selection)
    write_json(out / "analysis.json", dict(schema=V10T.schema("analysis"), ok_cells=report["ok_cells"],
                                           planned_cells=report["planned_cells"], **checks, alpha=alpha, beta=beta,
                                           balanced_gradual_pdr=means,
                                           route_changes={a: sum(r["route_changes"] for r in rows if r["policy"] == a)
                                                          for a in TUNE_ARMS}, rows=_strip(rows)))
    write_json(out / "manifest.json", dict(files={p.name: sha256(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps(dict(alpha=alpha, beta=beta, means=means, ok_cells=report["ok_cells"], **checks), indent=1))


def forecast_mse(rows, selection):
    predictor = Predictor(selection["models"]["gnn"])
    per = {}
    for r in rows:
        if r["policy"] != "lq-route" or r["context"] != "gradual":
            continue
        feats, ewma, gnn, errs = Features(), None, None, {"ewma_tuned": [], "gnn_tuned": [], "ewma_050": []}
        frames = r["frames"]
        for i, f in enumerate(frames):
            inputs = feats.update(f)
            rx = np.asarray(f["rx_ratio"])
            g = predictor.predict_inputs(*inputs)
            ewma = rx if ewma is None else selection["alpha"] * rx + (1 - selection["alpha"]) * ewma
            gnn = g if gnn is None else selection["beta"] * g + (1 - selection["beta"]) * gnn
            if i + HORIZON >= len(frames):
                continue
            t = np.mean([np.asarray(x["rx_ratio"]) for x in frames[i + 1:i + HORIZON + 1]], axis=0)
            m = ~np.eye(len(t), dtype=bool)
            for name, est in (("ewma_tuned", ewma), ("gnn_tuned", gnn), ("ewma_050", feats.ewma)):
                errs[name].append(float(np.mean((est[m] - t[m]) ** 2)))
        for name, v in errs.items():
            per.setdefault(name, {}).setdefault(r["regime"], []).append(np.mean(v))
    return {name: float(np.mean([np.mean(v) for v in regimes.values()])) for name, regimes in per.items()}


def analyze(protocol_path, run_root, output):
    phase, report, rows = load_traces(protocol_path, run_root, loader=load)
    selection = json.loads((Path(phase["_path"]).parent / "selection.json").read_text())
    annotate(rows)
    checks = technical(phase, rows)
    out = reserve(output)
    data = blocks(rows, TEST_ARMS, regimes=V10.regimes)
    static = blocks(rows, TEST_ARMS, context="static", regimes=V10.regimes)
    pairs = (("gnn-tuned", "ewma-tuned"), ("ewma-tuned", "lq-route"), ("gnn-tuned", "lq-route"))
    metrics = ["actual_udp_pdr", "route_changes", "disconnected_seconds"]
    if all(r["background_pdr"] is not None for r in rows):
        metrics.append("background_pdr")
    result = dict(schema=V10.schema(phase["phase"] + "-analysis"), freeze_sha256=phase["_freeze_sha256"],
                  selection_sha256=phase["selection_sha256"], alpha=selection["alpha"], beta=selection["beta"],
                  planned_cells=report["planned_cells"], ok_cells=report["ok_cells"], **checks,
                  descriptive=[contrast(data, a, b, m) for a, b in pairs for m in metrics],
                  static_descriptive=[contrast(static, a, b, "actual_udp_pdr") for a, b in pairs],
                  bootstrap=dict(samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED, unit="gradual block, three arms"),
                  rows=_strip(rows))
    if phase["phase"] == "validation":
        criteria = {k: dict(passed=checks[k]) for k in ("route_integrity_ok", "forecast_traces_ok", "relays_stationary")}
        criteria["complete"] = dict(passed=report["ok_cells"] == report["planned_cells"])
        criteria["no_startup_blackout"] = dict(passed=not checks["startup_blackouts"], cells=checks["startup_blackouts"])
        go = all(c["passed"] for c in criteria.values())
        result["go"] = dict(go=go, criteria=criteria)
        write_json(out / "go.json", dict(schema=V10.schema("go"), go=go, criteria=criteria,
                                          validation_freeze_sha256=phase["_freeze_sha256"]))
    else:
        h1 = contrast(data, "gnn-tuned", "ewma-tuned", "actual_udp_pdr")
        low, high = h1["ci95"]
        verdict = "gnn_better" if low > 0 else "ewma_better" if high < 0 else "no_difference_detected"
        h2 = contrast(data, "ewma-tuned", "lq-route", "actual_udp_pdr")
        result["fixed_sequence"] = [
            dict(name="H1 gnn-tuned vs ewma-tuned", verdict=verdict, contrast=h1),
            dict(name="H2 ewma-tuned vs lq-route", rule="lower_gt_0", success=h2["ci95"][0] > 0,
                 interpretable=verdict != "no_difference_detected", contrast=h2)]
        result["forecast_mse"] = forecast_mse(rows, selection)
    write_json(out / "analysis.json", result)
    write_json(out / "manifest.json", dict(files={p.name: sha256(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps({k: result.get(k) for k in ("ok_cells", "alpha", "beta", "route_integrity_ok", "forecast_traces_ok",
                                                  "startup_blackouts", "go", "fixed_sequence", "forecast_mse")},
                     indent=1, default=str))
    return result
