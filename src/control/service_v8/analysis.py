"""Locked v8 test sequence; validation has technical gates only."""
import json
from pathlib import Path

import numpy as np

from ..service_v2.protocol import reserve, sha256, write_json
from ..service_v3.analysis import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, blocks, contrast
from ..service_v3.study import load_traces
from ..service_v4.study import route_integrity
from ..service_v7.study import background, recorded_command, startup_blackout, transmissions
from .features import Features, HORIZON
from .predictor import Predictor
from .study import ROUTE_ARMS, V8, arm_flags, forecast_trace_ok, load

PAIRS = (("gnn-route", "lq-route"), ("gnn-route", "mlp-route"),
         ("gnn-route", "olsr-etx-fast"), ("mlp-route", "lq-route"), ("lq-route", "olsr-etx-fast"))


def technical(phase, rows):
    integrity = {row["cell_id"]: route_integrity(row) for row in rows if row["policy"] in ROUTE_ARMS}
    flag_checks = []
    for row in rows:
        expected = arm_flags(row["policy"])
        stderr = (Path(row["episode_dir"]) / "simulator.stderr.log").read_text()
        command = recorded_command(row)
        flag_checks.append(command[-len(expected):] == expected and
                           (("UAVOLSR link_quality=etx" in stderr) == (row["policy"] == "olsr-etx-fast")))
    tx = {row["cell_id"]: transmissions(row) for row in rows}
    bg = {row["cell_id"]: background(row) for row in rows}
    checks = dict(route_integrity_ok=all(item["ok"] for item in integrity.values()), route_integrity=integrity,
                  flags_ok=all(flag_checks), transmissions_ok=all(value is not None for value in tx.values()),
                  background_ok=all(value is not None for value in bg.values()),
                  startup_blackouts=[row["cell_id"] for row in rows if startup_blackout(row)],
                  forecast_traces_ok=all(forecast_trace_ok(row, phase["selection_sha256"]) for row in rows),
                  relays_stationary=all(row["total_relay_travel_m"] < 1e-6 for row in rows))
    return checks, tx, bg


def forecast_evaluation(rows, selection):
    """All predictors evaluated on exactly the same held-out lq-route trajectories."""
    out = []
    predictors = {name: Predictor(selection["models"][name]) for name in ("gnn", "mlp")}
    for row in rows:
        if row["policy"] != "lq-route":
            continue
        errors = {name: [] for name in ("gnn", "mlp", "ewma")}
        features = Features()
        frames = row["frames"]
        for i, frame in enumerate(frames):
            inputs = features.update(frame)
            if i + HORIZON >= len(frames):
                continue
            target = np.mean([np.asarray(f["rx_ratio"]) for f in frames[i + 1:i + HORIZON + 1]], axis=0)
            mask = ~np.eye(len(target), dtype=bool)
            estimates = {name: predictor.predict_inputs(*inputs) for name, predictor in predictors.items()}
            estimates["ewma"] = features.ewma
            for name, predicted in estimates.items():
                errors[name].append(float(np.mean((predicted[mask] - target[mask]) ** 2)))
        out.append(dict(cell_id=row["cell_id"], regime=row["regime"], context=row["context"],
                        origins=len(errors["gnn"]), mse={name: float(np.mean(values)) for name, values in errors.items()}))
    return dict(unit="episode; both learned predictors and EWMA evaluated on the same lq-route observations",
                horizon_seconds=HORIZON, episodes=out,
                balanced_gradual_mse={name: float(np.mean([np.mean([r["mse"][name] for r in out
                                                 if r["context"] == "gradual" and r["regime"] == regime])
                                                 for regime in V8.regimes])) for name in ("gnn", "mlp", "ewma")})


def analyze(protocol_path, run_root, output):
    phase, report, rows = load_traces(protocol_path, run_root, loader=load)
    selection = json.loads((Path(phase["_path"]).parent / "selection.json").read_text())
    checks, tx, bg = technical(phase, rows)
    out = reserve(output)
    for row in rows:
        row["disconnected_seconds"] = sum(not f["source_dest_connected"] for f in row["frames"][1:])
        row["no_olsr_route_seconds"] = sum(not f["service_observation"]["route"]["success"] for f in row["frames"][1:])
        if checks["background_ok"]:
            rx, sent = bg[row["cell_id"]]
            row["background_pdr"] = rx / sent
    data = blocks(rows, phase["policies"], regimes=V8.regimes)
    static = blocks(rows, phase["policies"], context="static", regimes=V8.regimes)
    metrics = ["actual_udp_pdr", "disconnected_seconds", "no_olsr_route_seconds"]
    if checks["background_ok"]:
        metrics.append("background_pdr")
    result = dict(schema=V8.schema(phase["phase"] + "-analysis"), freeze_sha256=phase["_freeze_sha256"],
                  selection_sha256=phase["selection_sha256"], planned_cells=report["planned_cells"], ok_cells=report["ok_cells"],
                  **checks, descriptive=[contrast(data, a, b, metric) for a, b in PAIRS for metric in metrics],
                  static_descriptive=[contrast(static, a, b, "actual_udp_pdr") for a, b in PAIRS[:3]],
                  bootstrap=dict(samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED,
                                 unit="gradual scenario block, four arms together; equal n5/n7"),
                  ip_transmissions=tx,
                  control_cost_note="Central-report traffic is not simulated; v7's ratio-only reporting estimate does not cover GNN telemetry.",
                  rows=[{key: value for key, value in row.items() if key not in ("frames", "decisions")} for row in rows])
    if phase["phase"] == "validation":
        criteria = {key: dict(passed=checks[key]) for key in ("route_integrity_ok", "flags_ok", "transmissions_ok",
                                                            "background_ok", "forecast_traces_ok", "relays_stationary")}
        criteria["no_startup_blackout"] = dict(passed=not checks["startup_blackouts"], cells=checks["startup_blackouts"])
        go = all(item["passed"] for item in criteria.values())
        result["go"] = dict(go=go, criteria=criteria)
        write_json(out / "go.json", dict(schema=V8.schema("go"), go=go, criteria=criteria,
                                          validation_freeze_sha256=phase["_freeze_sha256"]))
    else:
        sequence, gate = [], True
        for a, b in PAIRS[:3]:
            comparison = contrast(data, a, b, "actual_udp_pdr")
            success = comparison["ci95"] is not None and comparison["ci95"][0] > 0
            sequence.append(dict(name=f"pdr_{a}_vs_{b}", rule="lower_gt_0", success=success,
                                 interpretable=gate, contrast=comparison))
            gate = gate and success
        result["fixed_sequence"] = sequence
        result["forecast_evaluation"] = forecast_evaluation(rows, selection)
    write_json(out / "analysis.json", result)
    write_json(out / "manifest.json", dict(files={path.name: sha256(path) for path in out.iterdir() if path.is_file()}))
    print(json.dumps({"phase": phase["phase"], "cells": report["ok_cells"],
                      "go": result.get("go"), "fixed_sequence": result.get("fixed_sequence")}, indent=2))
    return result
