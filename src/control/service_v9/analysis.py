"""Pre-specified partition endpoint and secondary delivery/cost outcomes."""
import json
from pathlib import Path
import numpy as np

from ..service_v2.protocol import reserve, sha256, write_json
from ..service_v3.analysis import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, blocks, contrast
from ..service_v3.study import load_traces
from ..service_v4.study import route_integrity
from ..service_v7.study import background, recorded_command, startup_blackout, transmissions
from ..service_v8.study import forecast_trace_ok
from .study import V9, arm_flags, load

STEPS = (("disconnected_seconds", "reactive-motion", "gnn-motion"),
         ("actual_udp_pdr", "gnn-motion", "reactive-motion"),
         ("disconnected_seconds", "mlp-motion", "gnn-motion"))


def safety(row):
    config = row["config"]
    for frame in row["frames"]:
        positions = np.asarray(frame["positions"])
        distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
        distances += np.eye(len(positions)) * 1e9
        relays = [i for i in range(config["num_uavs"]) if i not in (config["source"], config["destination"])]
        if (frame["safety_rejected"] or min(distances.ravel()) < config["min_separation"] - 1e-6
                or any(frame["travel"][i] > config["budget"] + 1e-6 for i in relays)):
            return False
    return True


def add_endpoints(row):
    frames = row["frames"][1:]
    lost = [not frame["source_dest_connected"] for frame in frames]
    longest = streak = 0
    for outage in lost:
        streak = streak + 1 if outage else 0
        longest = max(longest, streak)
    first_outage = next((frame["step"] for frame in frames if not frame["source_dest_connected"]), None)
    first_motion = next((d["step"] for d in row["decisions"] if d.get("commanded_motion")), None)
    row.update(disconnected_seconds=sum(lost), longest_outage_seconds=longest,
               no_olsr_route_seconds=sum(not frame["service_observation"]["route"]["success"] for frame in frames),
               motion_starts=sum(d["motion_control"]["started"] for d in row["decisions"]),
               safety_blocks=sum(d["motion_control"]["safety_blocked"] for d in row["decisions"]),
               first_motion_step=first_motion, first_outage_step=first_outage)


def analyze(protocol_path, run_root, output):
    phase, report, rows = load_traces(protocol_path, run_root, loader=load)
    out = reserve(output)
    integrity = {row["cell_id"]: route_integrity(row) for row in rows}
    bg = {row["cell_id"]: background(row) for row in rows}
    checks = dict(route_integrity_ok=all(i["ok"] for i in integrity.values()),
                  flags_ok=all(recorded_command(row)[-len(arm_flags(row["policy"])):] == arm_flags(row["policy"]) for row in rows),
                  transmissions_ok=all(transmissions(row) is not None for row in rows),
                  background_ok=all(value is not None for value in bg.values()),
                  no_startup_blackout=not any(startup_blackout(row) for row in rows),
                  safety_ok=all(safety(row) for row in rows),
                  route_only_stationary=all(row["total_relay_travel_m"] < 1e-6 for row in rows if row["policy"] == "route-only"),
                  forecasts_ok=all(forecast_trace_ok(dict(row, policy=row["policy"].replace("motion", "route")),
                                                    phase["selection_sha256"]) for row in rows))
    for row in rows:
        add_endpoints(row)
        if checks["background_ok"]:
            rx, sent = bg[row["cell_id"]]
            row["background_pdr"] = rx / sent
    data = blocks(rows, phase["policies"], regimes=V9.regimes)
    static = blocks(rows, phase["policies"], context="static", regimes=V9.regimes)
    pairs = [("gnn-motion", "reactive-motion"), ("gnn-motion", "mlp-motion")]
    pairs += [(arm, "route-only") for arm in ("gnn-motion", "reactive-motion", "mlp-motion")]
    metrics = ["actual_udp_pdr", "disconnected_seconds", "longest_outage_seconds", "no_olsr_route_seconds",
               "total_relay_travel_m", "motion_starts"]
    if checks["background_ok"]:
        metrics.append("background_pdr")
    result = dict(schema=V9.schema(phase["phase"] + "-analysis"), freeze_sha256=phase["_freeze_sha256"],
                  selection_sha256=phase["selection_sha256"], planned_cells=report["planned_cells"], ok_cells=report["ok_cells"],
                  technical=checks, route_integrity=integrity,
                  descriptive=[contrast(data, a, b, metric) for a, b in pairs for metric in metrics],
                  static_descriptive=[contrast(static, "gnn-motion", "reactive-motion", metric) for metric in metrics],
                  bootstrap=dict(samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED,
                                 unit="gradual scenario block, all four arms; equal n5/n7"),
                  rows=[{k: v for k, v in row.items() if k not in ("frames", "decisions")} for row in rows])
    if phase["phase"] == "validation":
        challenged = sum(row["disconnected_seconds"] >= 5 for row in rows
                         if row["policy"] == "route-only" and row["context"] == "gradual")
        criteria = {key: dict(passed=value) for key, value in checks.items()}
        criteria["partition_challenge"] = dict(passed=challenged >= 8, qualifying_baseline_episodes=challenged, required=8)
        go = all(item["passed"] for item in criteria.values())
        result["go"] = dict(go=go, criteria=criteria)
        write_json(out / "go.json", dict(schema=V9.schema("go"), go=go, criteria=criteria,
                                          validation_freeze_sha256=phase["_freeze_sha256"]))
    else:
        sequence, gate = [], True
        for metric, a, b in STEPS:
            comparison = contrast(data, a, b, metric)
            success = comparison["ci95"] is not None and comparison["ci95"][0] > 0
            sequence.append(dict(name=f"{metric}_{a}_vs_{b}", success=success, interpretable=gate, contrast=comparison))
            gate = gate and success
        result["fixed_sequence"] = sequence
    write_json(out / "analysis.json", result)
    write_json(out / "manifest.json", dict(files={path.name: sha256(path) for path in out.iterdir() if path.is_file()}))
    print(json.dumps({"phase": phase["phase"], "go": result.get("go"), "fixed_sequence": result.get("fixed_sequence")}, indent=2))
    return result
