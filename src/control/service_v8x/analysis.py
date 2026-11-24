"""Descriptive paired contrasts for v8x (exploratory; no go/no-go, no fixed sequence)."""
import json

from ..service_v2.protocol import reserve, sha256, write_json
from ..service_v3.analysis import BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, blocks, contrast
from ..service_v3.study import load_traces
from ..service_v4.study import route_integrity
from ..service_v7.study import startup_blackout
from ..service_v8.study import forecast_trace_ok
from .study import ARMS, V8X, load

PAIRS = (("gnn-route", "lq-route"), ("lq-ewma02", "lq-route"), ("lq-ewma01", "lq-route"),
         ("lq-dwell15", "lq-route"), ("gnn-route", "lq-ewma02"), ("gnn-route", "lq-ewma01"),
         ("gnn-route", "lq-dwell15"))


def route_changes(row):
    return sum(d["route_control"]["reason"] in ("engage", "switch", "release", "replace_broken", "release_broken")
               for d in row["decisions"])


def analyze(protocol_path, run_root, output):
    phase, report, rows = load_traces(protocol_path, run_root, loader=load)
    out = reserve(output)
    integrity = {row["cell_id"]: route_integrity(row)["ok"] for row in rows}
    for row in rows:
        row["route_changes"] = route_changes(row)
        row["disconnected_seconds"] = sum(not f["source_dest_connected"] for f in row["frames"][1:])
    data = blocks(rows, ARMS, regimes=V8X.regimes)
    changes = {arm: sum(r["route_changes"] for r in rows if r["policy"] == arm) for arm in ARMS}
    result = dict(schema=V8X.schema("analysis"), note="exploratory, descriptive only",
                  freeze_sha256=phase["_freeze_sha256"], planned_cells=report["planned_cells"], ok_cells=report["ok_cells"],
                  route_integrity_ok=all(integrity.values()),
                  forecast_traces_ok=all(forecast_trace_ok(r, phase["selection_sha256"]) for r in rows),
                  startup_blackouts=[r["cell_id"] for r in rows if startup_blackout(r)],
                  route_changes_total=changes,
                  mean_pdr={arm: sum(r["actual_udp_pdr"] for r in rows if r["policy"] == arm) /
                            sum(r["policy"] == arm for r in rows) for arm in ARMS},
                  contrasts=[contrast(data, a, b, m) for a, b in PAIRS for m in ("actual_udp_pdr", "route_changes")],
                  bootstrap=dict(samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED, unit="gradual block, five arms"),
                  rows=[{k: v for k, v in r.items() if k not in ("frames", "decisions")} for r in rows])
    write_json(out / "analysis.json", result)
    write_json(out / "manifest.json", dict(files={p.name: sha256(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps({k: result[k] for k in ("ok_cells", "route_integrity_ok", "forecast_traces_ok", "startup_blackouts",
                                             "route_changes_total", "mean_pdr")}, indent=1))
    for c in result["contrasts"]:
        print(c["arm"], "-", c["comparator"], c["metric"], c["point"], c["ci95"])
    return result
