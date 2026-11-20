"""Run an explicitly frozen development matrix; never select or tune cases.

Uses the existing stock diagnostics and service counters without simulator edits.
An existing output path is always refused, including an empty directory.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import diagnose_routing_service as diagnostics

STOCK_LOG = "ArpL3Protocol=level_all|prefix_time|prefix_node:ArpCache=level_all|prefix_time|prefix_node:WifiPhy=level_all|prefix_time|prefix_node:PhyEntity=level_debug|prefix_time|prefix_node"


def validate(protocol):
    if not isinstance(protocol, dict):
        raise ValueError("Protocol must be an object")
    if protocol.get("schema") != "prevention-v2-development-investigation-v1":
        raise ValueError("Expected explicit development investigation schema")
    cases = protocol.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 32:
        raise ValueError("Require 1..32 frozen cases")
    names = set()
    for case in cases:
        allowed = {"name", "num_uavs", "snapshots", "seed", "scenario", "spacing_fraction",
                   "app_rate_kbps", "warmup_s", "source", "destination", "mission_start_step",
                   "mission_speed_mps", "layout_seed", "random_layout", "packet_size_bytes",
                   "external_hold", "stock_logging", "channel_logging", "non_unicast_mode", "recovery_policy"}
        if not isinstance(case, dict) or set(case) - allowed:
            raise ValueError("Case must be an object containing only supported settings")
        name = case.get("name", "")
        if not diagnostics.re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."} or name in names:
            raise ValueError("Unsafe or duplicate case name")
        names.add(name)
        for field, lo, hi in (("num_uavs", 2, 12), ("snapshots", 2, 241), ("seed", 1, 2147483647)):
            value = case.get(field)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"Invalid {field}")
        for field, lo, hi in (("spacing_fraction", 0, 1), ("app_rate_kbps", 0, 256), ("warmup_s", 0, 60)):
            value = case.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo < value <= hi:
                raise ValueError(f"Invalid {field}")
        if case["spacing_fraction"] == 1:
            raise ValueError("Spacing must be below nominal range")
        if case.get("scenario") not in {"routing-sanity", "relay-prevention"}:
            raise ValueError("Unsupported diagnostic scenario")
        if case["scenario"] == "relay-prevention" and case["num_uavs"] < 3:
            raise ValueError("Prevention requires three nodes")
        external = case.get("external_hold", False)
        if type(external) is not bool or (external and case["scenario"] != "relay-prevention"):
            raise ValueError("External hold requires explicit prevention scenario")
        if case.get("recovery_policy", "none") != "none":
            raise ValueError("Investigation permits no controller")
        for key, default in (("source", 0), ("destination", case["num_uavs"] - 1)):
            value = case.get(key, default)
            if type(value) is not int or not 0 <= value < case["num_uavs"]:
                raise ValueError("Invalid source/destination")
        if case.get("source", 0) == case.get("destination", case["num_uavs"] - 1):
            raise ValueError("Endpoints must differ")
        speed = case.get("mission_speed_mps", 3)
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed) or not 0 <= speed <= 3:
            raise ValueError("Invalid mission speed")
        if type(case.get("mission_start_step", 10)) is not int or case.get("mission_start_step", 10) < 0:
            raise ValueError("Invalid mission start")
        if type(case.get("layout_seed", 1)) is not int or not 0 <= case.get("layout_seed", 1) < 2**31:
            raise ValueError("Invalid layout seed")
        if type(case.get("random_layout", False)) is not bool:
            raise ValueError("Random layout flag must be boolean")
        if case.get("non_unicast_mode") not in {None, "ErpOfdmRate6Mbps"}:
            raise ValueError("Only the named broadcast-mode diagnostic is permitted")
        if type(case.get("stock_logging", False)) is not bool:
            raise ValueError("Stock logging flag must be boolean")
        if type(case.get("channel_logging", False)) is not bool:
            raise ValueError("Channel logging flag must be boolean")
        if type(case.get("packet_size_bytes", 512)) is not int or not 64 <= case.get("packet_size_bytes", 512) <= 512:
            raise ValueError("Invalid packet size")
    return protocol


def execute(args):
    protocol = validate(json.loads(Path(args.protocol).read_text()))
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Output path already exists; preserve every earlier attempt")
    prefix = ["wsl", "-d", args.wsl_distribution, "--exec"] if args.wsl_distribution else []
    settings = SimpleNamespace(**vars(args), spacing_fraction=.55)
    plans = diagnostics.plans(settings, prefix)
    for plan in plans:
        case = plan["case"]
        plan["configuration"]["enableServiceObservation"] = "true"
        plan["configuration"]["appPacketSize"] = case.get("packet_size_bytes", 512)
        if case.get("non_unicast_mode"):
            plan["configuration"]["ns3::WifiRemoteStationManager::NonUnicastMode"] = case["non_unicast_mode"]
        if case.get("external_hold"):
            plan["configuration"]["recoveryPolicy"] = "external"
        plan["command"] = prefix + [args.binary] + [f"--{k}={v}" for k, v in plan["configuration"].items()]
        if case.get("stock_logging"):
            logs = STOCK_LOG + (":YansWifiChannel=level_all|prefix_time|prefix_node" if case.get("channel_logging") else "")
            plan["command"] = prefix + ["env", "NS_LOG=" + logs, args.binary] + [f"--{k}={v}" for k, v in plan["configuration"].items()]
    output.mkdir(parents=True, exist_ok=False)
    diagnostics.write_json(output / "plan.json", {"arguments": vars(args), "runs": plans})
    with (output / "protocol.json").open("xb") as stream:
        stream.write(Path(args.protocol).read_bytes())
    identity = diagnostics.provenance(settings, prefix)
    identity["investigation_runner_sha256"] = diagnostics.digest(__file__)
    diagnostics.write_json(output / "provenance.json", identity)
    source = output / "source_snapshot"
    source.mkdir()
    for path in (diagnostics.ROOT / "simulation/ns3").iterdir():
        if path.suffix in {".cc", ".h"} or path.name == "CMakeLists.txt":
            shutil.copyfile(path, source / path.name)
    shutil.copyfile(Path(__file__), source / Path(__file__).name)
    all_ok = True
    for plan in plans:
        folder = output / plan["name"]
        folder.mkdir()
        diagnostics.write_json(folder / "command.json", plan)
        status = {"success": False}
        try:
            if plan["case"].get("external_hold"):
                status["returncode"] = external_hold(plan, folder, args.timeout, protocol)
            else:
                with (folder / "stdout.log").open("x") as stdout, (folder / "stderr.log").open("x") as stderr:
                    result = subprocess.run(plan["command"], stdout=stdout, stderr=stderr, timeout=args.timeout)
                status["returncode"] = result.returncode
            if status["returncode"] == 0:
                summary = diagnostics.summarize(folder / "simulator")
                summary["service_final_accounting"] = json.loads((folder / "simulator/service_final_accounting.json").read_text())
                diagnostics.write_json(folder / "summary.json", summary)
                status["success"] = True
        except Exception as error:
            status["error"] = repr(error)
            process_status = folder / "process-status.json"
            if process_status.exists():
                status["returncode"] = json.loads(process_status.read_text())["returncode"]
        all_ok &= status["success"]
        diagnostics.write_json(folder / "status.json", status)
        print(json.dumps({"case": plan["name"], **status}), flush=True)
    diagnostics.write_json(output / "status.json", {"success": all_ok, "runs": len(plans)})
    return int(not all_ok)


def external_hold(plan, folder, timeout, protocol):
    """Existing live bridge, zero actions only; retain every measured frame."""
    sys.path.insert(0, str(diagnostics.ROOT))
    from src.control.bridge import SimulatorBridge
    from src.control.prevention.config import PreventionConfig
    case = plan["case"]
    config = PreventionConfig(num_uavs=case["num_uavs"], steps=case["snapshots"]-1,
        source=case.get("source", 0), destination=case.get("destination", case["num_uavs"]-1),
        comm_range=protocol.get("comm_range_m", 243),
        initial_spacing_fraction=case["spacing_fraction"])
    bridge = SimulatorBridge(plan["command"], config, folder / "stderr.log", min(60, timeout), stdout_path=folder / "stdout.log")
    deadline = time.monotonic() + timeout
    try:
        with (folder / "frames.jsonl").open("x") as stream:
            for step in range(case["snapshots"]):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Frozen case wall-clock deadline exceeded")
                bridge.timeout_s = min(60, remaining)
                frame = bridge.next_frame(step)
                stream.write(json.dumps(frame, allow_nan=False) + "\n")
                if step+1 < case["snapshots"]:
                    bridge.send_action(step, [0.] * (3*len(config.relay_ids)))
    finally:
        try:
            bridge.close()
        finally:
            diagnostics.write_json(folder / "process-status.json", {"returncode": bridge.process.returncode})
    return bridge.process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--wsl-distribution")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    # Defaults are never used for validated frozen cases; required by reused helpers.
    args.seed = 71001
    args.snapshots = 31
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Timeout must be finite and positive")
    try:
        return execute(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
