"""Explicit, preserved routing-service diagnostics. Review before drawing conclusions.

Example (first run with --dry-run):
  py scripts/analysis/diagnose_routing_service.py --binary /path/to/sim \
    --output reports/new-diagnostic --seed 71001 --cases static2,static3,static8,static12,moving8

Use --protocol to execute a separately frozen JSON matrix; no parameter selection
or causal inference is performed. All subprocesses receive argument arrays.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def summarize(output):
    """Keep flow delivery distinct from routing and overheard PHY observations."""
    edges = defaultdict(list)
    scenario = json.loads((output / "scenario.json").read_text())
    for row in rows(output / "edges.csv"):
        if int(row["connected"]): edges[int(row["time"])].append((int(row["src"]), int(row["dst"])))
    connected = 0
    for step in range(scenario["time_steps"]):
        graph = defaultdict(set)
        for a,b in edges[step]: graph[a].add(b); graph[b].add(a)
        reached = {scenario["source_id"]}; pending = list(reached)
        while pending:
            for neighbor in graph[pending.pop()]-reached:
                reached.add(neighbor); pending.append(neighbor)
        connected += scenario["dest_id"] in reached
    drops = Counter()
    for row in rows(output / "drop_counters_v1.csv"):
        drops["/".join(row[k] for k in ("layer", "reason", "traffic_class"))] += int(row["count"])
    metrics = rows(output / "closed_loop_metrics.csv") if (output / "closed_loop_metrics.csv").exists() else []
    probe_rows = rows(output / "directed_probes_v1.csv")
    # PHY TX is repeated for every receiver in the directed file: deduplicate.
    tx = {(r["time"],r["source"]): int(r["phy_tx_probe_count"]) for r in probe_rows}
    return {"snapshots": scenario["time_steps"], "probe_graph_source_dest_connected_windows": connected,
            "route_traversal_reasons": dict(Counter(r["reason"] for r in rows(output / "route_traversal_v1.csv"))),
            "flow_monitor_metrics": metrics, "drop_observations_by_layer_reason_class": dict(drops),
            "probe_phy_transmissions": sum(tx.values()), "directed_probe_receptions": sum(int(r["rx_probe_count"]) for r in probe_rows),
            "warning": "Drop traces overlap; PHY_rx includes overhearing. These are not data-flow lost packets."}


def plans(args, prefix):
    protocol = json.loads(Path(args.protocol).read_text()) if args.protocol else {}
    cases = protocol.get("cases")
    if cases is None:
        cases = []
        for name in args.cases.split(","):
            match = re.fullmatch(r"(static|moving)(2|3|8|12)", name)
            if not match: raise ValueError(f"Unsupported explicit smoke case: {name}")
            kind,n = match.groups(); n = int(n)
            cases.append({"name": name, "scenario": "routing-sanity" if kind=="static" else "relay-prevention",
                          "num_uavs": n, "source": 0, "destination": n-1, "seed": args.seed,
                          "snapshots": args.snapshots, "spacing_fraction": args.spacing_fraction,
                          "mission_start_step": 10})
    result=[]; names=set()
    for case in cases:
        name=case["name"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".",".."} or name in names:
            raise ValueError(f"Unsafe or duplicated case name: {name}")
        names.add(name)
        n=case["num_uavs"]; scenario=case["scenario"]
        if n<2 or scenario not in {"routing-sanity","relay-prevention"}:
            raise ValueError("Runner cases require n>=2 and static sanity or initially connected prevention hold")
        if case.get("recovery_policy","none")!="none": raise ValueError("Diagnostic runner requires recovery_policy=none")
        target=(Path(args.output).resolve()/name/"simulator")
        output=str(target)
        if args.wsl_distribution:
            output=subprocess.check_output(prefix+["wslpath","-a",output.replace("\\","/")],text=True).strip()
        values={"scenarioMode":scenario,"numUavs":n,"sourceId":case.get("source",0),"destId":case.get("destination",n-1),
                "seed":case.get("seed",args.seed),"timeSteps":case.get("snapshots",args.snapshots),
                "warmup":case.get("warmup_s",10),"enableAnim":"false","enableDataFlow":"true",
                "enableRoutingDiagnostics":"true","routingStrategy":"olsr","recoveryPolicy":"none",
                "xMax":protocol.get("area",[1800,1200])[0],"yMax":protocol.get("area",[1800,1200])[1],
                "zMin":protocol.get("altitude_bounds_m",[50,150])[0],"zMax":protocol.get("altitude_bounds_m",[50,150])[1],
                "commRange":protocol.get("comm_range_m",243),"appRateKbps":case.get("app_rate_kbps",256),
                "appPacketSize":protocol.get("packet_size_bytes",512),"outputDir":output,
                "sanitySpacingFraction":case.get("spacing_fraction",args.spacing_fraction),
                "preventionSpacingFraction":case.get("spacing_fraction",args.spacing_fraction),
                "preventionMissionStartStep":case.get("mission_start_step",10),
                "preventionMissionSpeed":case.get("mission_speed_mps",3),
                "recoveryLayoutSeed":case.get("layout_seed",1),
                "recoveryRandomLayout":str(case.get("random_layout",False)).lower()}
        result.append({"name":name,"case":case,"configuration":values,
                       "command":prefix+[args.binary]+[f"--{k}={v}" for k,v in values.items()]})
    return result


def capture(command):
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    return {"command":command,"returncode":result.returncode,"stdout":result.stdout,"stderr":result.stderr}


def provenance(args,prefix):
    # Hash the executable and its resolved shared objects in the runtime where
    # they are actually loaded, including libns3, libc and the dynamic loader.
    binary_hash=capture(prefix+["sha256sum",args.binary])
    if binary_hash["returncode"]: raise ValueError(binary_hash)
    linkage=capture(prefix+["ldd",args.binary])
    libraries=sorted(set(re.findall(r"(?:=>\s+)?(/[^\s()]+)",linkage["stdout"])))
    return {"created_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,"host":platform.platform(),
            "binary_sha256":binary_hash,"linkage":linkage,
            "runtime_sha256":capture(prefix+["sha256sum",*libraries]) if libraries else None,
            "runtime_uname":capture(prefix+["uname","-a"]),
            "ns3_version":capture(prefix+[args.binary,"--PrintVersion"]),
            "git_head":capture(["git","-C",str(ROOT),"rev-parse","HEAD"]),
            "git_status":capture(["git","-C",str(ROOT),"status","--porcelain"]),
            "source_sha256":{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/"simulation/ns3/uav-olsr-dataset.cc",*sorted((ROOT/"simulation/ns3").glob("*.h")),ROOT/"simulation/ns3/CMakeLists.txt",Path(__file__)]},
            "protocol_sha256":digest(args.protocol) if args.protocol else None,
            "interpretation":"Mechanism diagnostics; historical seeds are not fresh heldout validation; no parameter selection."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--wsl-distribution")
    parser.add_argument("--protocol")
    parser.add_argument("--cases",default="static2,static3,static8,static12,moving8")
    parser.add_argument("--snapshots",type=int,default=31)
    parser.add_argument("--spacing-fraction",type=float,default=.55)
    parser.add_argument("--timeout",type=float,default=600)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    prefix=["wsl","-d",args.wsl_distribution,"--exec"] if args.wsl_distribution else []
    run_plans=plans(args,prefix)
    if args.dry_run:
        print(json.dumps({"runs":run_plans},indent=2)); return 0
    output=Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output must be empty/new; existing and failed runs are preserved")
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/"plan.json",{"arguments":vars(args),"runs":run_plans})
    if args.protocol: shutil.copyfile(args.protocol,output/"protocol.json")
    write_json(output/"provenance.json",provenance(args,prefix))
    for path in (ROOT/"simulation/ns3").glob("*"):
        if path.suffix in {".cc",".h"} or path.name=="CMakeLists.txt":
            target=output/"source_snapshot"/path.name; target.parent.mkdir(exist_ok=True); shutil.copyfile(path,target)
    failed=False
    for plan in run_plans:
        folder=output/plan["name"]; folder.mkdir()
        write_json(folder/"command.json",plan)
        status={"success":False}
        try:
            with (folder/"stdout.log").open("x") as stdout, (folder/"stderr.log").open("x") as stderr:
                process=subprocess.run(plan["command"],stdout=stdout,stderr=stderr,timeout=args.timeout)
            status["returncode"]=process.returncode
            if process.returncode==0:
                write_json(folder/"summary.json",summarize(folder/"simulator")); status["success"]=True
        except Exception as error:
            status["error"]=repr(error)
        failed |= not status["success"]
        write_json(folder/"status.json",status)
    write_json(output/"status.json",{"success":not failed,"runs":len(run_plans)})
    return int(failed)


if __name__=="__main__":
    raise SystemExit(main())
