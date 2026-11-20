"""Focused routing diagnostics contracts; live checks use an explicit WSL binary."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("diagnose", ROOT / "scripts/analysis/diagnose_routing_service.py")


class RunnerContract(unittest.TestCase):
    def test_dry_run_explicit_grid(self):
        command = [sys.executable, str(spec.origin), "--binary", "/explicit/sim", "--output", "unused",
                   "--seed", "131002", "--cases", "static2,static3,static8,static12,moving8", "--dry-run"]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        plans = json.loads(result.stdout)["runs"]
        self.assertEqual(len(plans), 5)
        for plan in plans:
            self.assertIsInstance(plan["command"], list)
            self.assertIn("--enableRoutingDiagnostics=true", plan["command"])
            self.assertIn("--sourceId=0", plan["command"])


@unittest.skipUnless(os.environ.get("UAV_DIAGNOSTICS_BINARY"), "explicit live binary required")
class LiveDiagnostics(unittest.TestCase):
    def test_all_partial_traversal_failure_reasons(self):
        source = r'''
#include <iostream>
#include "routing-diagnostics.h"
int main() {
    auto emit=[](const auto& r) {
        std::cout << r.reason << "," << r.failing << ",";
        for (auto node:r.path) std::cout << node << ":";
        std::cout << "\n";
    };
    emit(TraverseDiagnosticRoute(0,2,3,[](uint32_t n){return DiagnosticRouteStep{true,true,n+1};}));
    emit(TraverseDiagnosticRoute(0,2,3,[](uint32_t n){return DiagnosticRouteStep{n==0,true,1};}));
    emit(TraverseDiagnosticRoute(0,2,3,[](uint32_t n){return DiagnosticRouteStep{true,n==0,1};}));
    emit(TraverseDiagnosticRoute(0,2,3,[](uint32_t n){return DiagnosticRouteStep{true,true,n==0?1:-1};}));
    emit(TraverseDiagnosticRoute(0,2,3,[](uint32_t n){return DiagnosticRouteStep{true,true,n==0?1:0};}));
    emit(TraverseDiagnosticRoute(0,2,1,[](uint32_t n){return DiagnosticRouteStep{true,true,n+1};}));
}
'''
        prefix=["wsl","-d","Ubuntu","--exec"] if os.name=="nt" else []
        def native(path):
            if os.name!="nt": return str(path)
            return subprocess.check_output(prefix+["wslpath","-a",str(path).replace("\\","/")],text=True).strip()
        with tempfile.TemporaryDirectory() as folder:
            executable=native(Path(folder)/"traversal-test")
            command=prefix+["g++","-std=c++17","-DUAV_DIAGNOSTICS_TRAVERSAL_ONLY","-x","c++","-I",native(ROOT/"simulation/ns3"),"-o",executable,"-"]
            result=subprocess.run(command,input=source,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            result=subprocess.run(prefix+[executable],text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.splitlines(), ["success,-1,0:1:2:","missing_protocol,1,0:1:",
                "missing_entry,1,0:1:","unknown_next,1,0:1:","loop,0,0:1:0:","hop_limit,1,0:1:"])

    def test_runner_preserves_failed_process_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"failed"
            command=[sys.executable, str(spec.origin), "--binary", "/bin/false", "--output", str(output),
                     "--seed", "131002", "--cases", "static2"]
            if os.name=="nt": command += ["--wsl-distribution", "Ubuntu"]
            self.assertEqual(subprocess.run(command+["--dry-run"],capture_output=True).returncode,0)
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertEqual(json.loads((output/"static2/status.json").read_text())["returncode"],1)
            for name in ("command.json","stdout.log","stderr.log"):
                self.assertTrue((output/"static2"/name).is_file())
            self.assertTrue(json.loads((output/"provenance.json").read_text())["runtime_sha256"])
            before={str(p.relative_to(output)):p.read_bytes() for p in output.rglob("*") if p.is_file()}
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)
            self.assertEqual(before,{str(p.relative_to(output)):p.read_bytes() for p in output.rglob("*") if p.is_file()})

    def run_sim(self, output, enabled=True, scenario="routing-sanity", n=3, extra=(), binary=None):
        output.mkdir(parents=True)
        prefix = ["wsl", "-d", "Ubuntu", "--"] if os.name == "nt" else []
        def linux_path(path):
            if os.name != "nt": return str(path)
            return subprocess.check_output(prefix + ["wslpath", "-a", str(path).replace("\\", "/")], text=True).strip()
        args = [binary or os.environ["UAV_DIAGNOSTICS_BINARY"], f"--outputDir={linux_path(output)}",
                f"--scenarioMode={scenario}", f"--numUavs={n}", "--sourceId=0", f"--destId={n-1}",
                "--seed=131002", "--timeSteps=6", "--enableDataFlow=true", "--enableAnim=false",
                "--recoveryPolicy=none",
                "--xMax=1800", "--yMax=600", "--preventionMissionStartStep=100", *extra]
        if binary is None: args.append(f"--enableRoutingDiagnostics={str(enabled).lower()}")
        result = subprocess.run(prefix + args, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return args

    def rows(self, path):
        with path.open(newline="") as stream: return list(csv.DictReader(stream))

    def test_static_schema_horizon_geometry_drops_and_observer_equality(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            self.run_sim(base / "off", False)
            args = self.run_sim(base / "on")
            for name in ("nodes.csv", "edges.csv", "traffic_log.csv", "closed_loop_metrics.csv", "flowmon.xml"):
                if (base / "off" / name).exists():
                    self.assertEqual((base / "off" / name).read_bytes(), (base / "on" / name).read_bytes(), name)
            routes = self.rows(base / "on/routing_entries_v1.csv")
            self.assertEqual({(int(r["time"]), int(r["node_id"])) for r in routes},
                             {(t,n) for t in range(6) for n in range(3)})
            traversals = self.rows(base / "on/route_traversal_v1.csv")
            self.assertEqual(len(traversals), 6)
            self.assertTrue(all(r["reason"] in {"success", "missing_protocol", "missing_entry", "unknown_next", "loop", "hop_limit"} for r in traversals))
            nodes = self.rows(base / "on/nodes.csv")
            for r in nodes:
                self.assertAlmostEqual(float(r["x"]), 900 + (int(r["node_id"])-1)*.55*243, places=3)
                self.assertEqual(float(r["y"]), 300)
                self.assertEqual(float(r["z"]), 100)
                self.assertEqual(float(r["speed"]), 0)
            drops = self.rows(base / "on/drop_counters_v1.csv")
            self.assertGreater(sum(int(r["count"]) for r in drops), 0)
            self.assertTrue(all(r["traffic_class"] in {"data","probe","OLSR","ARP","unknown"} for r in drops))
            directed = self.rows(base / "on/directed_probes_v1.csv")
            self.assertEqual(len(directed), 6*3*2)
            self.assertTrue(all(int(r["phy_tx_probe_count"])>0 for r in directed))
            self.assertTrue(all(int(r["rx_probe_count"])<=int(r["phy_tx_probe_count"]) for r in directed))
            scenario = json.loads((base / "on/scenario.json").read_text())
            self.assertEqual(scenario["diagnostic_scenario_kind"], "static_diagnostic")
            # A rerun must refuse before overwriting any original artifact.
            before = (base / "on/nodes.csv").read_bytes()
            prefix = ["wsl", "-d", "Ubuntu", "--"] if os.name == "nt" else []
            self.assertNotEqual(subprocess.run(prefix + args, capture_output=True).returncode, 0)
            self.assertEqual(before, (base / "on/nodes.csv").read_bytes())

    def test_moving_scenarios_observer_equality(self):
        with tempfile.TemporaryDirectory() as folder:
            for scenario in ("relay-prevention", "relay-recovery"):
                base = Path(folder) / scenario
                for enabled in (False, True):
                    self.run_sim(base / str(enabled), enabled, scenario, 8, ("--recoveryLayoutSeed=31002",))
                for file in (base / "False").iterdir():
                    if file.name in {"scenario.json"}: continue
                    self.assertEqual(file.read_bytes(), (base / "True" / file.name).read_bytes(), file.name)

    def test_unconverged_missing_entry_and_converged_success_reasons(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/"early"
            self.run_sim(output, n=2, extra=("--warmup=0",))
            records=self.rows(output/"route_traversal_v1.csv")
            self.assertIn("missing_entry", {r["reason"] for r in records})
            for row in records:
                if row["reason"]=="missing_entry":
                    self.assertNotEqual(row["failing_node"], "")
                    self.assertNotEqual(row["partial_path"], "")
            output=Path(folder)/"converged"
            self.run_sim(output, n=2)
            records=self.rows(output/"route_traversal_v1.csv")
            self.assertIn("success", {r["reason"] for r in records})
            for row in records:
                if row["reason"]=="success":
                    self.assertEqual(row["failing_node"], "")
                    self.assertEqual(row["partial_path"], "0->1")

    @unittest.skipUnless(os.environ.get("UAV_DIAGNOSTICS_BEFORE_BINARY"), "explicit before binary required")
    def test_before_binary_default_scientific_outputs_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            for scenario in ("relay-prevention", "relay-recovery"):
                base=Path(folder)/scenario
                self.run_sim(base/"before", False, scenario, 8, binary=os.environ["UAV_DIAGNOSTICS_BEFORE_BINARY"])
                self.run_sim(base/"after", False, scenario, 8)
                for file in (base/"before").iterdir():
                    if file.name=="scenario.json": continue
                    self.assertEqual(file.read_bytes(), (base/"after"/file.name).read_bytes(),file.name)


if __name__ == "__main__": unittest.main()
