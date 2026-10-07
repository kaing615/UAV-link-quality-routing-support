"""Check actual delivery and forwarding on a fully connected three-node topology."""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


def run(binary: str, root: Path, name: str, paths: list[str]) -> dict:
    output = root / name
    output.mkdir()
    plan = output / "plan.csv"
    with plan.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "source", "destination", "route_found", "route_path", "route_cost"])
        for step, path in enumerate(paths):
            writer.writerow([step, 0, 2, int(bool(path)), path, 0])
    completed = subprocess.run(
        [
            binary,
            "--numUavs=3",
            "--sourceId=0",
            "--destId=2",
            "--timeSteps=8",
            "--xMax=1",
            "--yMax=1",
            "--zMin=100",
            "--zMax=101",
            "--commRange=1000",
            "--rwpSpeedMin=0.001",
            "--rwpSpeedMax=0.002",
            "--warmup=2",
            "--enableAnim=false",
            "--enableDataFlow=true",
            "--routingStrategy=hop",
            "--appRateKbps=64",
            f"--routePlan={plan}",
            f"--outputDir={output}",
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    with (output / "closed_loop_metrics.csv").open() as handle:
        return next(csv.DictReader(handle))


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        blocked = run(sys.argv[1], root, "blocked", [""] * 4)
        direct = run(sys.argv[1], root, "direct", ["0->2"] * 4)
        relayed = run(sys.argv[1], root, "relayed", ["0->1->2"] * 4)
        partial = run(sys.argv[1], root, "partial", ["", "", "0->2", "0->2"])
        assert int(blocked["rx_packets"]) == 0 and float(blocked["route_found_rate"]) == 0
        assert int(direct["rx_packets"]) > 0 and int(direct["forwarded_packets"]) == 0
        assert int(relayed["rx_packets"]) > 0 and int(relayed["forwarded_packets"]) > 0
        assert 0 < float(partial["pdr"]) < 0.6
        assert int(partial["offered_packets"]) > int(partial["tx_packets"])
        print("PASS: no-route drop, direct path, mandatory relay, offered-load PDR")


if __name__ == "__main__":
    main()
