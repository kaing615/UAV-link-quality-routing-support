"""Guard diagnostic input/output boundaries; optional live stock binary seam."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analysis/investigate_prevention_v2.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("investigate_v2", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def protocol():
    return {"schema": "prevention-v2-development-investigation-v1", "cases": [{
        "name": "compact", "num_uavs": 2, "snapshots": 3, "seed": 71001,
        "scenario": "routing-sanity", "spacing_fraction": .15,
        "app_rate_kbps": 256, "warmup_s": 10}]}


@pytest.mark.parametrize("field,value", [("name", "../escape"), ("seed", True),
    ("spacing_fraction", float("nan")), ("snapshots", 100000),
    ("recovery_policy", "external"), ("destination", 2), ("mission_speed_mps", -1),
    ("external_hold", "true"), ("stock_logging", "true"), ("non_unicast_mode", "invented"),
    ("pretend_supported_flag", 5)])
def test_malformed_protocol_refused(field, value):
    data = protocol()
    data["cases"][0][field] = value
    with pytest.raises(ValueError):
        runner.validate(data)


def test_existing_output_refused_before_binary_launch(tmp_path):
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps(protocol()))
    output = tmp_path / "already-there"
    output.mkdir()
    sentinel = output / "keep"
    sentinel.write_bytes(b"preserve\x00earlier evidence")
    result = subprocess.run([sys.executable, str(SCRIPT), "--protocol", str(config),
        "--output", str(output), "--binary", "must-never-be-launched"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "Output path already exists" in result.stderr
    assert sentinel.read_bytes() == b"preserve\x00earlier evidence"
    assert sorted(p.name for p in output.iterdir()) == ["keep"]


@pytest.mark.skipif(not os.environ.get("UAV_INVESTIGATION_LIVE_BINARY"), reason="Explicit live binary required")
@pytest.mark.parametrize("external", [False, True])
def test_real_stock_binary_service_seam(tmp_path, external):
    config = tmp_path / "protocol.json"
    data = protocol()
    if external:
        data['cases'][0].update(num_uavs=3, scenario='relay-prevention', external_hold=True, mission_start_step=30)
    config.write_text(json.dumps(data))
    output = tmp_path / "live"
    command = [sys.executable, str(SCRIPT), "--protocol", str(config), "--output", str(output),
               "--binary", os.environ["UAV_INVESTIGATION_LIVE_BINARY"]]
    if os.environ.get("UAV_INVESTIGATION_WSL"):
        command += ["--wsl-distribution", os.environ["UAV_INVESTIGATION_WSL"]]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "compact/summary.json").read_text())
    counts = summary["service_final_accounting"]
    assert counts["offered_packets"] == counts["source_tx_packets"] > 0
    assert counts["sink_rx_packets"] == counts["flowmonitor_rx_packets"] > 0
    assert (output / "compact/simulator/directed_probes_v1.csv").is_file()
    assert (output / "compact/simulator/route_traversal_v1.csv").is_file()
    if external:
        frames = [json.loads(x) for x in (output / "compact/frames.jsonl").read_text().splitlines()]
        assert len(frames) == 3
        assert sum(f['service_observation']['window']['sink_rx_packets'] for f in frames) == counts['sink_rx_packets']
        assert all(sum(f['travel']) == 0 for f in frames)
