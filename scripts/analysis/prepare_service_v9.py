"""Bind unchanged v8 predictors and fresh seeds to the motion protocol."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.control.service_v3.protocol import make_plan, load_plan
from src.control.service_v9.study import V9


def main():
    output = ROOT / "reports/uav_service_v9"
    output.mkdir(parents=True, exist_ok=True)
    smoke = output / "development-seeds.json"
    if not smoke.exists():
        with smoke.open("x", encoding="utf-8") as stream:
            json.dump(dict(scenarios=[dict(layout_seed=3599001, radio_seed=3699001),
                                      dict(layout_seed=3599002, radio_seed=3699002)]), stream, indent=2)
    old_plan = ROOT / "reports/uav_service_v8/seed-plan.json"
    previous = json.loads(old_plan.read_text())["historical_audit"]["files"]
    historical = [ROOT / path for path in previous] + [old_plan, smoke]
    plan = output / "seed-plan.json"
    if not plan.exists():
        with plan.open("x", encoding="utf-8") as stream:
            json.dump(make_plan(historical, V9), stream, indent=2)
    load_plan(plan, V9)
    source = ROOT / "reports/uav_service_v8/model/selection.json"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == "ad2e0741a0d7509e922bbd8c186cca0477ecd98ec67440c81b4c8cd12ebbe49f"
    selection = json.loads(source.read_text())
    selection.update(schema=V9.schema("tuning-selection"), source_v8_selection_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     reuse="identical frozen v8 weights; no refitting or outcome-based selection",
                     motion=dict(on=.70, off=.80, confirmation=2, recovery_confirmation=3, burst=10, cooldown=5, deadband_m=1.))
    target = output / "selection.json"
    if not target.exists():
        with target.open("x", encoding="utf-8") as stream:
            json.dump(selection, stream, indent=2, allow_nan=False)
    paths = list((ROOT / "src/control/service_v9").glob("*.py"))
    paths += [ROOT / path for path in ("scripts/analysis/prepare_service_v9.py", "scripts/analysis/smoke_service_v9.py",
                                      "orchestration/service_v9_queue.py", "docs/service_v9_protocol.md", "tests/test_service_v9.py")]
    paths += [plan, smoke, target]
    with tarfile.open(output / "source-bundle.tar.gz", "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    print(json.dumps({"historical_files": len(historical), "files": len(paths),
                      "selection_sha256": hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
