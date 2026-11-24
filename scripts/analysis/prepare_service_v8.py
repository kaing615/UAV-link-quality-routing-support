"""Create seed audit and a small additive source bundle for the existing runner."""
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.control.service_v3.protocol import make_plan, load_plan
from src.control.service_v8.study import V8


def main():
    output = ROOT / "reports/uav_service_v8"
    smoke = output / "development-seeds.json"
    if not smoke.exists():
        with smoke.open("x", encoding="utf-8") as stream:
            json.dump({"purpose": "development-only smoke; excluded from formal v8",
                       "scenarios": [{"layout_seed": 3399001, "radio_seed": 3499001},
                                     {"layout_seed": 3399002, "radio_seed": 3499002}]}, stream, indent=2)
    v7 = ROOT / "reports/uav_service_v7/seed-plan.json"
    previous = json.loads(v7.read_text())["historical_audit"]["files"]
    historical = [ROOT / path for path in previous] + [v7, smoke, output / "development/manifest.json"]
    plan_path = output / "seed-plan.json"
    if not plan_path.exists():
        with plan_path.open("x", encoding="utf-8") as stream:
            json.dump(make_plan(historical, V8), stream, indent=2)
    load_plan(plan_path, V8)
    paths = list((ROOT / "src/control/service_v8").glob("*.py"))
    paths += [ROOT / path for path in ("scripts/train/export_service_v8_data.py", "scripts/train/train_service_v8_gnn.py",
                "scripts/analysis/prepare_service_v8.py", "scripts/analysis/smoke_service_v8.py",
                "orchestration/service_v8_queue.py", "docs/service_v8_protocol.md", "tests/test_service_v8.py")]
    paths += [plan_path, smoke]
    bundle = output / "source-bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for path in paths:
            archive.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
    print(json.dumps({"plan": str(plan_path), "historical_files": len(historical),
                      "bundle": str(bundle), "files": len(paths)}))


if __name__ == "__main__":
    main()
