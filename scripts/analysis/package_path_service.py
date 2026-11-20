"""Freeze a turnkey source/binary/model/evidence snapshot in an exclusive archive."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import sys

if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.analysis.run_path_service_study import ROOT,inventory
from src.control.prevention.prediction import sha256


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=False)
    paths={ROOT/name for name in inventory()}
    paths|={ROOT/name for name in ['src/__init__.py','src/utils/__init__.py','src/utils/experiment.py','pyproject.toml',
                                'requirements-rl.txt','build/recovery/uav-olsr-dataset','build/recovery/source.sha256',
                                'reports/uav_prevention_pilot_2026-11-11/predictor.json']}
    paths|=set((ROOT/'scripts/analysis').glob('*path_service*.py'))
    paths|=set((ROOT/'tests').glob('test_path_service.py'))|{ROOT/'tests/cpp/test_service_observation_filter.cc'}
    paths.add(ROOT/'tests/test_bridge_stdout_shutdown.py')
    spec=ROOT/'.superpowers/sdd/2026-11-11-routing-service'
    paths.add(spec/'remote-handoff.json')
    paths.add(spec/'runtime-fix1-handoff.json')
    paths|={p for directory in ('task-3-runtime-fix1-evidence','task-3-runtime-fix1-before') for p in (spec/directory).rglob('*') if p.is_file()}
    paths|={p for p in spec.rglob('task-3*') if p.is_file() and 'task-3-before' not in p.parts}
    paths|={p for p in (spec/'task-3-frozen').rglob('*') if p.is_file()}
    paths|={p for p in (spec/'task-3-engineering').rglob('*') if p.is_file()}
    paths|={p for p in (ROOT/'reports/uav_path_service_engineering_2026-11-11_v3').rglob('*') if p.is_file()}
    paths|={p for p in (ROOT/'reports/uav_path_service_collector_engineering_2026-11-11').rglob('*') if p.is_file()}
    paths|={p for p in (ROOT/'reports/uav_path_service_runtime_fix1_collector_engineering_2026-11-11').rglob('*') if p.is_file()}
    paths={p for p in paths if p.is_file()}
    manifest={p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(paths)}
    with (args.output/'inventory.json').open('x',encoding='utf-8') as stream:json.dump(manifest,stream,indent=2)
    archive=args.output/'source.tar.gz'
    def permissions(info):
        info.mode=0o755 if info.name=='build/recovery/uav-olsr-dataset' or info.name.endswith('.sh') else 0o644
        return info
    with tarfile.open(archive,'x:gz') as tar:
        for p in sorted(paths):tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False,filter=permissions)
    (args.output/'archive.sha256').write_text(sha256(archive)+'  source.tar.gz\n',encoding='ascii')
    print(json.dumps(dict(archive=str(archive.resolve()),sha256=sha256(archive),files=len(paths),bytes=archive.stat().st_size)))

if __name__=='__main__':main()
