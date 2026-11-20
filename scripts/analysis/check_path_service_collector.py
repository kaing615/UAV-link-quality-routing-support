"""One tight positive-delivery local engineering case through reused real collector."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys

if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.control.bridge import make_launcher
from src.control.prevention.config import PreventionConfig
from src.control.prevention.prediction import LearnedPredictor,strict_json,sha256
from src.control.path_service.study import collect_case
from scripts.analysis.run_path_service_study import write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--wsl');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]; model=root/'reports/uav_prevention_pilot_2026-11-11/predictor.json'
    method=root/'.superpowers/sdd/2026-11-11-routing-service/task-3-frozen/method.json'
    protocol=dict(config=asdict(PreventionConfig(num_uavs=3,destination=2,steps=20,initial_spacing_fraction=.15)),model_sha256=sha256(model),method_sha256=sha256(method))
    row=collect_case(scenario=dict(regime='engineering_n3',num_uavs=3,initial_spacing_fraction=.15,layout_seed=31001,radio_seed=131001),
                     policy='path-service',protocol=protocol,launcher=make_launcher(args.binary.resolve(),args.wsl),predictor=LearnedPredictor.load(model),
                     output=args.output/'case',provenance=dict(scope='historical diagnostic engineering; no study/tuning'))
    write(args.output/'evidence.json',row,True)
    if row['status']!='ok' or row['application_received_packets']<=0: raise RuntimeError(row)
    print(dict(status=row['status'],offered=row['application_offered_packets'],received=row['application_received_packets'],elapsed_steps=row['elapsed_steps']))

if __name__=='__main__':main()
