"""Fit the prevention logistic model using disjoint real train/calibration traces.

Example (from repository root):
  python -m scripts.train.train_prevention_predictor --train-root results/train \
      --calibration-root results/calibration --output results/model
Output is a fresh directory containing predictor.json and training_report.json.
No test data, threshold search, unsafe pickle or additional ML package is used.
"""
import argparse
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.control.prevention.evaluation import require_empty_output, source_provenance
from src.control.prevention.prediction import fit_predictor, load_traces, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-root', type=Path, required=True)
    parser.add_argument('--calibration-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Empty/new output directory')
    parser.add_argument('--l2', type=float, default=.01, help='Predeclared logistic L2; never tune on test')
    args = parser.parse_args(argv)
    require_empty_output(args.output)
    start = time.monotonic()
    train = load_traces(args.train_root, expected_split='train')
    calibration = load_traces(args.calibration_root, expected_split='calibration')
    # Roots may contain operational shards; every manifest/episode is validated.
    model = fit_predictor(train, calibration, l2=args.l2)
    provenance = dict(command=[sys.executable, '-m', 'scripts.train.train_prevention_predictor',
                               *(list(argv) if argv is not None else sys.argv[1:])],
                      arguments={k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      source_hashes=source_provenance())
    model.artifact['provenance'] = provenance
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    model.save(output/'predictor.json')
    report = dict(schema='prevention-training-v1', model_sha256=sha256(output/'predictor.json'),
                  train_episodes=len(train), calibration_episodes=len(calibration),
                  samples={h: {k: v for k, v in m.items() if 'samples' in k or 'positive' in k}
                           for h, m in model.artifact['models'].items()},
                  elapsed_s=time.monotonic()-start, provenance=provenance,
                  limitations='Training/calibration labels depend on collection policy evolution. Hold/reactive mixture is not passive counterfactual truth. No test fitting or threshold selection.')
    with (output/'training_report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f'Logistic predictor: {output / "predictor.json"}', flush=True)
    return report


if __name__ == '__main__': main()
