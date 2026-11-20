"""Collect explicit prevention seed grids or score frozen-model held-out traces.

Collector/evaluator: --split train|calibration|test --policies hold reactive ...
  --layout-seeds 11001:11012 --radio-offset 100000 --binary PATH --output NEWDIR
For non-offset pairs use --scenario-grid JSON ([{"layout_seed":...,"radio_seed":...}]).
Offline HOLD reliability: --score-traces TEST/episodes/hold --model MODEL/predictor.json
  --split test --output NEWDIR. No simulator is started in scoring mode.
All output directories must be empty/new. Repeated --score-traces roots support
operational shards. The primary live comparison and offline reliability are
separate outputs. A complete comparison requires every requested grid cell.
"""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.control.bridge import make_launcher
from src.control.evaluation import runtime_versions, verify_build_manifest
from src.control.prevention.config import PreventionConfig
from src.control.prevention.controller import POLICIES
from src.control.prevention.evaluation import (
    evaluate_predictor, require_empty_output, run_grid, source_provenance, validate_grid,
)
from src.control.prevention.prediction import LearnedPredictor, load_traces, sha256, strict_json


def parse_seed_spec(spec):
    """Comma-separated ints and inclusive first:last ranges; duplicates refused."""
    result = []
    try:
        for part in spec.split(','):
            if ':' in part:
                first, last = map(int, part.split(':'))
                if last < first or last-first > 100000: raise ValueError('Invalid seed range')
                result.extend(range(first, last+1))
            else: result.append(int(part))
    except (TypeError, ValueError) as exc: raise ValueError('Invalid explicit seed specification') from exc
    if not result or len(result) != len(set(result)) or any(not 0 <= v < 2**31 for v in result):
        raise ValueError('Seeds must be unique integers in [0,2**31)')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'calibration', 'test'), required=True)
    parser.add_argument('--policies', nargs='+', choices=POLICIES)
    grid = parser.add_mutually_exclusive_group()
    grid.add_argument('--layout-seeds', help='Inclusive ranges and commas, e.g. 31001:31020')
    grid.add_argument('--scenario-grid', type=Path)
    parser.add_argument('--radio-offset', type=int, default=100000)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--wsl', help='WSL distribution; omit for native Linux')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--routing-delay', type=float, default=3.)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--fixed-layout', action='store_true', help='Default uses bounded random layout')
    parser.add_argument('--score-traces', action='append', type=Path, default=[])
    parser.add_argument('--bins', type=int, default=10)
    args = parser.parse_args(argv)
    require_empty_output(args.output)  # before binary/model loading or any launcher
    command = [sys.executable, '-m', 'scripts.analysis.evaluate_prevention',
               *(list(argv) if argv is not None else sys.argv[1:])]
    arguments = {k: str(v.resolve()) if isinstance(v, Path) else [str(p.resolve()) for p in v] if k == 'score_traces' else v
                 for k, v in vars(args).items()}
    provenance = dict(command=command, arguments=arguments, source_hashes=source_provenance(), versions=runtime_versions())
    model = LearnedPredictor.load(args.model) if args.model else None
    if args.model: provenance['model'] = dict(path=str(args.model.resolve()), sha256=sha256(args.model))
    if args.score_traces:
        if args.split != 'test' or model is None: raise ValueError('Offline reliability requires a model and held-out test split')
        if args.policies or args.layout_seeds or args.scenario_grid or args.binary:
            raise ValueError('Use either offline trace scoring or live seed-grid evaluation')
        episodes = [e for root in args.score_traces for e in load_traces(root, expected_split='test')]
        if len({e.episode_id for e in episodes}) != len(episodes): raise ValueError('Duplicate test episodes across score roots')
        result = evaluate_predictor(model, episodes, args.threshold, args.bins)
        result['provenance'] = provenance
        output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
        with (output/'reliability.json').open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        print(f'Offline reliability: {output / "reliability.json"}', flush=True)
        return result
    if not args.policies or not args.binary or not (args.layout_seeds or args.scenario_grid):
        raise ValueError('Live evaluation needs explicit --policies, --binary and a seed grid')
    if args.scenario_grid:
        rows = strict_json(args.scenario_grid)
        scenarios = [(row['layout_seed'], row['radio_seed']) for row in rows]
        provenance['scenario_grid_sha256'] = sha256(args.scenario_grid)
    else:
        scenarios = [(seed, seed+args.radio_offset) for seed in parse_seed_spec(args.layout_seeds)]
    validate_grid(args.policies, scenarios)
    binary = args.binary.resolve()
    if not binary.is_file(): raise FileNotFoundError(binary)
    manifest = verify_build_manifest(binary)
    if not manifest: raise ValueError('A validated simulator source.sha256 build manifest is required')
    provenance.update(binary_sha256=sha256(binary), build_manifest=manifest)
    launcher = make_launcher(binary, args.wsl)
    return run_grid(config=PreventionConfig(steps=args.steps, random_layout=not args.fixed_layout),
                    launcher=launcher, output=args.output, policies=args.policies, scenarios=scenarios,
                    predictor=model, threshold=args.threshold, routing_delay_s=args.routing_delay,
                    split=args.split, provenance=provenance)


if __name__ == '__main__':
    result = main()
    if result.get('paired_valid') is False: raise SystemExit(2)
