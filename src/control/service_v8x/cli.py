"""Exploratory v8x entry points; same collector as v8."""
import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="service-v8x")
    commands = parser.add_subparsers(dest="command", required=True)
    seeds = commands.add_parser("seed-plan")
    seeds.add_argument("--historical", nargs="+", type=Path, required=True)
    seeds.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--phase", choices=("validation", "test"), required=True)
    for name in ("seed-plan", "output", "model", "reference-model", "selection"):
        freeze.add_argument("--" + name, type=Path, required=True)
    for name in ("inherits", "go"):
        freeze.add_argument("--" + name, type=Path)
    freeze.add_argument("--launcher", nargs="+", required=True)
    freeze.add_argument("--ns3-identity", required=True)
    freeze.add_argument("--compiler-identity", required=True)
    run = commands.add_parser("run-case")
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--cell-id", required=True)
    run.add_argument("--attempt", type=int, choices=(0,), required=True)
    run.add_argument("--output", type=Path, required=True)
    for name in ("validate", "analyze"):
        command = commands.add_parser(name)
        for field in ("protocol", "run-root", "output"):
            command.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from ..service_v2.protocol import write_json
        from ..service_v3 import protocol as p3
        from .study import V8X as V8, load
        if args.command == "seed-plan":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, p3.make_plan(args.historical, V8))
            p3.load_plan(args.output, V8)
        elif args.command == "freeze":
            p3.freeze(args.phase, args.seed_plan, args.output, args.model, args.reference_model, args.launcher,
                      args.ns3_identity, args.compiler_identity, args.selection, args.inherits, args.go, study=V8)
        elif args.command == "run-case":
            from .study import run_case
            result = run_case(args.protocol, args.cell_id, args.attempt, args.output)
            return 0 if result["status"] == "ok" else 1
        elif args.command == "validate":
            if args.output.exists():
                raise FileExistsError(str(args.output))
            from ..service_v2.study import validate_collection
            result = validate_collection(args.protocol, args.run_root, loader=load)
            write_json(args.output, result)
            return 0 if result["complete"] else 1
        elif args.command == "analyze":
            from .analysis import analyze
            analyze(args.protocol, args.run_root, args.output)
        return 0
    except KeyboardInterrupt:
        print("Interrupted; partial artifacts retained", file=sys.stderr)
        return 130
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
