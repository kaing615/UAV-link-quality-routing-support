"""Strict opt-in entry points. Imports are lazy for standalone PDF rendering."""
import argparse
from pathlib import Path
import sys


def main(argv=None):
    parser=argparse.ArgumentParser(prog='service-v2')
    commands=parser.add_subparsers(dest='command',required=True)
    audit=commands.add_parser('seed-audit')
    for name in ('candidate-inventory','historical-root','output'): audit.add_argument('--'+name,type=Path,required=True)
    freeze=commands.add_parser('freeze')
    freeze.add_argument('--phase',choices=('train','calibration','development','pilot','formal'),required=True)
    for name in ('seed-plan','design','output','reference-model'): freeze.add_argument('--'+name,type=Path,required=True)
    freeze.add_argument('--model',type=Path)
    run=commands.add_parser('run-case')
    run.add_argument('--protocol',type=Path,required=True); run.add_argument('--cell-id',required=True)
    run.add_argument('--attempt',type=int,choices=(0,1),required=True); run.add_argument('--output',type=Path,required=True)
    fit=commands.add_parser('fit')
    for name in ('train-run','calibration-run','output'): fit.add_argument('--'+name,type=Path,required=True)
    for name in ('validate','analyze'):
        command=commands.add_parser(name)
        for field in ('protocol','run-root','output'): command.add_argument('--'+field,type=Path,required=True)
    report=commands.add_parser('report')
    for name in ('analysis','engineering','output'): report.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        from .protocol import strict_json,write_json,freeze_phase,audit_seed_inventory
        if args.command=='seed-audit':
            if args.output.exists(): raise FileExistsError(str(args.output))
            write_json(args.output,audit_seed_inventory(strict_json(args.candidate_inventory),args.historical_root))
        elif args.command=='freeze':
            freeze_phase(args.phase,args.seed_plan,args.design,args.output,args.model,args.reference_model)
        elif args.command=='run-case':
            from .study import collect_case
            result=collect_case(args.protocol,args.cell_id,args.attempt,args.output)
            if result['status']!='ok': return result['actual_exit'] if type(result['actual_exit']) is int and 0<result['actual_exit']<256 else 1
        elif args.command=='fit':
            if args.output.exists(): raise FileExistsError(str(args.output))
            from .study import load_service_episodes
            from .prediction import fit_service_predictor
            train=load_service_episodes(args.train_run,'train')
            calibration=load_service_episodes(args.calibration_run,'calibration')
            fit_service_predictor(train,calibration).save_exclusive(args.output)
        elif args.command=='validate':
            if args.output.exists(): raise FileExistsError(str(args.output))
            from .study import validate_collection
            result=validate_collection(args.protocol,args.run_root); write_json(args.output,result)
            return 0 if result['complete'] else 1
        elif args.command=='analyze':
            from .analysis import analyze_collection
            analyze_collection(args.protocol,args.run_root,args.output)
        elif args.command=='report':
            from .reporting import build_report
            build_report(args.analysis,args.engineering,args.output)
        return 0
    except KeyboardInterrupt:
        print('Interrupted; partial artifacts retained',file=sys.stderr); return 130
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr); return 1


if __name__=='__main__':
    raise SystemExit(main())
