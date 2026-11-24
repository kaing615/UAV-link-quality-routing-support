"""v9 frozen packages: retain v3 checks, bind the registered motion scenario."""
from pathlib import Path
from ..service_v2.protocol import sha256, strict_json, write_json
from ..service_v3.protocol import (V3, V2_MODEL_SHA256, load_plan, make_plan, strict_json_value)

def _cells(plan, phase, study=V3):
    from dataclasses import asdict
    from .study import scenario_config
    return [dict(s, cell_id=s['id']+'__'+p, policy=p, config=asdict(scenario_config(s)),
                 output_path=s['id']+'__'+p, attempts=[0]) for s in plan['phases'][phase] for p in study.policies[phase]]


def freeze(phase, seed_plan, output, model, reference_model, launcher, ns3_identity, compiler_identity,
           selection=None, inherits=None, go=None, study=V3):
    """Freeze one phase. validation needs the tuning selection; test also needs
    the validation package it inherits and that package's go decision."""
    from dataclasses import asdict
    import hashlib
    from ..service_v2.config import ServiceV2Config
    from ..service_v2.protocol import (ACCEPTED_REFERENCE_SOURCES, REFERENCE_SHA256, ROOT, reserve,
                                       runtime_identity, safe_path, source_paths)
    if phase not in study.counts: raise ValueError('Unknown phase')
    if (phase == 'tuning') != (selection is None): raise ValueError('Selection required exactly for validation/test')
    if (phase == 'test') != (inherits is not None and go is not None): raise ValueError('Test must inherit validation + go')
    out = reserve(output)
    plan = load_plan(seed_plan, study)
    if sha256(model) != V2_MODEL_SHA256: raise ValueError('Must reuse the formal v2 predictor unchanged')
    if sha256(reference_model) != REFERENCE_SHA256: raise ValueError('Exact reference model mismatch')
    for name, digest in ACCEPTED_REFERENCE_SOURCES.items():
        if sha256(safe_path(ROOT, name)) != digest: raise ValueError('Accepted legacy reference source changed: '+name)
    inherited = None
    if selection is not None:
        chosen = strict_json(selection)
        if chosen.get('schema') != study.schema('tuning-selection') or chosen.get('predictor_sha256') != V2_MODEL_SHA256:
            raise ValueError('Tuning selection schema/predictor')
    if phase == 'test':
        inherited = load(inherits, study)
        decision = strict_json(go)
        if (inherited['phase'] != 'validation' or decision.get('schema') != study.schema('go') or decision.get('go') is not True
                or decision.get('validation_freeze_sha256') != inherited['_freeze_sha256']
                or inherited['selection_sha256'] != sha256(selection)):
            raise ValueError('Test requires a go decision from the validation package with the same selection')
    from ..service_v2.prediction import ServicePredictor
    cells = _cells(plan, phase, study)
    if study.predictor_scenarios: ServicePredictor.load(model).validate_scenarios(cells)
    runtime = runtime_identity(dict(launcher=list(launcher), ns3_identity=ns3_identity, compiler_identity=compiler_identity))
    sources = {}
    for p in source_paths(ROOT):
        name = p.relative_to(ROOT).as_posix(); raw = p.read_bytes()
        target = out/'source'/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        sources[name] = dict(original_sha256=digest, executed_sha256=digest,
                             normalized_sha256=hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest(), transformation='identity')
    if inherited is not None and strict_json(Path(inherited['_path']).parent/'source-manifest.json') != sources:
        raise ValueError('Test source differs from the validation package')
    copies = [('seed-plan.json', seed_plan), ('reference-model.json', reference_model), ('new-model.json', model)]
    copies += [('selection.json', selection)] if selection is not None else []
    copies += [('go.json', go)] if go is not None else []
    for name, source in copies:
        (out/name).write_bytes(Path(source).read_bytes())
    write_json(out/'method.json', asdict(ServiceV2Config()))
    write_json(out/'source-manifest.json', sources)
    write_json(out/'runtime.json', runtime)
    protocol = dict(schema=study.schema('phase'), phase=phase, policies=list(study.policies[phase]), cells=cells,
        expected_episodes=len(cells), reference_model=REFERENCE_SHA256, new_model=V2_MODEL_SHA256,
        seed_plan_sha256=sha256(seed_plan), method_sha256=sha256(out/'method.json'),
        selection_sha256=sha256(selection) if selection is not None else None,
        inherited_validation_freeze_sha256=inherited['_freeze_sha256'] if inherited else None,
        runtime=runtime, source_root_at_freeze=str(ROOT), package_root=str(out.resolve()),
        protocol_path_at_freeze=str((out/'protocol.json').resolve()), working_directory=str((out/'source').resolve()),
        retry='no retries; stop and retain evidence on any cell failure', service_observation=True,
        run_argv=[runtime['python_executable'], '-m', study.cli_module, 'run-case',
                  '--protocol', str((out/'protocol.json').resolve()), '--cell-id', '{cell_id}',
                  '--attempt', '{attempt}', '--output', '{exclusive_attempt_output}'])
    write_json(out/'protocol.json', protocol)
    files = {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'freeze.json', dict(schema=study.schema('freeze'), files=files))
    load(out/'protocol.json', study)
    return out/'protocol.json'


def load(path, study=V3):
    from dataclasses import asdict
    from ..service_v2.config import ServiceV2Config
    from ..service_v2.protocol import REFERENCE_SHA256, safe_path
    path = Path(path).resolve(); root = path.parent
    freeze_record = strict_json(root/'freeze.json')
    if freeze_record.get('schema') != study.schema('freeze'): raise ValueError('Freeze schema')
    files = freeze_record['files']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual != set(files) | {'freeze.json'}: raise ValueError('Unlisted/missing frozen file')
    for name, digest in files.items():
        if sha256(safe_path(root, name)) != digest: raise ValueError('Frozen file mismatch: '+name)
    p = strict_json(path)
    if p.get('schema') != study.schema('phase') or p.get('phase') not in study.counts: raise ValueError('Phase schema')
    plan = load_plan(root/'seed-plan.json', study)
    if (sha256(root/'seed-plan.json') != p['seed_plan_sha256'] or p['policies'] != list(study.policies[p['phase']])
            or p['cells'] != _cells(plan, p['phase'], study) or p['expected_episodes'] != len(p['cells'])):
        raise ValueError('Cell inventory mismatch')
    if (p['reference_model'] != REFERENCE_SHA256 or sha256(root/'reference-model.json') != REFERENCE_SHA256
            or p['new_model'] != V2_MODEL_SHA256 or sha256(root/'new-model.json') != V2_MODEL_SHA256
            or strict_json(root/'method.json') != strict_json_value(asdict(ServiceV2Config()))
            or p['method_sha256'] != sha256(root/'method.json')): raise ValueError('Method/model mismatch')
    if p['runtime'] != strict_json(root/'runtime.json'): raise ValueError('Runtime manifest disagreement')
    has_selection = (root/'selection.json').is_file()
    if has_selection != (p['phase'] != 'tuning') or (has_selection and sha256(root/'selection.json') != p['selection_sha256']):
        raise ValueError('Selection binding mismatch')
    p['_path'] = str(path); p['_freeze_sha256'] = sha256(root/'freeze.json')
    p['supported_policies'] = study.all_policies
    return p


def strict_json_value(value):
    import json
    from ..service_v2.protocol import strict_loads
    return strict_loads(json.dumps(value))
