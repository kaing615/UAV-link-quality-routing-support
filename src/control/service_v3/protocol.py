"""Service v3 seed plan (seeds audited disjoint from every v2 seed) and frozen phase packages.

See docs/service_v3_protocol.md. CLI: python -m src.control.service_v3.cli.
"""
from pathlib import Path
import re

from ..service_v2.protocol import _canonical_hash, _seeds, sha256, strict_json, write_json

SCHEMA = 'service-v3-seed-plan-v1'
# (static, gradual) scenarios per regime; tuning pairs share a seed pair.
COUNTS = {'tuning': (8, 8), 'validation': (4, 12), 'test': (4, 20)}
POLICIES = {
    'tuning': ('hold', 'service-v2', 'service-v2-no-forecast'),
    'validation': ('hold', 'service-v2', 'service-v2-no-forecast', 'service-v3-threshold', 'service-v3-gate'),
    'test': ('hold', 'service-v2', 'service-v2-no-forecast', 'service-v3-threshold', 'service-v3-gate'),
}
LAYOUT_BASE, RADIO_BASE = 2_400_000, 2_500_000
REGIMES, CONTEXTS = ('n4', 'n5'), ('static', 'gradual')


def build_phases():
    phases, k = {}, 0
    for phase, (static, gradual) in COUNTS.items():
        rows = []
        for regime in REGIMES:
            if phase == 'tuning':  # within-split pairing, as v2 train/calibration
                for i in range(static):
                    k += 1
                    for context in CONTEXTS:
                        rows.append(dict(id=f'{phase}-{regime}-{context}-{i+1:02d}', regime=regime, context=context,
                                         layout_seed=LAYOUT_BASE+k, radio_seed=RADIO_BASE+k))
                continue
            for context, n in zip(CONTEXTS, (static, gradual)):
                for i in range(n):
                    k += 1
                    rows.append(dict(id=f'{phase}-{regime}-{context}-{i+1:02d}', regime=regime, context=context,
                                     layout_seed=LAYOUT_BASE+k, radio_seed=RADIO_BASE+k))
        phases[phase] = rows
    return phases


def validate_phases(phases):
    if set(phases) != set(COUNTS): raise ValueError('v3 phases')
    owner = {}
    for phase, rows in phases.items():
        ids = set()
        for s in rows:
            if set(s) != {'id', 'regime', 'context', 'layout_seed', 'radio_seed'}: raise ValueError('Scenario fields')
            if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]*', s['id']) or s['id'] in ids: raise ValueError('Scenario id')
            ids.add(s['id'])
            if s['regime'] not in REGIMES or s['context'] not in CONTEXTS: raise ValueError('Regime/context')
            for key in ('layout_seed', 'radio_seed'):
                if type(s[key]) is not int or not 0 <= s[key] < 2**31: raise ValueError('Seed range')
            pair = (s['layout_seed'], s['radio_seed'])
            owner.setdefault(pair, []).append((phase, s['regime'], s['context']))
        for regime in REGIMES:
            for context, n in zip(CONTEXTS, COUNTS[phase]):
                if sum(s['regime'] == regime and s['context'] == context for s in rows) != n:
                    raise ValueError('Wrong phase cardinality: '+phase)
    layout, radio = {}, {}
    for (lay, rad), uses in owner.items():
        phase, regime = uses[0][:2]
        paired = phase == 'tuning' and sorted(u[2] for u in uses) == ['gradual', 'static'] \
            and all(u[:2] == (phase, regime) for u in uses)
        if len(uses) > 1 and not paired: raise ValueError('Seed pair reused across contexts/phases')
        for seed, seen in ((lay, layout), (rad, radio)):
            if seen.setdefault(seed, (lay, rad)) != (lay, rad): raise ValueError('Seed identifier collision')
    return phases


def audit(phases, historical_files):
    """Reject any seed already used by a historical (v2 or older) artifact."""
    validate_phases(phases)
    layout, radio, files = set(), set(), {}
    for path in sorted(Path(p) for p in historical_files):
        files[path.as_posix()] = sha256(path)
        _seeds(strict_json(path), layout, radio)
    if not files: raise ValueError('No historical inventory given')
    for rows in phases.values():
        for s in rows:
            if s['layout_seed'] in layout or s['radio_seed'] in radio: raise ValueError('Historical seed collision: '+s['id'])
    evidence = dict(files=files, layout_seeds=sorted(layout), radio_seeds=sorted(radio),
                    inventory_sha256=_canonical_hash(phases))
    return dict(evidence, evidence_sha256=_canonical_hash(evidence))


def make_plan(historical_files):
    phases = build_phases()
    return dict(schema=SCHEMA, policies={k: list(v) for k, v in POLICIES.items()}, phases=phases,
                historical_audit=audit(phases, historical_files))


def load_plan(path):
    plan = strict_json(path)
    if plan.get('schema') != SCHEMA or plan.get('policies') != {k: list(v) for k, v in POLICIES.items()}:
        raise ValueError('v3 seed plan schema/policies')
    validate_phases(plan['phases'])
    a = plan['historical_audit']
    evidence = {k: v for k, v in a.items() if k != 'evidence_sha256'}
    if a['evidence_sha256'] != _canonical_hash(evidence) or a['inventory_sha256'] != _canonical_hash(plan['phases']):
        raise ValueError('Historical audit integrity mismatch')
    layout, radio = set(a['layout_seeds']), set(a['radio_seeds'])
    for rows in plan['phases'].values():
        for s in rows:
            if s['layout_seed'] in layout or s['radio_seed'] in radio: raise ValueError('Historical seed collision')
    return plan


# ---- Frozen phase packages (same guarantees as v2 freeze_phase/load_phase) ----

PHASE_SCHEMA = 'service-v3-phase-v1'
V2_MODEL_SHA256 = '719816f80b9df51c5544422522265e50bd454768869523036a8b8f4b1c657c10'  # formal v2 new-model.json
ALL_POLICIES = POLICIES['validation']


def _cells(plan, phase):
    from dataclasses import asdict
    from ..service_v2.protocol import scenario_config
    return [dict(s, cell_id=s['id']+'__'+p, policy=p, config=asdict(scenario_config(s)),
                 output_path=s['id']+'__'+p, attempts=[0, 1]) for s in plan['phases'][phase] for p in POLICIES[phase]]


def freeze(phase, seed_plan, output, model, reference_model, launcher, ns3_identity, compiler_identity,
           selection=None, inherits=None, go=None):
    """Freeze one v3 phase. validation needs the tuning selection; test also needs
    the validation package it inherits and that package's go decision."""
    from dataclasses import asdict
    import hashlib
    from ..service_v2.config import ServiceV2Config
    from ..service_v2.protocol import (ACCEPTED_REFERENCE_SOURCES, REFERENCE_SHA256, ROOT, reserve,
                                       runtime_identity, safe_path, source_paths)
    if phase not in COUNTS: raise ValueError('Unknown v3 phase')
    if (phase == 'tuning') != (selection is None): raise ValueError('Selection required exactly for validation/test')
    if (phase == 'test') != (inherits is not None and go is not None): raise ValueError('Test must inherit validation + go')
    out = reserve(output)
    plan = load_plan(seed_plan)
    if sha256(model) != V2_MODEL_SHA256: raise ValueError('v3 must reuse the formal v2 predictor unchanged')
    if sha256(reference_model) != REFERENCE_SHA256: raise ValueError('Exact reference model mismatch')
    for name, digest in ACCEPTED_REFERENCE_SOURCES.items():
        if sha256(safe_path(ROOT, name)) != digest: raise ValueError('Accepted legacy reference source changed: '+name)
    inherited = None
    if selection is not None:
        chosen = strict_json(selection)
        if chosen.get('schema') != 'service-v3-tuning-selection-v1' or chosen.get('predictor_sha256') != V2_MODEL_SHA256:
            raise ValueError('Tuning selection schema/predictor')
    if phase == 'test':
        inherited = load(inherits)
        decision = strict_json(go)
        if (inherited['phase'] != 'validation' or decision.get('schema') != 'service-v3-go-v1' or decision.get('go') is not True
                or decision.get('validation_freeze_sha256') != inherited['_freeze_sha256']
                or inherited['selection_sha256'] != sha256(selection)):
            raise ValueError('Test requires a go decision from the validation package with the same selection')
    from ..service_v2.prediction import ServicePredictor
    cells = _cells(plan, phase)
    ServicePredictor.load(model).validate_scenarios(cells)
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
    protocol = dict(schema=PHASE_SCHEMA, phase=phase, policies=list(POLICIES[phase]), cells=cells,
        expected_episodes=len(cells), reference_model=REFERENCE_SHA256, new_model=V2_MODEL_SHA256,
        seed_plan_sha256=sha256(seed_plan), method_sha256=sha256(out/'method.json'),
        selection_sha256=sha256(selection) if selection is not None else None,
        inherited_validation_freeze_sha256=inherited['_freeze_sha256'] if inherited else None,
        runtime=runtime, source_root_at_freeze=str(ROOT), package_root=str(out.resolve()),
        protocol_path_at_freeze=str((out/'protocol.json').resolve()), working_directory=str((out/'source').resolve()),
        retry='one identical-cell technical/infrastructure retry only', service_observation=True,
        run_argv=[runtime['python_executable'], '-m', 'src.control.service_v3.cli', 'run-case',
                  '--protocol', str((out/'protocol.json').resolve()), '--cell-id', '{cell_id}',
                  '--attempt', '{attempt}', '--output', '{exclusive_attempt_output}'])
    write_json(out/'protocol.json', protocol)
    files = {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write_json(out/'freeze.json', dict(schema='service-v3-freeze-v1', files=files))
    load(out/'protocol.json')
    return out/'protocol.json'


def load(path):
    from dataclasses import asdict
    from ..service_v2.config import ServiceV2Config
    from ..service_v2.protocol import REFERENCE_SHA256, safe_path
    path = Path(path).resolve(); root = path.parent
    freeze_record = strict_json(root/'freeze.json')
    if freeze_record.get('schema') != 'service-v3-freeze-v1': raise ValueError('Freeze schema')
    files = freeze_record['files']
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual != set(files) | {'freeze.json'}: raise ValueError('Unlisted/missing frozen file')
    for name, digest in files.items():
        if sha256(safe_path(root, name)) != digest: raise ValueError('Frozen file mismatch: '+name)
    p = strict_json(path)
    if p.get('schema') != PHASE_SCHEMA or p.get('phase') not in COUNTS: raise ValueError('Phase schema')
    plan = load_plan(root/'seed-plan.json')
    if (sha256(root/'seed-plan.json') != p['seed_plan_sha256'] or p['policies'] != list(POLICIES[p['phase']])
            or p['cells'] != _cells(plan, p['phase']) or p['expected_episodes'] != len(p['cells'])):
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
    p['supported_policies'] = ALL_POLICIES
    return p


def strict_json_value(value):
    import json
    from ..service_v2.protocol import strict_loads
    return strict_loads(json.dumps(value))
