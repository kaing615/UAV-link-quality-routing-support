"""Service v3 seed plan: new scenario seeds, audited disjoint from every v2 seed.

Freeze/run machinery is reused from service_v2 once the v3 policies exist;
this module only fixes *which* scenarios each v3 phase uses.  See
docs/service_v3_protocol.md for the pre-registered analysis.
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Write the audited v3 seed plan (refuses to overwrite).')
    parser.add_argument('--historical', nargs='+', required=True, help='v2 seed-plan JSON files to audit against')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, make_plan(args.historical))
    load_plan(args.output)
    print(args.output, sha256(args.output))
