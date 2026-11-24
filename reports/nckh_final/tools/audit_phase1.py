"""Check archived phase-1 numbers against their bundled per-run CSV evidence."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean
from zipfile import ZipFile

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parents[1]
archive = ROOT / 'reports/Bao_cao_tien_do_NCKH_UAV_2026-10-31_nguon.zip'
prefix = 'reports/core_matrix_bursty_fixed/'
with ZipFile(archive) as z:
    raw = z.read(prefix + 'metrics_runs.csv')
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    paired_raw = z.read(prefix + 'progress_paired_effects.csv')
    paired = list(csv.DictReader(io.StringIO(paired_raw.decode('utf-8-sig'))))
assert len(rows) == 6720
keys = [(r['run_name'],r['model_id'],r['target'],r['horizon']) for r in rows]
assert len(set(keys)) == len(keys)
assert len({r['run_name'] for r in rows}) == 210
assert all(r['protocol'] == 'held-seed-full-run' and math.isfinite(float(r['macro_f1'])) for r in rows)
means = {}
false_stable = {}
for target in ('qos','survival'):
    for horizon in ('1','2','3','5'):
        name = target + '/' + horizon
        means[name] = {model:mean(float(r['macro_f1']) for r in rows if
                                  (r['model_id'],r['target'],r['horizon'])==(model,target,horizon))
                       for model in ('edge-sage','xgb','logreg','persistence')}
        false_stable[name] = {}
        for model in ('edge-sage','xgb'):
            condition_means=[]
            for condition in sorted({r['condition'] for r in rows}):
                values=[float(r['false_stable']) for r in rows if r['false_stable'] and
                        (r['model_id'],r['target'],r['horizon'],r['condition'])==(model,target,horizon,condition)]
                if values:
                    condition_means.append(mean(values))
            false_stable[name][model]=mean(condition_means)
        effect = next(p for p in paired if (p['target'],p['horizon'])==(target,horizon))
        assert abs(means[name]['edge-sage']-means[name]['xgb']-float(effect['delta'])) < 1e-12
assert sum(v['edge-sage'] > v['xgb'] for v in false_stable.values()) == 7
result = dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    metrics_sha256=hashlib.sha256(raw).hexdigest(),paired_sha256=hashlib.sha256(paired_raw).hexdigest(),
    rows=len(rows), episodes=210, means=means, false_stable=false_stable, paired=paired,
    scope='Read-only audit of archived metrics and paired point estimates; no retraining or new phase-1 inference.')
(REPORT/'evidence/phase1-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(rows=6720,episodes=210,all_paired_points_match=True,false_stable_worse_combinations=7),indent=2))
