"""Export compact evidence after analysis; never run or alter simulation cells."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import tarfile


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    phases = [phase for phase in ('validation', 'test') if (root / f'analysis/{phase}/analysis.json').exists()]
    assert phases and phases[0] == 'validation'
    version = root.name.rsplit('-', 1)[-1]
    files = [root / 'selection.json', root / f'repo/docs/service_{version}_protocol.md',
             root / f'repo/orchestration/service_{version}_queue.py', root / 'development/runner.sh']
    supplements = {}
    for phase in phases:
        report = read(root / f'runs/{phase}-validation.json')
        assert report['complete'] and report['planned_cells'] == report['ok_cells']
        files += [root / f'runs/{phase}-validation.json']
        files += list((root / f'analysis/{phase}').glob('*.json'))
        files += [p for p in (root / f'packages/{phase}').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        for name in ('current.json', 'queue-finished.json'):
            files.append(root / f'evidence/queue-{phase}/{name}')
        files += [root / f'evidence/queue-{phase}.log', root / f'evidence/queue-{phase}-launched.txt']
        timings = defaultdict(list)
        per_episode = []
        examples = {}
        motion = []
        gradual_n5 = [row for row in report['rows'] if row['context'] == 'gradual' and row['regime'] == 'n5']
        example_seed = min(row['layout_seed'] for row in gradual_n5)
        for row in report['rows']:
            assert row['status'] == 'ok'
            attempt = root / f'runs/{phase}' / row['cell_id'] / 'attempt-0'
            result = read(attempt / 'result.json')
            assert result['status'] == 'ok' and result['actual_exit'] == 0 and result['error'] is None
            values = [item['decision_seconds'] * 1000 for item in read(attempt / 'timings.json')]
            assert len(values) == 240
            timings[row['policy']].extend(values)
            per_episode.append(dict(cell_id=row['cell_id'], mean_ms=statistics.mean(values),
                                    median_ms=statistics.median(values), max_ms=max(values)))
            files += [attempt / name for name in ('result.json', 'artifacts.json', 'timings.json')]
            if version == 'v9':
                episode = attempt / result['episode_path']
                trace = episode / 'transitions.jsonl'
                assert digest(trace) == result['trace_sha256']
                frames = [json.loads(line)['frame'] for line in trace.read_text().splitlines()]
                decisions = [json.loads(line) for line in (episode.parent / 'decisions.jsonl').read_text().splitlines()]
                config = row['config']
                relays = [i for i in range(config['num_uavs']) if i not in (config['source'], config['destination'])]
                motion.append(dict(cell_id=row['cell_id'], context=row['context'], regime=row['regime'], policy=row['policy'],
                    layout_seed=row['layout_seed'], radio_seed=row['radio_seed'],
                    first_motion_step=next((d['step'] for d in decisions if d['commanded_motion']),None),
                    first_outage_step=next((f['step'] for f in frames[1:] if not f['source_dest_connected']),None),
                    relay_travel_m={str(i):frames[-1]['travel'][i] for i in relays},
                    budget_exhausted_frames={str(i):sum(f['travel'][i]>=config['budget']-1e-6 for f in frames[1:]) for i in relays},
                    safety_blocked_decisions=sum(d['motion_control']['safety_blocked'] for d in decisions),
                    simulator_safety_rejections=sum(f['safety_rejected'] for f in frames),
                    motion_starts=sum(d['motion_control']['started'] for d in decisions)))
            if row['context'] == 'gradual' and row['regime'] == 'n5' and row['layout_seed'] == example_seed:
                episode = attempt / result['episode_path']
                # Copy the original trace, not a reconstructed trajectory.
                trace = episode / 'transitions.jsonl'
                assert digest(trace) == result['trace_sha256']
                files += [trace, episode.parent / 'decisions.jsonl', episode / 'config.json']
                examples[row['policy']] = str(trace.relative_to(root))
        summary = {}
        for arm, values in timings.items():
            ordered = sorted(values)
            summary[arm] = dict(n=len(values), mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                                p95_ms=ordered[int(.95 * (len(ordered) - 1))], max_ms=max(values))
        supplements[phase] = dict(decision_timing_ms=summary, timing_episodes=per_episode,
            motion_episodes=motion,
            example_traces=examples, example_selection='Lowest layout seed among n5 gradual scenarios; all arms, regardless of outcome.',
            timing_note='Measured complete controller.act latency on server CPU during four-worker simulation; not pure neural inference or UAV/control-channel latency.')
    if (root / 'smoke/smoke-finished.json').exists():
        files.append(root / 'smoke/smoke-finished.json')
    supplement = root / 'descriptive-supplement.json'
    assert not supplement.exists()
    supplement.write_text(json.dumps(dict(phases=supplements, host=platform.node(), platform=platform.platform()), indent=2) + '\n')
    files.append(supplement)
    files = sorted(set(files))
    manifest = root / 'export-manifest.json'
    assert not manifest.exists()
    manifest.write_text(json.dumps(dict(files={str(p.relative_to(root)): digest(p) for p in files}), indent=2) + '\n')
    target = root / 'results-export.tar.gz'
    with tarfile.open(target, 'x:gz') as archive:
        for path in files + [manifest]:
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    print(json.dumps(dict(archive=str(target), files=len(files), bytes=target.stat().st_size,
                          sha256=digest(target), phases=phases)))


if __name__ == '__main__':
    main()
