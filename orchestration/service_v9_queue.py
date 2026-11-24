"""Fixed-argv queue for one frozen v9 phase (copy of service_v6_queue; v6 evidence pins that file).

The v2 file is left untouched because its sha256 is recorded in v2 evidence.
Runs protocol run_argv per cell, 4 workers, no retries, no analysis.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    protocol_path = args.protocol.resolve(strict=True)
    protocol = json.loads(protocol_path.read_bytes())
    if protocol['schema'] != 'service-v9-phase-v1':
        raise ValueError('Unexpected protocol schema')
    cells = protocol['cells']
    if len(cells) != protocol['expected_episodes']:
        raise ValueError('Declared cell count mismatch')
    identities = [cell['cell_id'] for cell in cells]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate cell IDs')
    if any(Path(name).name != name or name in ('.', '..') for name in identities):
        raise ValueError('Cell ID is not a safe basename')
    working_directory = Path(protocol['working_directory']).resolve(strict=True)
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise FileExistsError(run_root)
    args.evidence.mkdir(parents=True, exist_ok=False)
    commands = []
    for cell in cells:
        substitutions = {'{cell_id}': cell['cell_id'], '{attempt}': '0',
                         '{exclusive_attempt_output}': str(run_root / cell['cell_id'] / 'attempt-0')}
        command = [substitutions.get(token, token) for token in protocol['run_argv']]
        if any('{' in token or '}' in token for token in command):
            raise ValueError('Unresolved argv placeholder')
        commands.append({'cell_id': cell['cell_id'], 'argv': command})
    selected_environment = {name: os.environ.get(name) for name in
        ('LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    write_new(args.evidence / 'queue-plan.json', {
        'schema': 'service-v9-queue-v1', 'protocol': str(protocol_path),
        'protocol_sha256': digest(protocol_path), 'queue_source_sha256': digest(__file__),
        'working_directory': str(working_directory), 'run_root': str(run_root),
        'max_workers': 4, 'attempt': 0, 'automatic_retry': False, 'stop_pending_on_wrapper_error': True,
        'dry_run': args.dry_run, 'environment': selected_environment, 'commands': commands})
    if args.dry_run:
        return 0
    args.lock.parent.mkdir(parents=True, exist_ok=True)
    with args.lock.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Other study queues cannot overlap this queue. The parent also checks
        # local/remote simulation inventories before invoking it.
        running = []
        for entry in Path('/proc').iterdir():
            if entry.name.isdigit():
                try:
                    executable = (entry / 'exe').resolve(strict=True)
                    if executable.name == 'uav-olsr-dataset':
                        running.append(int(entry.name))
                except (OSError, RuntimeError):
                    pass
        if running:
            write_new(args.evidence / 'blocked-existing-simulations.json', {'pids': running})
            raise RuntimeError('Existing simulator processes; parent must reconcile global queue')
        run_root.mkdir(parents=True, exist_ok=False)
        states = {name: {'state': 'planned'} for name in identities}
        mutex = threading.Lock()
        stop_pending = threading.Event()
        revision = 0

        def publish():
            nonlocal revision
            revision += 1
            value = {'revision': revision, 'time_unix': time.time(), 'cells': states}
            path = args.evidence / f'status-{revision:06d}.json'
            write_new(path, value)
            temporary = args.evidence / f'.current-{revision:06d}.json'
            write_new(temporary, value)
            os.replace(temporary, args.evidence / 'current.json')

        def execute(item):
            name = item['cell_id']
            started = time.time()
            with mutex:
                if stop_pending.is_set():
                    result = {'state': 'not_started_after_queue_failure', 'direct_wrapper_exit': None}
                    states[name] = result
                    publish()
                    return result
                states[name] = {'state': 'running', 'started_unix': started}
                publish()
            result = {'state': 'wrapper_finished', 'started_unix': started}
            try:
                with (args.evidence / f'{name}.stdout.log').open('xb') as stdout:
                    with (args.evidence / f'{name}.stderr.log').open('xb') as stderr:
                        child = subprocess.run(item['argv'], cwd=working_directory,
                            stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, check=False)
                result['direct_wrapper_exit'] = child.returncode
            except Exception as error:
                result.update(state='queue_error', error=f'{type(error).__name__}: {error}',
                              direct_wrapper_exit=None)
            result.update(finished_unix=time.time(), wall_seconds=time.time() - started)
            with mutex:
                if result['direct_wrapper_exit'] != 0:
                    stop_pending.set()
                states[name] = result
                publish()
            return result

        publish()
        with ThreadPoolExecutor(max_workers=4) as workers:
            futures = [workers.submit(execute, item) for item in commands]
            results = [future.result() for future in as_completed(futures)]
        write_new(args.evidence / 'queue-finished.json', {
            'cells': len(results), 'wrapper_exit_zero': sum(r['direct_wrapper_exit'] == 0 for r in results),
            'note': 'Wrapper completion is not collection validation; validate every frozen cell separately.'})
        return 0 if all(r['direct_wrapper_exit'] == 0 for r in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
