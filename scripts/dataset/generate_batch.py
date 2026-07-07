from __future__ import annotations
import os
import random
import subprocess
import sys
from pathlib import Path
import yaml
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_SCRIPT = PROJECT_ROOT / 'scripts' / 'dataset' / 'run_one_dataset_ns3.sh'

def main() -> None:
    params = yaml.safe_load((PROJECT_ROOT / 'params.yaml').read_text())['generate']
    count = int(params['count'])
    prefix = str(params.get('prefix', 'ns3big'))
    n_min = int(params['num_uavs_min'])
    n_max = int(params['num_uavs_max'])
    rng = random.Random(int(params['base_seed']))
    for i in range(1, count + 1):
        seed = rng.randint(10000, 50000)
        mobility = rng.choice(['random-waypoint', 'gauss-markov'])
        num_uavs = rng.randint(n_min, n_max)
        comm_range = rng.randint(180, 280)
        time_steps = rng.randint(80, 150)
        speed_min = rng.randint(2, 5)
        speed_max = speed_min + rng.randint(3, 7)
        area_side = int(500 * (num_uavs / 8.0) ** 0.5)
        tag = 'rwp' if mobility == 'random-waypoint' else 'gm'
        run_name = f'{prefix}_{i:03d}_{tag}_s{seed}_n{num_uavs}_c{comm_range}_t{time_steps}'
        print(f'[{i}/{count}] {run_name}', flush=True)
        env = {**os.environ, 'SIM_NUM_UAVS': str(num_uavs), 'SIM_COMM_RANGE': str(comm_range), 'SIM_TIME_STEPS': str(time_steps), 'SIM_RWP_SPEED_MIN': str(speed_min), 'SIM_RWP_SPEED_MAX': str(speed_max), 'SIM_X_MAX': str(area_side), 'SIM_Y_MAX': str(area_side)}
        subprocess.run(['bash', str(PIPELINE_SCRIPT), run_name, str(seed), mobility], env=env, check=True, cwd=PROJECT_ROOT)
    print(f"[OK] generated {count} datasets with prefix '{prefix}'")
if __name__ == '__main__':
    sys.exit(main())