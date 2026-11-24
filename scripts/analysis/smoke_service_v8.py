"""Two development-only live cells; print technical checks, never select by PDR."""
import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.control.prevention.evaluation import run_grid
from src.control.prevention.prediction import LearnedPredictor
from src.control.service_v2.protocol import scenario_config, sha256
from src.control.service_v2.study import read_frames, reconcile_episode
from src.control.service_v3.protocol import V2_MODEL_SHA256
from src.control.service_v4.study import route_integrity
from src.control.service_v7.study import background, recorded_command, startup_blackout, transmissions
from src.control.service_v8.study import arm_flags, env_for, forecast_trace_ok, make_factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import fcntl
    with args.lock.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        phase = dict(_path=str(args.selection.parent / "smoke-protocol.json"),
                     selection_sha256=sha256(args.selection), new_model=V2_MODEL_SHA256, method_sha256="smoke-only")
        factory = make_factory(phase, None, LearnedPredictor.load(args.reference_model))
        evidence = []
        for i, (regime, policy) in enumerate((("n5", "gnn-route"), ("n7", "mlp-route")), 1):
            config = scenario_config(dict(regime=regime, context="gradual"))
            output = args.output / regime
            run_grid(config=config, launcher=[str(args.launcher)], output=output,
                     policies=[policy], scenarios=[(3399000 + i, 3499000 + i)],
                     controller_factory=factory, supported_policies=(policy,),
                     env_factory=partial(env_for(policy), service_observation=True),
                     provenance=dict(purpose="technical smoke only", selection_sha256=phase["selection_sha256"]))
            records = json.loads((output / "episodes.json").read_text())
            if len(records) != 1 or records[0]["status"] != "ok":
                raise RuntimeError(str(records))
            episode = Path(records[0]["episode_dir"])
            reconciliation = reconcile_episode(episode, config)
            row = dict(policy=policy, config=asdict(config), episode_dir=str(episode),
                       frames=read_frames(episode, config),
                       decisions=[json.loads(line) for line in (episode.parent / "decisions.jsonl").read_text().splitlines()])
            check = dict(regime=regime, policy=policy, integrity=route_integrity(row)["ok"],
                         flags=recorded_command(row)[-len(arm_flags(policy)):] == arm_flags(policy),
                         forecast=forecast_trace_ok(row, phase["selection_sha256"]),
                         tx=transmissions(row) is not None, background=background(row) is not None,
                         no_blackout=not startup_blackout(row), stationary=reconciliation["total_relay_travel_m"] < 1e-6)
            print(json.dumps(check), flush=True)
            assert all(value for key, value in check.items() if key not in ("regime", "policy"))
            evidence.append(check)
        (args.output / "smoke-finished.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
