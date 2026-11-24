"""Technical-only smoke of common positioning law; no outcome-based tuning."""
import argparse
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.control.prevention.evaluation import run_grid
from src.control.prevention.prediction import LearnedPredictor
from src.control.service_v2.protocol import sha256
from src.control.service_v2.study import read_frames, reconcile_episode
from src.control.service_v3.protocol import V2_MODEL_SHA256
from src.control.service_v4.study import route_integrity
from src.control.service_v7.study import background, startup_blackout, transmissions
from src.control.service_v9.study import env_for, make_factory, scenario_config
from src.control.service_v9.analysis import safety


def main():
    parser = argparse.ArgumentParser()
    for name in ("selection", "reference-model", "launcher", "output", "lock"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import fcntl
    with args.lock.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        phase = dict(_path=str(args.selection.parent / "smoke-protocol.json"), selection_sha256=sha256(args.selection),
                     new_model=V2_MODEL_SHA256, method_sha256="smoke-only")
        factory = make_factory(phase, None, LearnedPredictor.load(args.reference_model))
        evidence = []
        for i, (regime, policy) in enumerate((("n5", "gnn-motion"), ("n7", "reactive-motion")), 1):
            config = scenario_config(dict(regime=regime, context="gradual"))
            output = args.output / regime
            run_grid(config=config, launcher=[str(args.launcher)], output=output, policies=[policy],
                     scenarios=[(3599000 + i, 3699000 + i)], controller_factory=factory,
                     supported_policies=(policy,), env_factory=partial(env_for(policy), service_observation=True),
                     provenance=dict(purpose="technical smoke only", selection_sha256=phase["selection_sha256"]))
            records = json.loads((output / "episodes.json").read_text())
            if len(records) != 1 or records[0]["status"] != "ok":
                raise RuntimeError(str(records))
            episode = Path(records[0]["episode_dir"])
            reconcile_episode(episode, config)
            row = dict(policy=policy, config=asdict(config), episode_dir=str(episode), frames=read_frames(episode, config),
                       decisions=[json.loads(line) for line in (episode.parent / "decisions.jsonl").read_text().splitlines()])
            check = dict(regime=regime, policy=policy, integrity=route_integrity(row)["ok"], safety=safety(row),
                         tx=transmissions(row) is not None, background=background(row) is not None,
                         no_blackout=not startup_blackout(row))
            print(json.dumps(check), flush=True)
            assert all(value for key, value in check.items() if key not in ("regime", "policy"))
            evidence.append(check)
        (args.output / "smoke-finished.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
