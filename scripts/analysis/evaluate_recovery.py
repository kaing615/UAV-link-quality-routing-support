"""Paired held-out evaluation of recovery PPO and three motion baselines."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.control.bridge import make_launcher
from src.control.config import RecoveryConfig
from src.control.environment import Ns3RecoveryEnv
from src.control.evaluation import (
    ScenarioSeeds,
    baseline_action,
    evaluate_episode,
    git_dirty,
    hierarchical_bootstrap_recovery,
    recovery_source_hashes,
    runtime_versions,
    scenario_seeds,
    sha256_file,
    summarize_episodes,
    validate_paired_grid,
    validate_seed_partitions,
    verify_build_manifest,
)


def _model_manifest(path: Path) -> dict:
    manifest_path = path.parent / "run_config.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"model provenance missing: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _check_model_seeds(manifest: dict, scenarios: list[ScenarioSeeds]) -> None:
    partitions = manifest["seed_partitions"]
    train_first = ScenarioSeeds(**partitions["training_first"])
    train_last = ScenarioSeeds(**partitions["training_last_reserved"])
    for scenario in scenarios:
        if train_first.layout_seed <= scenario.layout_seed <= train_last.radio_seed:
            raise ValueError("test layout seed overlaps reserved training range")
        if train_first.layout_seed <= scenario.radio_seed <= train_last.radio_seed:
            raise ValueError("test radio seed overlaps reserved training range")
    validation = [ScenarioSeeds(**row) for row in partitions["validation"]]
    validate_seed_partitions([], validation, scenarios)


def _save_figure(output: Path, records, model_paths: list[Path], paired_valid: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = summarize_episodes(records)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = list(summaries)
    axes[0].bar(labels, [summaries[name]["recovery_rate"] or 0 for name in labels], color="#366A8A")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Confirmed recovery fraction")
    axes[0].tick_params(axis="x", labelrotation=30)
    axes[0].set_title("Held-out episodes" if paired_valid else "Descriptive only: incomplete paired grid")
    plotted = False
    single_step = False
    for path in model_paths:
        history_path = path.parent / "validation_history.csv"
        if history_path.is_file():
            with history_path.open(newline="", encoding="utf-8") as stream:
                history = list(csv.DictReader(stream))
            if history:
                steps = [int(row["timesteps"]) for row in history]
                returns = [float(row["mean_validation_return"]) for row in history]
                if len(set(steps)) == 1:
                    # Two checkpoint phases at the same step are not a learning
                    # curve. Display the individual measurements explicitly.
                    single_step = True
                    phases = [row.get("phase", "checkpoint").replace("_", " ") for row in history]
                    axes[1].scatter(phases, returns, label=path.parent.name)
                else:
                    axes[1].plot(steps, returns, marker="o", label=path.parent.name)
                plotted = True
    axes[1].set_xlabel("Checkpoint phase (one training step count)" if single_step else "Training environment steps")
    axes[1].set_ylabel("Mean validation return")
    axes[1].set_title(
        "Validation at final training step"
        if single_step
        else "Validation learning curve"
        if plotted
        else "No training history supplied"
    )
    if plotted:
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "learning_recovery.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("evaluation output must be an empty directory or a new path")
    binary = args.binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    build_hashes = verify_build_manifest(binary)
    binary_hash = sha256_file(binary)
    source_hashes = recovery_source_hashes()
    output.mkdir(parents=True, exist_ok=True)
    config = RecoveryConfig(steps=args.steps)
    launcher = make_launcher(binary, args.wsl)
    scenarios = scenario_seeds("test", args.scenarios)
    model_paths = [path.resolve() for path in args.model]
    manifests = [_model_manifest(path) for path in model_paths]
    for manifest in manifests:
        _check_model_seeds(manifest, scenarios)
        if manifest["config"] != asdict(config):
            raise ValueError("evaluation config differs from model training config")
        if manifest["provenance"]["binary_sha256"] != binary_hash:
            raise ValueError("evaluation binary differs from model training binary")
    model_artifacts = []
    for path, manifest in zip(model_paths, manifests):
        model_hash = sha256_file(path)
        expected_key = (
            "best_model_sha256"
            if path.name == "best_model.zip"
            else "final_model_sha256"
            if path.name == "model.zip"
            else None
        )
        expected = manifest.get("result", {}).get(expected_key) if expected_key else None
        if expected is not None and expected != model_hash:
            raise ValueError(f"model hash differs from training manifest: {path}")
        model_artifacts.append(
            {
                "path": str(path),
                "sha256": model_hash,
                "training_manifest_sha256": sha256_file(path.parent / "run_config.json"),
            }
        )
    model_seeds = [int(m["training"]["policy_seed"]) for m in manifests]
    if len(set(model_seeds)) != len(model_seeds):
        raise ValueError("duplicate policy training seeds")

    policies = []
    for baseline in ("zero", "random", "relay_chain"):
        policies.append((baseline, None, None))
    if model_paths:
        from stable_baselines3 import PPO

        for path, seed in zip(model_paths, model_seeds):
            policies.append(("ppo", seed, PPO.load(path, device="cpu")))
    records = []
    for policy_name, policy_seed, model in policies:
        for index, scenario in enumerate(scenarios):
            policy_label = f"{policy_name}_{policy_seed}" if policy_seed is not None else policy_name
            episode_root = output / "episodes" / policy_label / f"scenario_{index:03d}"
            env = Ns3RecoveryEnv(config=config, launcher=launcher, output_root=episode_root, training=False)
            if model is None:
                # Identical random sequence per scenario if this baseline is rerun.
                rng = np.random.default_rng(np.random.SeedSequence([scenario.layout_seed, scenario.radio_seed]))

                def action(obs, frame, name=policy_name, generator=rng):
                    return baseline_action(name, frame, config, generator)
            else:

                def action(obs, frame, selected=model):
                    return selected.predict(obs, deterministic=True)[0]

            record = evaluate_episode(
                env, action, scenario, policy_name, policy_seed, trace_path=episode_root / "trajectory.jsonl"
            )
            records.append(record)
            print(f"{policy_label} scenario {index + 1}/{len(scenarios)}: {record.status}", flush=True)
    rows = [record.to_dict() for record in records]
    (output / "episodes.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = summarize_episodes(records)
    paired_grid_error = None
    try:
        validate_paired_grid(records)
    except ValueError as exc:
        paired_grid_error = str(exc)
    for name in summaries:
        if paired_grid_error is not None:
            interval = None
            summaries[name]["bootstrap_error"] = paired_grid_error
        else:
            try:
                interval = hierarchical_bootstrap_recovery([record for record in records if record.policy == name])
            except ValueError as exc:
                interval = None
                summaries[name]["bootstrap_error"] = str(exc)
        summaries[name]["hierarchical_recovery_rate_95pct_ci"] = interval
    report = {
        "scope": "single_policy_seed_exploratory" if len(model_paths) < 2 else "multi_policy_seed",
        "scenarios": [asdict(scenario) for scenario in scenarios],
        "config": asdict(config),
        "models": [str(path) for path in model_paths],
        "summary": summaries,
        "paired_comparison_valid": paired_grid_error is None,
        "paired_grid_error": paired_grid_error,
        "provenance": {
            "binary": str(binary),
            "binary_sha256": binary_hash,
            "build_manifest": build_hashes,
            "python_source_sha256": source_hashes,
            "python_source_sha256_at_completion": recovery_source_hashes(),
            "git_dirty": git_dirty(),
            "dependency_versions": runtime_versions(),
            "model_artifacts": model_artifacts,
        },
    }
    report["provenance"]["source_changed_during_run"] = (
        report["provenance"]["python_source_sha256_at_completion"] != source_hashes
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _save_figure(output, records, model_paths, paired_grid_error is None)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--wsl", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, action="append", default=[], help="Repeat for distinct trained policy seeds"
    )
    parser.add_argument("--scenarios", type=int, default=20)
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    if args.scenarios <= 0:
        parser.error("scenarios must be positive")
    result = run(args)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
