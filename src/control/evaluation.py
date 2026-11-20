"""Episode-level recovery evaluation and reproducible baseline policies."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(paths: Iterable[Path], root: Path) -> dict[str, str]:
    """Hash named source files using stable repository-relative keys."""
    root = root.resolve()
    result = {}
    for candidate in paths:
        path = Path(candidate).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"source file missing or outside root: {candidate}")
        result[path.relative_to(root).as_posix()] = sha256_file(path)
    return dict(sorted(result.items()))


def recovery_source_hashes() -> dict[str, str]:
    sources = list((REPOSITORY_ROOT / "src" / "control").glob("*.py"))
    sources += [
        REPOSITORY_ROOT / "scripts" / "train" / "train_recovery_ppo.py",
        REPOSITORY_ROOT / "scripts" / "analysis" / "evaluate_recovery.py",
    ]
    return file_hashes(sources, REPOSITORY_ROOT)


def git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPOSITORY_ROOT,
            capture_output=True, text=True, check=False, timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def runtime_versions() -> dict[str, str | None]:
    def version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": version("torch"),
        "gymnasium": version("gymnasium"),
        "stable_baselines3": version("stable-baselines3"),
        "matplotlib": version("matplotlib"),
    }


@dataclass(frozen=True)
class ScenarioSeeds:
    layout_seed: int
    radio_seed: int


@dataclass
class EpisodeRecord:
    policy: str
    policy_seed: int | None
    layout_seed: int
    radio_seed: int
    status: str
    recovery_time_s: int | None
    elapsed_steps: int
    return_: float
    travel_total_m: float | None = None
    travel_max_m: float | None = None
    rejected_actions: int = 0
    connected_fraction: float | None = None
    refragmentations: int = 0
    tx_packets: int | None = None
    rx_packets: int | None = None
    offered_packets: int | None = None
    pdr: float | None = None
    mean_delay_ms: float | None = None
    inference_time_s: float = 0.0
    error: str | None = None
    episode_dir: str | None = None
    terminal_reason: str | None = None
    requested_layout_seed: int | None = None
    requested_radio_seed: int | None = None
    final_components: int | None = None
    budget_exhausted: bool = False

    @property
    def censored(self) -> bool:
        return self.status in {"horizon", "budget_exhausted"}

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["return"] = result.pop("return_")
        result["censored"] = self.censored
        return result


def validate_seed_partitions(
    training: Iterable[ScenarioSeeds], validation: Iterable[ScenarioSeeds], test: Iterable[ScenarioSeeds]
) -> None:
    """Reject reuse of either source of scenario randomness across splits."""
    partitions = {"training": list(training), "validation": list(validation), "test": list(test)}
    for name, scenarios in partitions.items():
        if len(set(scenarios)) != len(scenarios):
            raise ValueError(f"duplicate scenario in {name}")
        if any(s.layout_seed < 0 or s.radio_seed < 0 for s in scenarios):
            raise ValueError("scenario seeds must be nonnegative")
    for left, right in (("training", "validation"), ("training", "test"), ("validation", "test")):
        for field in ("layout_seed", "radio_seed"):
            a = {getattr(s, field) for s in partitions[left]}
            b = {getattr(s, field) for s in partitions[right]}
            if a & b:
                raise ValueError(f"{field} overlap between {left} and {right}: {sorted(a & b)}")


def scenario_seeds(split: str, count: int, policy_seed: int = 1) -> list[ScenarioSeeds]:
    """Reserved disjoint seed ranges; policy seed varies training episodes only."""
    if count < 0 or not 0 <= policy_seed < 999 or count >= 50_000:
        raise ValueError("seed range exceeded")
    offsets = {"training": 100_000, "validation": 100_000_000, "test": 200_000_000}
    if split not in offsets:
        raise ValueError(f"unknown split {split}")
    base = offsets[split] + (policy_seed * 100_000 if split == "training" else 0)
    return [ScenarioSeeds(base + 2 * i, base + 2 * i + 1) for i in range(count)]


def baseline_action(name: str, frame: dict[str, Any], config: Any, rng: np.random.Generator) -> np.ndarray:
    movable = [i for i in range(config.num_uavs) if i not in (config.source, config.destination)]
    shape = (len(movable), 3)
    if name == "zero":
        action = np.zeros(shape)
    elif name == "random":
        action = rng.uniform(-1.0, 1.0, size=shape)
    elif name == "relay_chain":
        positions = np.asarray(frame["positions"], dtype=float)
        source = positions[config.source]
        destination = positions[config.destination]
        axis = destination - source
        denominator = float(axis @ axis)
        order = sorted(movable, key=lambda i: float((positions[i] - source) @ axis) / denominator if denominator else i)
        action = np.zeros(shape)
        for rank, node in enumerate(order, start=1):
            target = source + axis * rank / (len(order) + 1)
            delta = (target - positions[node]) / config.max_speed
            norm = np.linalg.norm(delta)
            action[movable.index(node)] = delta / max(1.0, norm)
    else:
        raise ValueError(f"unknown baseline {name}")
    return action.astype(np.float32).reshape(-1)


def read_final_metrics(path: Path, *, normally_completed: bool) -> dict[str, float | int | None]:
    if not normally_completed:
        raise ValueError("metrics require normal simulator completion")
    if not path.is_file():
        raise FileNotFoundError(f"final simulator metrics missing: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"expected one final metrics row in {path}")
    row = rows[0]
    result: dict[str, float | int | None] = {}
    for name in ("tx_packets", "rx_packets", "offered_packets"):
        result[name] = int(row[name])
    for name in ("pdr", "mean_delay_ms"):
        value = float(row[name])
        result[name] = value if math.isfinite(value) else None
    if result["offered_packets"] and result["pdr"] is not None:
        measured = result["rx_packets"] / result["offered_packets"]
        if abs(measured - result["pdr"]) > 1e-5:
            raise ValueError("PDR disagrees with final simulator packet counts")
    return result


def verify_build_manifest(binary: Path) -> dict[str, str] | None:
    """Validate every declared source and binary hash if a build manifest exists."""
    binary = Path(binary).resolve()
    manifest = binary.parent / "source.sha256"
    if not manifest.is_file():
        return None
    repository = binary.parent.parent.parent
    hashes = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64:
            raise ValueError(f"malformed build manifest line: {line!r}")
        path = (repository / relative).resolve()
        if not path.is_relative_to(repository) or not path.is_file():
            raise ValueError(f"build manifest source missing or outside repository: {relative}")
        actual = sha256_file(path)
        if actual != digest:
            raise ValueError(f"stale build manifest for {relative}")
        hashes[relative] = digest
    binary_relative = binary.relative_to(repository).as_posix()
    if binary_relative not in hashes:
        raise ValueError("build manifest omits simulator binary")
    return hashes


def _connected(frame: dict[str, Any], config: Any) -> bool:
    return bool(frame["components"] == 1 and frame["source_dest_connected"] and frame["olsr_route_usable"])


def evaluate_episode(
    env: Any,
    policy: Callable[[np.ndarray, dict[str, Any]], np.ndarray],
    scenario: ScenarioSeeds,
    policy_name: str,
    policy_seed: int | None,
    trace_path: Path | None = None,
) -> EpisodeRecord:
    """Run one live simulator episode; technical faults never become policy failures."""
    record = EpisodeRecord(
        policy_name,
        policy_seed,
        scenario.layout_seed,
        scenario.radio_seed,
        "technical_error",
        None,
        0,
        0.0,
        requested_layout_seed=scenario.layout_seed,
        requested_radio_seed=scenario.radio_seed,
    )
    initial_travel = None
    connected_count = 0
    last_connected = False
    trace = []
    try:
        observation, reset_info = env.reset(
            options={"layout_seed": scenario.layout_seed, "radio_seed": scenario.radio_seed}
        )
        record.layout_seed = int(reset_info["layout_seed"])
        record.radio_seed = int(reset_info["radio_seed"])
        if (record.layout_seed, record.radio_seed) != (scenario.layout_seed, scenario.radio_seed):
            raise ValueError("simulator reset used different layout/radio seeds than requested")
        initial_travel = np.asarray(env.last_frame["travel"], dtype=float)
        record.episode_dir = str(env.episode_output_dir)
        while record.elapsed_steps < env.config.steps:
            frame = env.last_frame
            started = time.perf_counter()
            action = policy(observation, frame)
            record.inference_time_s += time.perf_counter() - started
            observation, reward, terminated, truncated, info = env.step(action)
            if trace_path is not None:
                trace.append(
                    {
                        "step": record.elapsed_steps,
                        "previous_frame": frame,
                        "proposed_action": np.asarray(action).tolist(),
                        "next_frame": env.last_frame,
                        "reward": float(reward),
                        "reward_components": info.get("reward_components"),
                        "terminal_reason": info.get("terminal_reason"),
                    }
                )
            record.elapsed_steps += 1
            record.return_ += float(reward)
            frame = env.last_frame
            record.final_components = int(frame["components"])
            record.budget_exhausted = bool(info.get("budget_exhausted", False))
            connected = frame["components"] == 1
            connected_count += int(connected)
            if record.recovery_time_s is not None and last_connected and not connected:
                record.refragmentations += 1
            last_connected = connected
            record.rejected_actions += int(bool(info.get("safety_rejected", frame.get("safety_rejected", False))))
            if info.get("success") and record.recovery_time_s is None:
                record.recovery_time_s = int(info.get("recovery_time_s") or record.elapsed_steps)
            if terminated or truncated:
                record.terminal_reason = info.get("terminal_reason") or ("horizon" if truncated else "budget_exhausted")
                record.status = record.terminal_reason
                break
        else:
            record.status = "horizon"
        if record.recovery_time_s is not None and record.status in {"horizon", "budget_exhausted"}:
            record.status = "success"
        if record.recovery_time_s is not None and record.elapsed_steps < env.config.steps:
            raise ValueError("evaluation stopped after recovery before the horizon")
        if record.status == "success" and record.recovery_time_s is None:
            raise ValueError("success reported without recovery time")
        if record.status not in {"success", "horizon", "budget_exhausted"}:
            raise ValueError(f"invalid terminal reason {record.status}")
        travel = np.asarray(env.last_frame["travel"], dtype=float) - initial_travel
        record.travel_total_m = float(travel.sum())
        record.travel_max_m = float(travel.max())
        record.connected_fraction = connected_count / record.elapsed_steps if record.elapsed_steps else None
        env.close()  # QUIT/normal exit must flush the simulator's final metrics.
        record.__dict__.update(
            read_final_metrics(Path(record.episode_dir) / "closed_loop_metrics.csv", normally_completed=True)
        )
    except Exception as exc:
        record.status = "technical_error"
        record.recovery_time_s = None
        record.error = f"{type(exc).__name__}: {exc}"
        try:
            env.close()
        except Exception:
            pass
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("w", encoding="utf-8") as stream:
            for row in trace:
                stream.write(json.dumps(row) + "\n")
    return record


def _mean(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else None


def summarize_episodes(records: Iterable[EpisodeRecord]) -> dict[str, dict[str, Any]]:
    """Aggregate scenarios within policy seed, then policy seeds equally."""
    grouped: dict[str, list[EpisodeRecord]] = {}
    for row in records:
        grouped.setdefault(row.policy, []).append(row)
    summaries = {}
    for policy, rows in grouped.items():
        valid = [r for r in rows if r.status != "technical_error"]
        by_seed: dict[int | None, list[EpisodeRecord]] = {}
        for row in valid:
            by_seed.setdefault(row.policy_seed, []).append(row)
        seed_rates = [sum(r.status == "success" for r in seed_rows) / len(seed_rows) for seed_rows in by_seed.values()]
        summaries[policy] = {
            "episodes": len(valid),
            "technical_errors": len(rows) - len(valid),
            "successes": sum(r.status == "success" for r in valid),
            "censored": sum(r.censored for r in valid),
            "policy_seeds": len(by_seed),
            "recovery_rate": _mean(seed_rates),
            "pooled_episode_recovery_rate": _mean([float(r.status == "success") for r in valid]),
            "time_to_recovery_success_only_mean_s": _mean(r.recovery_time_s for r in valid),
            "travel_total_mean_m": _mean(_mean(r.travel_total_m for r in seed_rows) for seed_rows in by_seed.values()),
            "travel_max_mean_m": _mean(_mean(r.travel_max_m for r in seed_rows) for seed_rows in by_seed.values()),
            "connected_fraction_mean": _mean(
                _mean(r.connected_fraction for r in seed_rows) for seed_rows in by_seed.values()
            ),
            "refragmentations_mean": _mean(
                _mean(r.refragmentations for r in seed_rows) for seed_rows in by_seed.values()
            ),
            "pdr_mean": _mean(_mean(r.pdr for r in seed_rows) for seed_rows in by_seed.values()),
            "mean_delay_ms": _mean(_mean(r.mean_delay_ms for r in seed_rows) for seed_rows in by_seed.values()),
            "scope": "multi_policy_seed" if len(by_seed) > 1 else "single_policy_seed_exploratory",
        }
    return summaries


def validate_paired_grid(records: Iterable[EpisodeRecord]) -> None:
    """Require every policy run to cover the identical set of scenario pairs."""
    groups: dict[tuple[str, int | None], set[tuple[int, int]]] = {}
    for row in records:
        if row.status == "technical_error":
            raise ValueError("technical errors prevent a complete paired comparison")
        key = (row.policy, row.policy_seed)
        scenario = (row.layout_seed, row.radio_seed)
        group = groups.setdefault(key, set())
        if scenario in group:
            raise ValueError(f"duplicate scenario {scenario} for {key}")
        group.add(scenario)
    grids = list(groups.values())
    if grids and any(grid != grids[0] for grid in grids[1:]):
        raise ValueError("policies have unequal paired scenario grids")


def hierarchical_bootstrap_recovery(
    records: Iterable[EpisodeRecord], replicates: int = 1000, seed: int = 1
) -> tuple[float, float] | None:
    """Resample policy seeds and the same scenario IDs within each replicate."""
    rows = [r for r in records if r.status != "technical_error"]
    groups: dict[int | None, dict[tuple[int, int], EpisodeRecord]] = {}
    for row in rows:
        scenario = (row.layout_seed, row.radio_seed)
        group = groups.setdefault(row.policy_seed, {})
        if scenario in group:
            raise ValueError(f"duplicate scenario {scenario} for policy seed {row.policy_seed}")
        group[scenario] = row
    if len(groups) < 2:
        return None
    keys = list(groups)
    scenarios = list(groups[keys[0]])
    expected = set(scenarios)
    if any(set(group) != expected for group in groups.values()):
        raise ValueError("policy seeds have unequal paired scenario grids")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        selected = rng.choice(len(keys), size=len(keys), replace=True)
        scenario_sample = rng.choice(len(scenarios), size=len(scenarios), replace=True)
        seed_means = []
        for index in selected:
            group = groups[keys[int(index)]]
            seed_means.append(np.mean([group[scenarios[int(i)]].status == "success" for i in scenario_sample]))
        estimates.append(float(np.mean(seed_means)))
    return tuple(float(x) for x in np.quantile(estimates, [0.025, 0.975]))
