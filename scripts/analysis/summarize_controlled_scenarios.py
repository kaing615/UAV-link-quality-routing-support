"""Validate and summarize paired controlled-scenario datasets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCENARIOS = ("baseline", "fast", "sparse", "dense")
RUN_PATTERN = re.compile(r"^stress_(baseline|fast|sparse|dense)_(rwp|gm)_s(\d+)$")


def _churn(edges: pd.DataFrame) -> float:
    connected = edges[edges["connected"].astype(int) == 1]
    by_time = {
        int(time): {tuple(sorted((int(row.src), int(row.dst)))) for row in group.itertuples()}
        for time, group in connected.groupby("time")
    }
    times = sorted(edges["time"].astype(int).unique())
    values = []
    for previous, current in zip(times, times[1:], strict=False):
        before, after = by_time.get(previous, set()), by_time.get(current, set())
        union = before | after
        values.append(len(before ^ after) / max(len(union), 1))
    return float(np.mean(values)) if values else 0.0


def collect_inventory(raw_root: Path, graph_root: Path) -> pd.DataFrame:
    rows = []
    for raw_dir in sorted(path for path in raw_root.glob("stress_*") if path.is_dir()):
        match = RUN_PATTERN.match(raw_dir.name)
        if not match:
            continue
        scenario, mobility_tag, seed_text = match.groups()
        graph_dir = graph_root / raw_dir.name
        raw_required = [raw_dir / name for name in ("scenario.json", "nodes.csv", "edges.csv")]
        graph_required = [
            graph_dir / "graph_dataset" / f"{split}.pt" for split in ("train", "val", "test")
        ]
        raw_complete = all(path.exists() and path.stat().st_size > 0 for path in raw_required)
        graph_complete = all(path.exists() and path.stat().st_size > 0 for path in graph_required)
        row: dict[str, object] = {
            "run_name": raw_dir.name,
            "scenario": scenario,
            "seed": int(seed_text),
            "mobility": "random-waypoint" if mobility_tag == "rwp" else "gauss-markov",
            "raw_complete": raw_complete,
            "graph_complete": graph_complete,
            "complete": raw_complete and graph_complete,
        }
        if raw_complete:
            metadata = json.loads((raw_dir / "scenario.json").read_text(encoding="utf-8"))
            nodes = pd.read_csv(raw_dir / "nodes.csv", usecols=["speed", "degree"])
            edges = pd.read_csv(raw_dir / "edges.csv", usecols=["time", "src", "dst", "connected"])
            row.update(
                {
                    "num_uavs": int(metadata["num_uavs"]),
                    "time_steps": int(metadata["time_steps"]),
                    "comm_range": float(metadata["comm_range"]),
                    "configured_speed_min": float(metadata["rwp_speed_range"][0]),
                    "configured_speed_max": float(metadata["rwp_speed_range"][1]),
                    "mean_speed": float(nodes["speed"].mean()),
                    "p95_speed": float(nodes["speed"].quantile(0.95)),
                    "mean_degree": float(nodes["degree"].mean()),
                    "isolated_rate": float((nodes["degree"] == 0).mean()),
                    "connected_edge_rate": float(edges["connected"].mean()),
                    "topology_churn": _churn(edges),
                }
            )
        labels_path = graph_dir / "features" / "edges_labeled.csv"
        if labels_path.exists():
            labels = pd.read_csv(labels_path, usecols=["label"])["label"]
            row["positive_ratio"] = float(labels.mean()) if len(labels) else np.nan
            row["labeled_edges"] = int(len(labels))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "mean_speed",
        "p95_speed",
        "mean_degree",
        "isolated_rate",
        "connected_edge_rate",
        "topology_churn",
        "positive_ratio",
        "labeled_edges",
    ]
    rows = []
    for scenario in SCENARIOS:
        group = inventory[inventory["scenario"] == scenario]
        row: dict[str, object] = {
            "scenario": scenario,
            "runs": int(len(group)),
            "complete_runs": int(group["complete"].sum()) if not group.empty else 0,
            "unique_seeds": int(group["seed"].nunique()) if not group.empty else 0,
            "rwp_runs": int((group["mobility"] == "random-waypoint").sum()) if not group.empty else 0,
            "gm_runs": int((group["mobility"] == "gauss-markov").sum()) if not group.empty else 0,
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean()) if metric in group and len(group) else np.nan
            row[f"{metric}_std"] = float(group[metric].std()) if metric in group and len(group) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_checks(
    inventory: pd.DataFrame,
    summary: pd.DataFrame,
    expected_per_scenario: int,
) -> pd.DataFrame:
    values = summary.set_index("scenario")
    rows = []
    for scenario in SCENARIOS:
        complete = int(values.loc[scenario, "complete_runs"])
        rows.append(
            {
                "check": f"{scenario}_complete_runs",
                "observed": complete,
                "expected": expected_per_scenario,
                "status": "PASS" if complete == expected_per_scenario else "FAIL",
            }
        )
    paired = inventory[inventory["complete"]].groupby(["seed", "mobility"])["scenario"].nunique()
    paired_count = int((paired == len(SCENARIOS)).sum())
    rows.append(
        {
            "check": "complete_paired_seed_sets",
            "observed": paired_count,
            "expected": expected_per_scenario,
            "status": "PASS" if paired_count == expected_per_scenario else "FAIL",
        }
    )
    comparisons = [
        ("fast_speed_above_baseline", values.loc["fast", "mean_speed_mean"], values.loc["baseline", "mean_speed_mean"], ">"),
        ("fast_churn_above_baseline", values.loc["fast", "topology_churn_mean"], values.loc["baseline", "topology_churn_mean"], ">"),
        ("sparse_degree_below_baseline", values.loc["sparse", "mean_degree_mean"], values.loc["baseline", "mean_degree_mean"], "<"),
        ("dense_degree_above_baseline", values.loc["dense", "mean_degree_mean"], values.loc["baseline", "mean_degree_mean"], ">"),
    ]
    for name, observed, reference, operator in comparisons:
        passed = observed > reference if operator == ">" else observed < reference
        rows.append(
            {
                "check": name,
                "observed": observed,
                "expected": f"{operator} {reference:.6f}",
                "status": "PASS" if bool(passed) else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def _write_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    labels = summary["scenario"].str.title().tolist()
    colors = ["#7f7f7f", "#d62728", "#4c78a8", "#2ca02c"]
    specs = [
        ("mean_speed", "Mean speed (m/s)"),
        ("mean_degree", "Mean node degree"),
        ("connected_edge_rate", "Connected-edge rate"),
        ("topology_churn", "Topology churn"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (metric, title) in zip(axes.ravel(), specs, strict=True):
        ax.bar(labels, summary[f"{metric}_mean"], color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Controlled-scenario manipulation checks")
    fig.tight_layout()
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "scenario_manipulation_checks.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures / "scenario_manipulation_checks.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_report(
    raw_root: Path,
    graph_root: Path,
    output_dir: Path,
    *,
    expected_per_scenario: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory = collect_inventory(raw_root, graph_root)
    if inventory.empty:
        raise FileNotFoundError(f"No controlled runs found under {raw_root}")
    summary = summarize_inventory(inventory)
    checks = build_checks(inventory, summary, expected_per_scenario)
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_dir / "run_inventory.csv", index=False)
    summary.to_csv(output_dir / "scenario_summary.csv", index=False)
    checks.to_csv(output_dir / "manipulation_checks.csv", index=False)
    _write_figure(summary, output_dir)
    status = "PASS" if (checks["status"] == "PASS").all() else "FAIL"
    report = (
        "# Controlled Scenario Execution Report\n\n"
        f"Overall status: **{status}**\n\n"
        f"Expected runs per scenario: **{expected_per_scenario}**  \n"
        f"Inventory rows: **{len(inventory)}**\n\n"
        "## Scenario summary\n\n"
        f"{summary.to_markdown(index=False)}\n\n"
        "## Acceptance checks\n\n"
        f"{checks.to_markdown(index=False)}\n"
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return inventory, summary, checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw_snapshots"))
    parser.add_argument("--graph-root", type=Path, default=Path("data/graph_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/controlled_scenarios"))
    parser.add_argument("--expected-per-scenario", type=int, default=100)
    parser.add_argument("--fail-on-check", action="store_true")
    args = parser.parse_args()
    _, _, checks = generate_report(
        args.raw_root,
        args.graph_root,
        args.output_dir,
        expected_per_scenario=args.expected_per_scenario,
    )
    failed = int((checks["status"] == "FAIL").sum())
    print(f"[CONTROLLED SCENARIOS] {'PASS' if failed == 0 else 'FAIL'}: {failed} failed checks")
    print(f"Report: {args.output_dir / 'REPORT.md'}")
    if args.fail_on_check and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
