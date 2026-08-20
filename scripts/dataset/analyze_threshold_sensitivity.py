"""Measure QoS-label sensitivity across SNR/loss/delay threshold grids."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

SNR_VALUES = (15.0, 18.0, 21.0)
LOSS_VALUES = (0.05, 0.10, 0.20)
DELAY_VALUES = (5.0, 10.0, 20.0)
REFERENCE = (18.0, 0.10, 10.0)


def _labels(frame: pd.DataFrame, tau_snr: float, tau_loss: float, tau_delay: float) -> pd.Series:
    return (
        frame["connected_next"].eq(1)
        & frame["snr_next"].ge(tau_snr)
        & frame["packet_loss_next"].le(tau_loss)
        & frame["delay_next"].le(tau_delay)
    )


def evaluate_threshold_grid(
    labeled: pd.DataFrame,
    run_name: str,
    horizon: int,
    snr_values: tuple[float, ...] | list[float] = SNR_VALUES,
    loss_values: tuple[float, ...] | list[float] = LOSS_VALUES,
    delay_values: tuple[float, ...] | list[float] = DELAY_VALUES,
) -> pd.DataFrame:
    """Return split-level label prevalence and agreement with the reference definition."""
    required = {
        "split",
        "connected_next",
        "snr_next",
        "packet_loss_next",
        "delay_next",
    }
    missing = required.difference(labeled.columns)
    if missing:
        raise ValueError(f"labeled data missing columns: {sorted(missing)}")

    rows = []
    for split, frame in labeled[labeled["split"] != "purged"].groupby("split", sort=False):
        reference = _labels(frame, *REFERENCE)
        for tau_snr, tau_loss, tau_delay in product(snr_values, loss_values, delay_values):
            labels = _labels(frame, tau_snr, tau_loss, tau_delay)
            rows.append(
                {
                    "run_name": run_name,
                    "horizon": horizon,
                    "split": split,
                    "tau_snr": tau_snr,
                    "tau_loss": tau_loss,
                    "tau_delay": tau_delay,
                    "n_samples": len(frame),
                    "positive_ratio": float(labels.mean()),
                    "agreement_with_reference": float(labels.eq(reference).mean()),
                    "connected_ratio": float(frame["connected_next"].eq(1).mean()),
                    "snr_pass_ratio": float(frame["snr_next"].ge(tau_snr).mean()),
                    "loss_pass_ratio": float(frame["packet_loss_next"].le(tau_loss).mean()),
                    "delay_pass_ratio": float(frame["delay_next"].le(tau_delay).mean()),
                }
            )
    return pd.DataFrame(rows)


def analyze(pilot_root: Path, output_dir: Path) -> tuple[Path, Path]:
    frames = []
    for labeled_path in sorted(pilot_root.glob("qos/k*/**/features/edges_labeled.csv")):
        run_dir = labeled_path.parents[1]
        split_path = run_dir / "splits" / "time_splits.csv"
        if not split_path.exists():
            continue
        horizon = int(labeled_path.parents[2].name.removeprefix("k"))
        labeled = pd.read_csv(labeled_path)
        splits = pd.read_csv(split_path)
        merged = labeled.merge(splits[["time", "split"]], on="time", validate="many_to_one")
        frames.append(evaluate_threshold_grid(merged, run_dir.name, horizon))
    if not frames:
        raise FileNotFoundError(f"No pilot QoS labels found under {pilot_root}")

    detailed = pd.concat(frames, ignore_index=True)
    group_cols = ["horizon", "split", "tau_snr", "tau_loss", "tau_delay"]
    summary = (
        detailed.groupby(group_cols)
        .agg(
            n_runs=("run_name", "nunique"),
            n_samples=("n_samples", "sum"),
            positive_ratio_mean=("positive_ratio", "mean"),
            positive_ratio_std=("positive_ratio", "std"),
            agreement_mean=("agreement_with_reference", "mean"),
            agreement_std=("agreement_with_reference", "std"),
            connected_ratio_mean=("connected_ratio", "mean"),
            snr_pass_ratio_mean=("snr_pass_ratio", "mean"),
            loss_pass_ratio_mean=("loss_pass_ratio", "mean"),
            delay_pass_ratio_mean=("delay_pass_ratio", "mean"),
        )
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = output_dir / "threshold_sensitivity_by_run.csv"
    summary_path = output_dir / "threshold_sensitivity_summary.csv"
    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)
    return detailed_path, summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=Path("data/multihorizon"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/threshold_sensitivity"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detailed_path, summary_path = analyze(args.pilot_root, args.output_dir)
    print(f"[OK] detailed: {detailed_path}")
    print(f"[OK] summary : {summary_path}")
