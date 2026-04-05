from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def assign_label(
    connected_next: int,
    snr_next: float,
    packet_loss_next: float,
    delay_next: float,
    tau_snr: float,
    tau_loss: float,
    tau_delay: float,
) -> int:
    if int(connected_next) != 1:
        return 0
    if snr_next < tau_snr:
        return 0
    if packet_loss_next > tau_loss:
        return 0
    if delay_next > tau_delay:
        return 0
    return 1


def build_labeled_edges(
    edges_features_csv: Path,
    output_csv: Path,
    tau_snr: float = 18.0,
    tau_loss: float = 0.10,
    tau_delay: float = 10.0,
) -> Path:
    edges = pd.read_csv(edges_features_csv)

    current_edges = edges[edges["connected"] == 1].copy()
    current_edges = current_edges[current_edges["time"] < edges["time"].max()].copy()

    next_shifted = edges[["time", "src", "dst", "connected", "snr", "packet_loss", "delay"]].copy()
    next_shifted["time"] = next_shifted["time"] - 1
    next_shifted = next_shifted.rename(
        columns={
            "connected": "connected_next",
            "snr": "snr_next",
            "packet_loss": "packet_loss_next",
            "delay": "delay_next",
        }
    )

    merged = current_edges.merge(next_shifted, on=["time", "src", "dst"], how="left")

    merged["connected_next"] = merged["connected_next"].fillna(0).astype(int)
    merged["snr_next"] = merged["snr_next"].fillna(float("-inf"))
    merged["packet_loss_next"] = merged["packet_loss_next"].fillna(1.0)
    merged["delay_next"] = merged["delay_next"].fillna(float("inf"))

    merged["label"] = [
        assign_label(cn, sn, pl, de, tau_snr, tau_loss, tau_delay)
        for cn, sn, pl, de in zip(
            merged["connected_next"],
            merged["snr_next"],
            merged["packet_loss_next"],
            merged["delay_next"],
        )
    ]
    merged["label_name"] = merged["label"].map({1: "stable", 0: "at_risk"})

    merged["tau_snr"] = tau_snr
    merged["tau_loss"] = tau_loss
    merged["tau_delay"] = tau_delay

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)
    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build t+1 edge labels.")
    parser.add_argument("--edges-features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tau-snr", type=float, default=18.0)
    parser.add_argument("--tau-loss", type=float, default=0.10)
    parser.add_argument("--tau-delay", type=float, default=10.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_labeled_edges(
        args.edges_features,
        args.output,
        tau_snr=args.tau_snr,
        tau_loss=args.tau_loss,
        tau_delay=args.tau_delay,
    )
    print(f"[OK] wrote {out}")
