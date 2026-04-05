from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_time_split(
    edges_labeled_csv: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Path:
    df = pd.read_csv(edges_labeled_csv)
    times = sorted(df["time"].unique().tolist())
    n = len(times)

    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, n - n_train - n_val)

    if n_train + n_val + n_test > n:
        n_test = n - n_train - n_val

    if n_test <= 0:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    train_times = times[:n_train]
    val_times = times[n_train:n_train + n_val]
    test_times = times[n_train + n_val:]

    train_set = set(train_times)
    val_set = set(val_times)

    split_df = pd.DataFrame({"time": times})
    split_df["split"] = split_df["time"].map(
        lambda t: "train" if t in train_set else ("val" if t in val_set else "test")
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    split_csv = output_dir / "time_splits.csv"
    split_df.to_csv(split_csv, index=False)

    for name, values in [("train", train_times), ("val", val_times), ("test", test_times)]:
        (output_dir / f"{name}_times.txt").write_text("\n".join(str(v) for v in values), encoding="utf-8")

    return split_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split time steps into train/val/test.")
    parser.add_argument("--edges-labeled", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out = build_time_split(args.edges_labeled, args.output_dir, args.train_ratio, args.val_ratio)
    print(f"[OK] wrote {out}")
