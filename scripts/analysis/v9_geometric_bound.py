"""Exploratory post-hoc oracle for v9: how much of the disconnection is geometric.

Fits p(d) = P(rx_ratio >= 0.5 | 3D distance d) from every node pair of every
trace, then for each gradual episode computes the outage an ideal placement
would still have: relays evenly spaced on the source-destination segment,
moved instantly, so every hop has length D_t/(n-1).

  expected oracle outage = sum_t 1 - p(D_t/(n-1))^(n-1)   (hops independent)
  d50 bound              = #frames with D_t/(n-1) > d50   (p(d50) = 0.5)

ponytail: equal spacing on a straight line is optimal only if p(d) is
monotone and hops fade independently; an oracle that foresaw fading could do
slightly better. Not part of the pre-registered analysis.

Usage: python3 v9_geometric_bound.py <runs/test dir> [out.json]
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

BIN_M = 5.0
ARMS = ("route-only", "reactive-motion", "gnn-motion", "mlp-motion")


def load(path):
    return [json.loads(line)["frame"] for line in path.read_text().splitlines()]


def fit_link_model(traces):
    hits, totals = defaultdict(int), defaultdict(int)
    for frames in traces:
        for f in frames[1:]:
            p, rx = f["positions"], f["rx_ratio"]
            for i in range(len(p)):
                for j in range(i + 1, len(p)):
                    b = int(math.dist(p[i], p[j]) // BIN_M)
                    totals[b] += 1
                    hits[b] += rx[i][j] >= 0.5
    # running minimum from short to long distance -> monotone non-increasing p(d)
    model, best = {}, 1.0
    for b in range(max(totals) + 1):
        if totals[b] >= 30:
            best = min(best, hits[b] / totals[b])
        model[b] = best
    return model, totals


def p_link(model, d):
    b = int(d // BIN_M)
    return model.get(b, 0.0) if b <= max(model) else 0.0


def d50(model):
    return next(b * BIN_M for b in sorted(model) if model[b] < 0.5)


def oracle(model, frames, n):
    expected, beyond, half = 0.0, 0, d50(model)
    for f in frames[1:]:
        hop = math.dist(f["positions"][0], f["positions"][-1]) / (n - 1)
        expected += 1.0 - p_link(model, hop) ** (n - 1)
        beyond += hop > half
    return expected, beyond


def mean(xs):
    return sum(xs) / len(xs)


def main(root, out=None):
    episodes = {}
    for t in sorted(Path(root).glob("*/attempt-0/collector/episodes/*/*/episode_*/transitions.jsonl")):
        cell = t.parts[len(Path(root).parts)]
        block, arm = cell.split("__")
        episodes[(block, arm)] = load(t)
    model, _ = fit_link_model(episodes.values())
    rows = []
    for block in sorted({b for b, _ in episodes if "gradual" in b}):
        frames = episodes[(block, "route-only")]
        n = len(frames[0]["positions"])
        exp_out, d50_out = oracle(model, frames, n)
        row = dict(block=block, regime=block.split("-")[1], oracle_expected_s=round(exp_out, 2), d50_bound_s=d50_out)
        for arm in ARMS:
            row[arm] = sum(not f["source_dest_connected"] for f in episodes[(block, arm)][1:])
        rows.append(row)
    summary = {}
    for regime in sorted({r["regime"] for r in rows}):
        rs = [r for r in rows if r["regime"] == regime]
        summary[regime] = {k: round(mean([r[k] for r in rs]), 2) for k in ("oracle_expected_s", "d50_bound_s", *ARMS)}
    summary["balanced"] = {k: round(mean([summary[g][k] for g in summary]), 2) for k in summary[rows[0]["regime"]]}
    result = dict(d50_m=d50(model), bin_m=BIN_M, link_model={b * BIN_M: round(v, 4) for b, v in model.items()},
                  summary=summary, blocks=rows, note="exploratory post-hoc; not pre-registered")
    print(json.dumps(dict(d50_m=result["d50_m"], summary=summary), indent=1))
    if out:
        Path(out).write_text(json.dumps(result, indent=1))


def _check():
    model = {b: 1.0 if b < 30 else 0.0 for b in range(60)}  # p=1 below 150 m
    frames = [{"positions": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]},
              {"positions": [[0, 0, 0], [0, 0, 0], [200, 0, 0]]},   # hop 100 m -> connected
              {"positions": [[0, 0, 0], [0, 0, 0], [400, 0, 0]]}]   # hop 200 m -> cut
    assert oracle(model, frames, 3) == (1.0, 1), oracle(model, frames, 3)


if __name__ == "__main__":
    _check()
    main(*sys.argv[1:])
