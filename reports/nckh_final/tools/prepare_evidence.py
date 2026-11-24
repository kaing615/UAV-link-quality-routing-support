"""Audit frozen v7 results and prepare descriptive report figures; no simulations."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT = Path(__file__).resolve().parents[1]
ROOT = REPORT.parents[1]
RESULTS = ROOT / "reports/uav_service_v7/results"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    analysis_path = RESULTS / "analysis/test/analysis.json"
    analysis = read(analysis_path)
    digest = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
    expected = read(analysis_path.with_name("manifest.json"))["files"]["analysis.json"]
    assert digest == expected == "0eb750b80ceebc9db0b45deb007cfcbbddf464027a1c194acdfa66502dfd7950"
    validations = {phase: read(RESULTS / f"runs/{phase}-validation.json")
                   for phase in ("tuning", "validation", "test")}
    for phase, count in (("tuning", 72), ("validation", 100), ("test", 240)):
        validation = validations[phase]
        assert validation["complete"] and validation["ok_cells"] == validation["planned_cells"] == count
        assert len(validation["rows"]) == count
        assert all(row["status"] == "ok" for row in validation["rows"])
    assert analysis["route_integrity_ok"]
    assert len(analysis["route_integrity"]) == 96
    assert all(item["ok"] for item in analysis["route_integrity"].values())
    assert all(analysis[key] for key in ("flags_ok", "transmissions_ok", "background_ok"))
    assert not analysis["startup_blackouts"]
    assert all(item["success"] and item["interpretable"] for item in analysis["fixed_sequence"])

    groups = defaultdict(dict)
    for row in validations["test"]["rows"]:
        key = (row["context"], row["regime"], row["layout_seed"], row["radio_seed"])
        assert row["policy"] not in groups[key]
        groups[key][row["policy"]] = row["actual_udp_pdr"]
    arms = ("hold", "olsr-etx", "olsr-etx-fast", "lq-route", "lq-route-delay")
    assert all(set(group) == set(arms) for group in groups.values())
    assert Counter((key[0], key[1]) for key in groups) == {
        ("gradual", "n5"): 20, ("gradual", "n7"): 20,
        ("static", "n5"): 4, ("static", "n7"): 4,
    }

    paired = []
    for key, group in sorted(groups.items()):
        if key[0] == "gradual":
            paired.append(dict(context=key[0], regime=key[1], layout_seed=key[2], radio_seed=key[3],
                               route_minus_fast_pp=100 * (group["lq-route"] - group["olsr-etx-fast"]),
                               delay_minus_fast_pp=100 * (group["lq-route-delay"] - group["olsr-etx-fast"])))
    for arm, field in (("lq-route", "route_minus_fast_pp"), ("lq-route-delay", "delay_minus_fast_pp")):
        locked = next(item["contrast"] for item in analysis["fixed_sequence"] if item["contrast"]["arm"] == arm)
        assert abs(sum(row[field] for row in paired) / 4000 - locked["point"]) < 1e-12

    evidence_dir = REPORT / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    with (evidence_dir / "v7-paired.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    overhead = analysis["overhead"]
    audit = {
        "analysis_sha256": digest,
        "phases": {phase: value["ok_cells"] for phase, value in validations.items()},
        "test_scenario_blocks": len(groups), "gradual_blocks": len(paired),
        "route_integrity_ok": 96,
        "method_cost_ratio_route_to_fast": overhead["lq-route"]["method_total_Bps"] / overhead["olsr-etx-fast"]["method_total_Bps"],
        "method_cost_ratio_route_to_etx": overhead["lq-route"]["method_total_Bps"] / overhead["olsr-etx"]["method_total_Bps"],
        "paired_wins": {field: sum(row[field] > 0 for row in paired) for field in ("route_minus_fast_pp", "delay_minus_fast_pp")},
        "fixed_sequence": analysis["fixed_sequence"],
        "static_descriptive": analysis["static_descriptive"],
        "means_gradual": {arm: sum(group[arm] for key, group in groups.items() if key[0] == "gradual") / 40 for arm in arms},
        "note": "Descriptive figures from frozen results; no new inference, seed selection or simulations.",
    }
    (evidence_dir / "v7-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "axes.axisbelow": True})
    images = REPORT / "images"
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    for ax, regime in zip(axes, ("n5", "n7")):
        subset = [row for row in paired if row["regime"] == regime]
        x = list(range(1, 21))
        for field, label, color, marker in (("route_minus_fast_pp", "Ghim tuyến", "#167b68", "o"),
                                            ("delay_minus_fast_pp", "Ghim tuyến, trễ 2 s", "#c36d25", "s")):
            ax.scatter(x, [row[field] for row in subset], s=27, marker=marker, label=label, color=color, alpha=.85)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_title(f"{regime}: 20 khối kịch bản gradual")
        ax.set_xlabel("Kịch bản (xếp theo layout seed)")
        ax.set_xticks([1, 5, 10, 15, 20])
    axes[0].set_ylabel("Chênh PDR so với ETX-fast (điểm %)")
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(images / f"fig_v7_paired.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    labels = ["OLSR", "ETX", "ETX-fast", "Ghim tuyến", "Ghim tuyến\ntrễ 2 s"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    bottom = [0.0] * 5
    for metric, label, color, hatch in (("olsr_Bps", "OLSR đo tại tầng IP", "#5077ad", None),
                                        ("probe_Bps", "Probe được tính cho phương pháp", "#348773", None),
                                        ("central_estimate_Bps", "Báo cáo/lệnh: ước lượng", "#dfa85a", "///")):
        values = [overhead[arm].get(metric, 0) / 1000 if metric == "olsr_Bps" or arm.startswith("lq-route") else 0 for arm in arms]
        ax.bar(labels, values, bottom=bottom, color=color, label=label, hatch=hatch, width=.62)
        bottom = [b + v for b, v in zip(bottom, values)]
    for x, value in enumerate(bottom):
        ax.text(x, value + .08, f"{value:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("Chi phí phương pháp (kB/s; 1 kB = 1.000 B)")
    ax.set_ylim(0, 7.5)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(images / f"fig_v7_overhead.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({key: value for key, value in audit.items() if key not in ("fixed_sequence", "static_descriptive")}, ensure_ascii=False, indent=2))
    print("Evidence and figures written to", REPORT)


if __name__ == "__main__":
    main()
