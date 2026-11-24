"""Read ONLY archived v7 HOLD observations into a bound development dataset."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "service-v8-development-data-v1", "source_study": "v7",
                "role": "Historical v7 data are development only for prospective v8.", "episodes": []}
    destination = args.output / "development.jsonl.gz"
    with gzip.open(destination, "wt", encoding="utf-8") as stream:
        for phase, split in (("test", "train"), ("validation", "early_stop")):
            validation = args.study_root / f"runs/{phase}-validation.json"
            report = json.loads(validation.read_text())
            assert report["complete"]
            for row in report["rows"]:
                if row["policy"] != "hold":
                    continue
                assert row["status"] == "ok"
                episode = args.study_root / "runs" / phase / row["cell_id"] / f"attempt-{row['attempt']}" / row["episode_path"]
                trace = episode / "transitions.jsonl"
                assert digest(trace) == row["trace_sha256"]
                frames = [json.loads(line)["frame"] for line in trace.read_text().splitlines()]
                assert [frame["step"] for frame in frames] == list(range(len(frames)))
                entry = {key: row[key] for key in ("cell_id", "layout_seed", "radio_seed", "regime", "context", "trace_sha256")}
                entry.update(split=split, frames=len(frames), source_validation_sha256=digest(validation))
                manifest["episodes"].append(entry)
                stream.write(json.dumps(dict(entry, frames=frames), separators=(",", ":")) + "\n")
    assert sum(e["split"] == "train" for e in manifest["episodes"]) == 48
    assert sum(e["split"] == "early_stop" for e in manifest["episodes"]) == 20
    manifest["data_sha256"] = digest(destination)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"episodes": len(manifest["episodes"]), "data_sha256": manifest["data_sha256"],
                      "bytes": destination.stat().st_size}))


if __name__ == "__main__":
    main()
