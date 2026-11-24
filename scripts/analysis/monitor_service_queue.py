"""Read-only queue status; never launch/retry anything. Exit 1 on a failed cell."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--phase", required=True)
args = parser.parse_args()
evidence = args.root / "evidence" / ("queue-" + args.phase)
current = json.loads((evidence / "current.json").read_text())
cells = current["cells"]
counts = Counter()
errors = []
for name, state in cells.items():
    if state["state"] == "wrapper_finished":
        counts["ok" if state["direct_wrapper_exit"] == 0 else "failed"] += 1
        if state["direct_wrapper_exit"] != 0:
            result = args.root / "runs" / args.phase / name / "attempt-0/result.json"
            errors.append(dict(cell_id=name, error=json.loads(result.read_text()).get("error") if result.exists() else state))
    else:
        counts[state["state"]] += 1
starts = [value["started_unix"] for value in cells.values() if "started_unix" in value]
ends = [value["finished_unix"] for value in cells.values() if "finished_unix" in value]
finished = evidence / "queue-finished.json"
print(json.dumps(dict(phase=args.phase, counts=dict(counts), errors=errors, finished=finished.exists(),
                      elapsed_s=round((max(ends) if finished.exists() and ends else time.time()) - min(starts), 1) if starts else 0,
                      completion=json.loads(finished.read_text()) if finished.exists() else None)))
raise SystemExit(1 if errors or counts["queue_error"] or counts["not_started_after_queue_failure"] else 0)
