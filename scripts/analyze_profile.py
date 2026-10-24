"""Analyze Nsight exports without confusing host scopes and CUDA time."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sqlite3


def union_ns(intervals):
    total, right = 0, None
    for start, end in sorted(intervals):
        if right is None or start > right:
            total += end - start
        elif end > right:
            total += end - right
        right = end if right is None else max(right, end)
    return total


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def analyze_sqlite(path, output):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    scopes = []
    for start, end, text, text_id, tid in db.execute(
            "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        name = text or strings.get(text_id, "")
        if name in ("rollout", "learner_forward", "teacher_forward",
                    "full_vocabulary_kl", "reference_forward", "backward", "optimizer", "update_42"):
            scopes.append(dict(start=start, end=end, name=name, tid=tid))
    update = next(s for s in scopes if s["name"] == "update_42")
    origin, limit = update["start"], update["end"]
    # Attribute GPU work to its correlated launch API and the launching thread.
    inner = sorted((s for s in scopes if s["name"] != "update_42"), key=lambda s: s["start"])
    starts = [s["start"] for s in inner]
    kernel_stats = defaultdict(lambda: [0, 0, 0])
    phase_stats = defaultdict(lambda: [0, 0])
    intervals, representative = [], []
    bins = Counter()
    selected = next(s for s in inner if s["name"] == "learner_forward")
    selected_end = next(s["end"] for s in inner
                        if s["name"] == "backward" and s["start"] > selected["start"])
    query = """
        SELECT k.start, k.end, k.shortName, k.streamId, r.start, r.globalTid
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
          ON r.correlationId = k.correlationId
         AND (r.globalTid & -16777216) = k.globalPid
        ORDER BY k.start
    """
    for start, end, name_id, stream, launch, tid in db.execute(query):
        name = strings[name_id]
        duration = end - start
        stat = kernel_stats[name]
        stat[0] += 1
        stat[1] += duration
        stat[2] = max(stat[2], duration)
        phase = "unattributed"
        if launch is not None:
            index = bisect_right(starts, launch) - 1
            if index >= 0 and launch < inner[index]["end"]:
                phase = inner[index]["name"]
                if tid != inner[index]["tid"]:
                    phase += "_cross_thread_temporal"
        phase_stats[phase][0] += 1
        phase_stats[phase][1] += duration
        clipped = (max(start, origin), min(end, limit))
        if clipped[1] > clipped[0]:
            intervals.append(clipped)
            bins[int((clipped[0] - origin) / 10_000_000)] += 1
        if selected["start"] <= start < selected_end:
            representative.append({"start_ms": (start - selected["start"]) / 1e6,
                                   "duration_us": duration / 1e3, "phase": phase,
                                   "stream": stream, "kernel": name})
    total_kernels = sum(s[0] for s in kernel_stats.values())
    actual_rows = db.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    if total_kernels != actual_rows:
        raise ValueError("Runtime correlation duplicated kernel rows")
    copies = list(db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY"))
    memsets = list(db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMSET"))
    all_intervals = intervals + [(max(s, origin), min(e, limit)) for s, e in copies + memsets
                                 if min(e, limit) > max(s, origin)]
    api = [{"name": strings[name], "calls": count, "host_seconds": duration / 1e9}
           for name, count, duration in db.execute(
               "SELECT nameId, COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_RUNTIME "
               "GROUP BY nameId ORDER BY SUM(end-start) DESC")]
    top = [{"name": name, "calls": count, "summed_device_seconds": duration / 1e9,
            "mean_us": duration / count / 1e3, "max_us": maximum / 1e3}
           for name, (count, duration, maximum) in
           sorted(kernel_stats.items(), key=lambda x: x[1][1], reverse=True)]
    summary = {
        "trace_sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
        "host_update_seconds": (limit - origin) / 1e9,
        "cuda_kernel_count": actual_rows,
        "summed_kernel_seconds": sum(s[1] for s in kernel_stats.values()) / 1e9,
        "kernel_union_seconds_in_update": union_ns(intervals) / 1e9,
        "kernel_copy_memset_union_seconds_in_update": union_ns(all_intervals) / 1e9,
        "phase_device_launch_attribution": {
            name: {"kernels": count, "summed_device_seconds": duration / 1e9}
            for name, (count, duration) in phase_stats.items()},
        "host_scopes": [{"name": s["name"], "start_seconds": (s["start"] - origin) / 1e9,
                         "end_seconds": (s["end"] - origin) / 1e9} for s in inner],
        "top_kernels": top[:30], "cuda_api": api,
        "representative_window": {"start_seconds": (selected["start"] - origin) / 1e9,
                                  "end_seconds": (selected_end - origin) / 1e9},
        "boundary": "CUDA interval union is activity coverage, not SM occupancy or HBM utilization",
    }
    write_json(output / (path.parent.name + "-cuda-summary.json"), summary)
    with (output / (path.parent.name + "-representative-kernels.csv")).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=representative[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(representative)
    with (output / (path.parent.name + "-launch-density.csv")).open("w") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["start_seconds", "kernels_started_in_10ms"])
        for index in range(int((limit - origin) / 10_000_000) + 1):
            writer.writerow([index / 100, bins[index]])
    db.close()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = {}
    for path in sorted(args.input.iterdir()):
        host_file = path / "observation/host-spans.json"
        if not host_file.exists():
            continue
        events = json.loads(host_file.read_text())
        host = defaultdict(float)
        for event in events:
            host[event["name"]] += event["host_wall_seconds"]
        metrics = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        contract = json.loads((path / "last-rollout-contract.json").read_text())
        cases[path.name] = {
            "host_seconds": dict(host), "metrics": metrics,
            "rollout_token_sha256": contract["token_sha256"],
            "rollout_mask_sha256": contract["mask_sha256"],
            "sample_ids_sha256": hashlib.sha256(json.dumps(contract["sample_ids"]).encode()).hexdigest(),
            "restoration": json.loads((path / "restoration-0040.json").read_text()),
            "configuration": json.loads((path / "observation/configuration.json").read_text()),
        }
        if (path / "capture.sqlite").exists():
            cases[path.name]["cuda"] = analyze_sqlite(path / "capture.sqlite", args.output)
    write_json(args.output / "measurements.json", cases)
    print(json.dumps({k: v["host_seconds"] for k, v in cases.items()}, indent=2))


if __name__ == "__main__":
    main()
