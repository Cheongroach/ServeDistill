"""Four-method DEV ablation with one common adapter and fresh Adam per method."""
import argparse
import dataclasses
import gc
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from servedistill.advantage_contracts import file_digest, verify_labels
from servedistill.data import load_rows
from servedistill.runtime import Runtime, write_json
from servedistill.scheduling import Config, schedule
from scripts.worker import publish


def main():
    parser = argparse.ArgumentParser()
    for key in ("student", "teacher", "data", "initial", "labels", "protocol", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["scope"] != "development_only" or args.seed not in protocol["seeds"]:
        raise ValueError("This entry point never evaluates holdouts")
    repo = Path(__file__).resolve().parents[1]
    bindings = {
        "train_sha256": args.data / "train.jsonl",
        "dev_sha256": args.data / "dev.jsonl",
        "labels_sha256": args.labels,
        "initial_adapter_sha256": args.initial / "adapter_model.safetensors",
        "initial_config_sha256": args.initial / "adapter_config.json",
    }
    if any(file_digest(path) != protocol[key] for key, path in bindings.items()):
        raise ValueError("Immutable DEV inputs or common adapter changed")
    for relative, expected in protocol["source_files"].items():
        if file_digest(repo / relative) != expected:
            raise ValueError("Frozen DEV execution source changed: " + relative)
    verify_labels(args.labels, args.data / "train.jsonl")
    args.output.mkdir(parents=True, exist_ok=False)
    dev = load_rows(args.data / "dev.jsonl")
    dev = [r for r in dev if r["domain"] in ("search", "relevance", "substitute")] + [
        r for domain in ("math", "instruction")
        for r in [r for r in dev if r["domain"] == domain][:100]]
    if len(dev) != 1100 or len({r["sample_id"] for r in dev}) != 1100:
        raise ValueError("Full DEV coverage must remain 1100 distinct rows")
    train = load_rows(args.data / "train.jsonl")
    results, schedules = {}, {}
    for label, spec in protocol["methods"].items():
        config = Config(**{**spec, "seed": args.seed})
        schedules[label] = [[r["sample_id"] for r in batch] for batch in schedule(train, config)]
    if len({json.dumps(value) for value in schedules.values()}) != 1:
        raise ValueError("Method prompt schedules differ")
    write_json(args.output / "binding.json", {
        "scope": "development_only", "seed": args.seed,
        "protocol_sha256": file_digest(args.protocol), "holdout_observed": False,
        "fresh_adam_per_method": True, "common_initial_adapter": protocol["initial_adapter_sha256"],
        "prompt_schedule_sha256": __import__("hashlib").sha256(
            json.dumps(next(iter(schedules.values()))).encode()).hexdigest(),
    })
    initial = Runtime(str(args.student), adapter_path=str(args.initial))
    results["initial"] = initial.evaluate(dev, args.output, "initial-dev", batch_size=8)
    del initial
    gc.collect()
    torch.cuda.empty_cache()
    events = []
    for label, spec in protocol["methods"].items():
        arm = args.output / label
        arm.mkdir()
        config = Config(**{**spec, "seed": args.seed})
        write_json(arm / "config.json", dataclasses.asdict(config))
        argv = [sys.executable, "-m", "servedistill.train", "--student", str(args.student),
                "--data", str(args.data), "--config", str(arm / "config.json"),
                "--adapter", str(args.initial), "--output", str(arm / "train")]
        if config.objective == "aopd":
            argv += ["--teacher", str(args.teacher)]
        if config.objective == "offline_sft":
            argv += ["--teacher-labels", str(args.labels)]
        started = time.time()
        with (arm / "training.log").open("w") as stream:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT)
            while True:
                try:
                    returncode = process.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    if args.destination:
                        publish(args.output, args.destination)
        events.append({"method": label, "started": started, "ended": time.time(),
                       "returncode": returncode})
        write_json(args.output / "execution.json", events)
        if args.destination:
            publish(args.output, args.destination)
        if returncode:
            raise RuntimeError("Development method failed: " + label)
        completed = json.loads((arm / "train/completed.json").read_text())
        if not completed["completed"] or completed["steps"] != config.steps:
            raise ValueError("Final complete update budget changed")
        checkpoint = arm / f"train/checkpoint-{config.steps:04d}"
        runtime = Runtime(str(args.student), adapter_path=str(checkpoint))
        results[label] = runtime.evaluate(dev, arm, "final-dev", batch_size=8)
        del runtime
        gc.collect()
        torch.cuda.empty_cache()
        write_json(arm / "evaluated-adapter.json", {
            "precision": "bf16", "step": config.steps,
            "adapter_sha256": file_digest(checkpoint / "adapter_model.safetensors"),
            "adapter_config_sha256": file_digest(checkpoint / "adapter_config.json"),
            "source": "fixed_final_update", "dev_rows": len(dev),
        })
        if args.destination:
            publish(args.output, args.destination)
    write_json(args.output / "completed.json", {
        "completed": True, "scope": "development_only", "holdout_observed": False,
        "seed": args.seed, "protocol_sha256": file_digest(args.protocol),
        "quality_advantage_supported": False, "summaries": results, "events": events,
    })
    if args.destination:
        publish(args.output, args.destination)


if __name__ == "__main__":
    main()
