"""Run one preregistered DEV anchor weight on the original frozen training source."""
import argparse
import dataclasses
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from scripts.worker import publish
from servedistill.advantage_contracts import file_digest
from servedistill.data import load_rows
from servedistill.runtime import Runtime, write_json
from servedistill.scheduling import Config, schedule


def variant_config(base, extension, variant, seed):
    if (extension["scope"] != "development_only" or extension["holdout_files_packaged"]
            or seed not in extension["seeds"] or seed not in base["seeds"]
            or variant not in extension["variants"]):
        raise ValueError("Unknown or non-DEV variant")
    weight = extension["variants"][variant]["anchor_ce_weight"]
    if weight not in (0.1, 1.0):
        raise ValueError("Undeclared anchor coefficient")
    return Config(**{**base["methods"]["aopd_anchor"], "seed": seed,
                     "anchor_ce_weight": weight})


def main():
    parser = argparse.ArgumentParser()
    for key in ("student", "teacher", "data", "initial", "protocol", "extension", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    base = json.loads(args.protocol.read_text())
    extension = json.loads(args.extension.read_text())
    repo = Path(__file__).resolve().parents[1]
    if file_digest(args.protocol) != extension["base_protocol_sha256"]:
        raise ValueError("Original frozen protocol changed")
    for relative, expected in {**base["source_files"], **extension["source_files"]}.items():
        if file_digest(repo / relative) != expected:
            raise ValueError("Frozen execution source differs: " + relative)
    for key, path in {
        "train_sha256": args.data / "train.jsonl", "dev_sha256": args.data / "dev.jsonl",
        "initial_adapter_sha256": args.initial / "adapter_model.safetensors",
        "initial_config_sha256": args.initial / "adapter_config.json",
    }.items():
        if file_digest(path) != base[key]:
            raise ValueError("Original DEV input binding changed: " + key)
    config = variant_config(base, extension, args.variant, args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", dataclasses.asdict(config))
    expected_schedule = [[r["sample_id"] for r in batch]
                         for batch in schedule(load_rows(args.data / "train.jsonl"), config)]
    write_json(args.output / "binding.json", {
        "scope": "development_only", "holdout_observed": False, "seed": args.seed,
        "variant": args.variant, "base_protocol_sha256": file_digest(args.protocol),
        "extension_sha256": file_digest(args.extension), "fresh_adam": True,
        "common_initial_adapter": base["initial_adapter_sha256"],
        "prompt_schedule_sha256": hashlib.sha256(json.dumps(expected_schedule).encode()).hexdigest(),
    })
    argv = [sys.executable, "-m", "servedistill.train", "--student", str(args.student),
            "--teacher", str(args.teacher), "--data", str(args.data),
            "--config", str(args.output / "config.json"), "--adapter", str(args.initial),
            "--output", str(args.output / "train")]
    started = time.time()
    with (args.output / "training.log").open("w") as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT)
        while True:
            try:
                returncode = process.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                if args.destination:
                    publish(args.output, args.destination)
    write_json(args.output / "execution.json", {
        "started": started, "ended": time.time(), "returncode": returncode, "argv": argv})
    if args.destination:
        publish(args.output, args.destination)
    if returncode:
        raise RuntimeError("Anchor weight training failed")
    completed = json.loads((args.output / "train/completed.json").read_text())
    actual = json.loads((args.output / "train/schedule.json").read_text())
    if not completed["completed"] or completed["steps"] != 320 or actual != expected_schedule:
        raise ValueError("Fixed update budget or paired schedule changed")
    dev = load_rows(args.data / "dev.jsonl")
    dev = [r for r in dev if r["domain"] in ("search", "relevance", "substitute")] + [
        r for domain in ("math", "instruction")
        for r in [r for r in dev if r["domain"] == domain][:100]]
    if len(dev) != 1100 or len({r["sample_id"] for r in dev}) != 1100:
        raise ValueError("Full DEV coverage changed")
    checkpoint = args.output / "train/checkpoint-0320"
    runtime = Runtime(str(args.student), adapter_path=str(checkpoint))
    summary = runtime.evaluate(dev, args.output, "final-dev", batch_size=8)
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
    evaluated = {
        "precision": "bf16", "step": 320, "source": "fixed_final_update", "dev_rows": 1100,
        "adapter_sha256": file_digest(checkpoint / "adapter_model.safetensors"),
        "adapter_config_sha256": file_digest(checkpoint / "adapter_config.json"),
    }
    write_json(args.output / "evaluated-adapter.json", evaluated)
    write_json(args.output / "completed.json", {
        "completed": True, "scope": "development_only", "holdout_observed": False,
        "seed": args.seed, "variant": args.variant, "summary": summary,
        "extension_sha256": file_digest(args.extension),
        "quality_advantage_supported": False, "evaluated_adapter": evaluated,
    })
    if args.destination:
        publish(args.output, args.destination)


if __name__ == "__main__":
    main()
