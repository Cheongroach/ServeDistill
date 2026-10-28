"""Train one prospectively frozen domain CE candidate on observed DEV inputs."""
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
from servedistill.scheduling import Config, schedule, update_learning_rate


def load_protocol(base_path, protocol_path):
    base = json.loads(base_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    required = {
        "domain01": {"search": .1, "relevance": .1, "substitute": .1,
                     "math": 1.0, "instruction": 1.0},
        "domain05": {"search": .5, "relevance": .5, "substitute": .5,
                     "math": 1.0, "instruction": 1.0},
    }
    if (protocol["base_protocol_sha256"] != file_digest(base_path)
            or base["scope"] != "development_only" or base["holdout_files_packaged"]
            or base["holdout_observed_in_this_round"]
            or protocol["scope"] != "development_only" or protocol["holdout_files_packaged"]
            or protocol["holdout_observed"] or protocol["seeds"] != base["seeds"]
            or len(set(protocol["seeds"])) != 4 or protocol["variants"] != required
            or protocol["quality_gates"] != base["quality_gates"]
            or protocol["selection_checkpoint"] != "fixed_final_update_320"
            or protocol["selection_precision"] != "bf16"):
        raise ValueError("Frozen two-candidate domain DEV binding changed")
    return base, protocol


def variant_config(base, protocol, variant, seed):
    if seed not in protocol["seeds"] or variant not in protocol["variants"]:
        raise ValueError("Unknown paired domain candidate or seed")
    config = Config(**{**base["methods"]["aopd_anchor"], "seed": seed,
                       "anchor_ce_weight": 0.0,
                       "anchor_ce_domain_weights": protocol["variants"][variant]})
    update_learning_rate(config, 1)
    return config


def main():
    parser = argparse.ArgumentParser()
    for key in ("student", "teacher", "data", "initial", "base-protocol", "protocol", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    base, protocol = load_protocol(args.base_protocol, args.protocol)
    repo = Path(__file__).resolve().parents[1]
    # Updated execution sources have their own manifest; old hashes remain historical.
    for relative, expected in protocol["source_files"].items():
        if file_digest(repo / relative) != expected:
            raise ValueError("Domain execution source changed: " + relative)
    for key, path in {
        "train_sha256": args.data / "train.jsonl", "dev_sha256": args.data / "dev.jsonl",
        "initial_adapter_sha256": args.initial / "adapter_model.safetensors",
        "initial_config_sha256": args.initial / "adapter_config.json",
    }.items():
        if file_digest(path) != base[key]:
            raise ValueError("Common DEV input changed: " + key)
    config = variant_config(base, protocol, args.variant, args.seed)
    expected_schedule = [[r["sample_id"] for r in batch]
                         for batch in schedule(load_rows(args.data / "train.jsonl"), config)]
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", dataclasses.asdict(config))
    write_json(args.output / "binding.json", {
        "scope": "development_only", "holdout_observed": False,
        "seed": args.seed, "variant": args.variant, "fresh_adam": True,
        "protocol_sha256": file_digest(args.protocol),
        "base_protocol_sha256": file_digest(args.base_protocol),
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
        raise RuntimeError("Domain anchor training failed")
    completed = json.loads((args.output / "train/completed.json").read_text())
    if (not completed["completed"] or completed["steps"] != config.steps
            or json.loads((args.output / "train/schedule.json").read_text()) != expected_schedule):
        raise ValueError("Fixed update budget or paired prompt schedule changed")
    dev = load_rows(args.data / "dev.jsonl")
    dev = [r for r in dev if r["domain"] in config.business_domains] + [
        r for domain in ("math", "instruction") for r in
        [r for r in dev if r["domain"] == domain][:100]]
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
        "protocol_sha256": file_digest(args.protocol), "quality_advantage_supported": False,
        "evaluated_adapter": evaluated,
    })
    if args.destination:
        publish(args.output, args.destination)


if __name__ == "__main__":
    main()
