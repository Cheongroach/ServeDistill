"""Bounded development screens for teacher repair and class-aware supervision."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from servedistill.data import canonical, load_rows
from servedistill.runtime import Runtime, write_json
from scripts.analyze_expanded import classification_metrics
from scripts.worker import publish


def evaluate(runtime, rows, output, label, destination):
    report = runtime.evaluate(rows, output, label, batch_size=8)
    evidence = load_rows(output / f"{label}.jsonl")
    write_json(output / f"{label}.classes.json", {
        domain: classification_metrics([row for row in evidence if row["domain"] == domain])
        for domain in ("relevance", "substitute")})
    publish(output, destination)
    return report


def prompt_screen(args, rows):
    examples = json.loads((args.data / "training-examples.json").read_text())
    results = {}
    for label, model in (("student", args.student), ("teacher", args.teacher)):
        runtime = Runtime(model)
        for variant in ("original", "training_examples"):
            edited = []
            for original in rows:
                row = dict(original)
                system = row.get("teacher_system", row.get("system", "")) if label == "teacher" else row.get("system", "")
                if variant == "training_examples" and row["domain"] in ("search", "relevance", "substitute"):
                    demonstrations = "\n".join(
                        "Example input: " + example["prompt"] + "\nExample output: "
                        + canonical(example["target"]) for example in examples[row["domain"]])
                    system += "\nLabeled training examples; apply the same definitions to the new query:\n" + demonstrations
                row["system"] = system
                edited.append(row)
            name = label + "-" + variant
            write_json(args.output / (name + ".prompt-provenance.json"), {
                "systems_sha256": hashlib.sha256(
                    canonical([row["system"] for row in edited]).encode()).hexdigest(),
                "example_source": "training-examples.json, train split only",
                "boundary": "Development prompt screen; input hash covers original user prompt",
            })
            results[name] = evaluate(runtime, edited, args.output, name, args.destination)
        del runtime
        gc.collect()
        torch.cuda.empty_cache()
    return results


def train_screen(args, rows):
    teacher = args.mode == "teacher_sft"
    model = args.teacher if teacher else args.student
    config = {
        "objective": "oracle_sft", "train_precision": "bf16" if teacher else "float32_torch",
        "seed": 17, "steps": 320, "prompts_per_step": 10, "group_size": 1,
        "microbatch": 4, "rollout_batch": 4, "max_new_tokens": 128,
        "learning_rate": args.learning_rate or (1e-4 if teacher else 5e-5), "dev_every": 80,
        "dev_per_domain": 40, "aggregation": "token", "behavior_tolerance": 0.15,
    }
    path = args.output / "config.json"
    write_json(path, config)
    argv = [
        sys.executable, "-m", "servedistill.train", "--student", str(model),
        "--data", str(args.data), "--config", str(path), "--output", str(args.output / "train")]
    started = time.time()
    with (args.output / "train.stdout.log").open("w") as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT)
        while True:
            try:
                returncode = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                publish(args.output, args.destination)
    write_json(args.output / "train-execution.json", {
        "returncode": returncode, "start_unix_seconds": started, "end_unix_seconds": time.time()})
    publish(args.output, args.destination)
    if returncode:
        raise RuntimeError("Supervised development screen failed")
    selected = json.loads((args.output / "train/selected.json").read_text())
    runtime = Runtime(model, adapter_path=str(args.output / "train" / selected["path"]))
    report = evaluate(runtime, rows, args.output, "selected-dev", args.destination)
    write_json(args.output / "selected.json", selected)
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
    if teacher:
        export = args.output / "teacher-export"
        with (args.output / "export.stdout.log").open("w") as stream:
            result = subprocess.run([
                sys.executable, "-m", "servedistill.export", "--model", str(model),
                "--adapter", str(args.output / "train" / selected["path"]),
                "--selection", str(args.output / "selected.json"), "--output", str(export)],
                stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("Repaired teacher export failed")
        publish(args.output, args.destination)
    return {"selected": selected, "development": report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prompt", "teacher_sft", "student_sft"), required=True)
    parser.add_argument("--learning-rate", type=float)
    for name in ("student", "teacher", "data", "output", "destination"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = load_rows(args.data / "dev.jsonl")
    # Whole real business dev slices; finite general checks remain explicitly limited.
    rows = [row for row in rows if row["domain"] in ("search", "relevance", "substitute")] + [
        row for domain in ("math", "instruction")
        for row in [r for r in rows if r["domain"] == domain][:100]]
    result = prompt_screen(args, rows) if args.mode == "prompt" else train_screen(args, rows)
    write_json(args.output / "screen-completed.json", {
        "completed": True, "mode": args.mode, "result": result,
        "boundary": "Development selection only; no confirmatory quality or cost advantage established"})
    publish(args.output, args.destination)


if __name__ == "__main__":
    main()
