"""Evaluate predeclared checkpoint rules on DEV without reading holdouts."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil

import torch

from servedistill.adapter_average import materialize_average
from servedistill.advantage_contracts import file_digest, verify_files
from servedistill.data import load_rows
from servedistill.runtime import Runtime, summarize, write_json


def publish(output: Path, destination: Path) -> None:
    for path in output.rglob("*"):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        target = destination / path.relative_to(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".upload")
        shutil.copyfile(path, temporary)
        temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "contract", "training", "student", "student-manifest", "dev",
        "output", "destination",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    if contract["scope"] != "development_only" or args.seed not in contract["seeds"]:
        raise ValueError("A declared DEV-only experiment is required")
    if args.dev.name != "dev.jsonl" or file_digest(args.dev) != contract["dev_sha256"]:
        raise ValueError("DEV data binding changed")
    for sibling in ("test.jsonl", "ood.jsonl"):
        if (args.dev.parent / sibling).exists():
            raise ValueError("Holdout data must not be packaged in the DEV study")
    seed_marker = json.loads((args.training / "training-seed-complete.json").read_text())
    if (
        not seed_marker["completed"]
        or seed_marker["holdout_observed"]
        or seed_marker["seed"] != args.seed
        or seed_marker["protocol_sha256"] != contract["source_protocol_sha256"]
    ):
        raise ValueError("Source training completion changed")
    verify_files(args.student_manifest, args.student, model_asset=True)
    rows = load_rows(args.dev)
    selected = [r for r in rows if r["domain"] in ("search", "relevance", "substitute")]
    selected += [
        r for domain in ("math", "instruction")
        for r in [x for x in rows if x["domain"] == domain][:100]
    ]
    if len(selected) != 1100:
        raise ValueError("BF16 DEV selection workload changed")
    args.output.mkdir(parents=True, exist_ok=True)
    adapters = {}
    aopd = args.training / f"aopd_seed{args.seed}"
    for label, spec in contract["aopd_rules"].items():
        if spec["kind"] == "fixed_checkpoint":
            adapter = aopd / "train" / f"checkpoint-{spec['step']:04d}"
        elif spec["kind"] == "lora_factor_mean_fp32":
            adapter = args.output / "adapters" / label
            materialize_average(aopd / "train", adapter, spec, 640)
        else:
            raise ValueError("Unknown declared DEV rule")
        adapters[label] = adapter
    for family in ("offline_sft", "oracle_sft"):
        arm = args.training / f"{family}_seed{args.seed}"
        complete = json.loads((arm / "training-completed.json").read_text())
        if not complete["completed"] or complete["holdout_observed"]:
            raise ValueError("SFT source training is incomplete")
        adapters[family] = arm / complete["evaluated_adapter_path"]
    outputs, records = {}, {}
    for label, adapter in adapters.items():
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            if not (adapter / name).is_file():
                raise ValueError("Missing persisted adapter: " + str(adapter))
        records[label] = {
            "adapter_sha256": file_digest(adapter / "adapter_model.safetensors"),
            "adapter_config_sha256": file_digest(adapter / "adapter_config.json"),
        }
        source_arm = label if label in ("offline_sft", "oracle_sft") else "aopd"
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            binding_key = str(
                (adapter / name).relative_to(args.training)
            ) if adapter.is_relative_to(args.training) else None
            binding = seed_marker["files"].get(binding_key)
            if binding and (
                file_digest(adapter / name) != binding["sha256"]
                or (adapter / name).stat().st_size != binding["bytes"]
            ):
                raise ValueError("Adapter binding changed")
        runtime = Runtime(str(args.student), adapter_path=str(adapter))
        part = selected if source_arm != "aopd" else [
            r for r in selected if r["domain"] == "substitute"
        ]
        runtime.evaluate(part, args.output, label)
        outputs[label] = load_rows(args.output / (label + ".jsonl"))
        del runtime
        gc.collect()
        torch.cuda.empty_cache()
        publish(args.output, args.destination)
    summaries = {}
    baseline = outputs["offline_sft"]
    for label in contract["aopd_rules"]:
        substitute = {r["sample_id"]: r for r in outputs[label]}
        routed = [
            substitute[r["sample_id"]] if r["domain"] == "substitute" else r
            for r in baseline
        ]
        summaries[label] = summarize(routed)
    write_json(args.output / "completed.json", {
        "completed": True,
        "seed": args.seed,
        "contract_sha256": file_digest(args.contract),
        "scope": "development_only",
        "holdout_observed": False,
        "precision": "bf16",
        "source_adapters": records,
        "routed_summaries": summaries,
        "boundary": "These DEV outputs cannot establish a new holdout advantage.",
    })
    publish(args.output, args.destination)


if __name__ == "__main__":
    main()
