"""Comparable warm serving measurements; backend and concurrency are explicit."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import grade, load_rows
from .runtime import Runtime, append_json, seed_all, stratified, write_json


def quantiles(values: list[float]) -> dict:
    return {f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)}


def benchmark_hf(args, rows, output):
    from transformers import GenerationConfig
    from transformers.generation.streamers import BaseStreamer
    runtime = Runtime(args.model, adapter_path=args.adapter)
    write_json(output / "contract.json", runtime.contract)
    warm = rows[:min(4, len(rows))]
    runtime.rollout(warm, args.max_new_tokens, sample=False, audit=False)
    records, summaries = [], {}

    class Timer(BaseStreamer):
        def __init__(self):
            self.prompt = True
            self.first_token = None
            self.last_token = None

        def put(self, value):
            if self.prompt:
                self.prompt = False
                return
            timestamp = time.monotonic()
            if self.first_token is None:
                self.first_token = timestamp
            self.last_token = timestamp

        def end(self):
            pass

    for concurrency in (1, 4, 16):
        latencies, ttfts, elapsed_total, token_total = [], [], 0.0, 0
        successes, request_total = 0, 0
        for index in range(0, len(rows), concurrency):
            batch = rows[index:index + concurrency]
            inputs = runtime.prompts(batch)
            timer = Timer()
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.inference_mode():
                result = runtime.student.generate(
                    **inputs, generation_config=GenerationConfig(
                        do_sample=False, max_new_tokens=args.max_new_tokens,
                        eos_token_id=runtime.eos_ids,
                        pad_token_id=runtime.tokenizer.pad_token_id, use_cache=True),
                    streamer=timer,
                )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            texts = runtime.tokenizer.batch_decode(
                result[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            elapsed_total += elapsed
            for row, text in zip(batch, texts):
                tokens = len(runtime.tokenizer.encode(text, add_special_tokens=False))
                token_total += tokens
                quality = grade(row, text)
                successes += int(quality["success"])
                request_total += 1
                # HF synchronous static batch reports batch completion latency for every row.
                latency = elapsed
                ttft = timer.first_token - started if timer.first_token is not None else None
                latencies.append(latency)
                if ttft is not None:
                    ttfts.append(ttft)
                record = {
                    "sample_id": row["sample_id"], "concurrency": concurrency,
                    "latency_seconds": latency, "ttft_seconds": ttft,
                    "output_tokens": tokens, "text": text, **quality,
                }
                records.append(record)
                append_json(output / "requests.jsonl", record)
        summaries[str(concurrency)] = {
            "n": request_total, "latency_seconds": quantiles(latencies),
            "ttft_seconds": quantiles(ttfts), "elapsed_seconds": elapsed_total,
            "requests_per_second": request_total / elapsed_total,
            "output_tokens_per_second": token_total / elapsed_total,
            "gpu_seconds_per_request_at_saturation": elapsed_total / request_total,
            "gpu_hours_per_million_requests": elapsed_total / request_total * 1e6 / 3600,
            "success_fraction": successes / request_total,
            "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
        }
    return {
        "backend": "transformers_sdpa_bf16_static_batch",
        "latency_boundary": "in-process generation; excludes HTTP, queue and retrieval",
        "ttft_boundary": "first streamed token callback; static batch timing",
        "cold_load_seconds": runtime.load_seconds, "concurrency": summaries,
    }


def benchmark_vllm(args, rows, output):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from .data import SYSTEM
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    started = time.monotonic()
    engine = LLM(
        model=args.model, dtype="bfloat16", tensor_parallel_size=1,
        max_model_len=4096, gpu_memory_utilization=0.8, trust_remote_code=False,
        enable_lora=bool(args.adapter), max_lora_rank=16,
    )
    cold = time.monotonic() - started
    adapter = LoRARequest("servedistill", 1, args.adapter) if args.adapter else None
    params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    def prompts(batch):
        return [tokenizer.apply_chat_template([
            {"role": "system", "content": row.get("system", SYSTEM)},
            {"role": "user", "content": row["prompt"]},
        ], tokenize=False, add_generation_prompt=True,
            enable_thinking=False) for row in batch]
    engine.generate(prompts(rows[:4]), params, lora_request=adapter, use_tqdm=False)
    summaries = {}
    for concurrency in (1, 4, 16):
        latencies, ttfts, request_count, token_count, elapsed_total, successes = [], [], 0, 0, 0.0, 0
        for index in range(0, len(rows), concurrency):
            batch = rows[index:index + concurrency]
            start = time.monotonic()
            responses = engine.generate(prompts(batch), params, lora_request=adapter, use_tqdm=False)
            elapsed_total += time.monotonic() - start
            for row, response in zip(batch, responses):
                completion = response.outputs[0]
                metric = response.metrics
                if metric is None or metric.finished_time is None or metric.first_token_time is None:
                    raise RuntimeError("vLLM version did not provide auditable per-request timestamps")
                latency = metric.finished_time - metric.arrival_time
                ttft = metric.first_token_time - metric.arrival_time
                latencies.append(latency)
                ttfts.append(ttft)
                tokens = len(completion.token_ids)
                token_count += tokens
                request_count += 1
                quality = grade(row, completion.text)
                successes += int(quality["success"])
                append_json(output / "requests.jsonl", {
                    "sample_id": row["sample_id"], "concurrency": concurrency,
                    "latency_seconds": latency, "ttft_seconds": ttft,
                    "output_tokens": tokens, "text": completion.text, **quality,
                })
        summaries[str(concurrency)] = {
            "n": request_count, "latency_seconds": quantiles(latencies),
            "ttft_seconds": quantiles(ttfts), "elapsed_seconds": elapsed_total,
            "requests_per_second": request_count / elapsed_total,
            "output_tokens_per_second": token_count / elapsed_total,
            "gpu_seconds_per_request_at_saturation": elapsed_total / request_count,
            "gpu_hours_per_million_requests": elapsed_total / request_count * 1e6 / 3600,
            "success_fraction": successes / request_count,
        }
    return {
        "backend": "vllm_bf16", "cold_engine_seconds": cold,
        "latency_boundary": "in-process scheduler/generation; excludes HTTP and retrieval",
        "concurrency": summaries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--per-domain", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    seed_all(20261024)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "requests.jsonl").exists():
        raise FileExistsError("Benchmark evidence already exists")
    rows = stratified(load_rows(args.data), args.per_domain)
    result = (benchmark_hf if args.backend == "hf" else benchmark_vllm)(args, rows, output)
    result.update({
        "device": torch.cuda.get_device_name(0),
        "precision": "BF16", "max_new_tokens": args.max_new_tokens,
        "price": "c currency units per GPU-hour; no assumed unit price",
    })
    write_json(output / "benchmark.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
