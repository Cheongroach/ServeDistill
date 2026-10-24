"""Streaming HTTP latency and effective throughput with bounded clients."""
from __future__ import annotations
import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
import aiohttp
import numpy as np
from .data import SYSTEM, TEACHER_SYSTEM, grade, load_rows

async def stream_request(session, endpoint, model, row, max_tokens):
    request_id = str(uuid.uuid4())
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": row.get("system", SYSTEM)},
                     {"role": "user", "content": row["prompt"]}],
        "temperature": 0.0, "max_tokens": max_tokens, "seed": 20261024,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True, "stream_options": {"include_usage": True},
    }
    started = time.monotonic()
    first_token = None
    fragments, response_ids, finish_reasons = [], set(), set()
    usage, done = None, False
    async with session.post(endpoint.rstrip("/") + "/v1/chat/completions", json=payload,
                            headers={"X-Request-Id": request_id}) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:500]}")
        async for raw in response.content:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if event.get("id"):
                response_ids.add(event["id"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                if choice.get("index") != 0:
                    raise RuntimeError("Unexpected multi-completion response")
                delta = choice.get("delta", {})
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    raise RuntimeError("Thinking was not disabled by the endpoint")
                content = delta.get("content")
                if content:
                    if first_token is None:
                        first_token = time.monotonic()
                    fragments.append(content)
                if choice.get("finish_reason"):
                    finish_reasons.add(choice["finish_reason"])
    ended = time.monotonic()
    if (not done or len(response_ids) != 1 or not usage or first_token is None
            or len(finish_reasons) != 1):
        raise RuntimeError("Incomplete stream, response ID, usage, or first content timestamp")
    text = "".join(fragments)
    record = {
        "sample_id": row["sample_id"], "input_sha256": row["input_sha256"],
        "domain": row["domain"], "locale": row.get("locale", "general"),
        "client_request_id": request_id, "server_response_id": next(iter(response_ids)),
        "latency_seconds": ended - started, "ttft_seconds": first_token - started,
        "decode_seconds": ended - first_token, "output_tokens": int(usage["completion_tokens"]),
        "prompt_tokens": int(usage["prompt_tokens"]), "finish_reasons": sorted(finish_reasons),
        "truncated": "length" in finish_reasons, "text": text, **grade(row, text),
    }
    return record

def quantiles(values):
    return {f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)}

async def sweep(session, args, rows, concurrency, output):
    queue = asyncio.Queue()
    for row in rows:
        queue.put_nowait(row)
    records = []
    started = time.monotonic()
    async def worker():
        while not queue.empty():
            row = queue.get_nowait()
            record = await stream_request(session, args.endpoint, args.model, row, args.max_new_tokens)
            record["concurrency"] = concurrency
            records.append(record)
            with (output / "requests.jsonl").open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    await asyncio.wait_for(asyncio.gather(*(worker() for _ in range(concurrency))),
                           timeout=args.sweep_timeout)
    elapsed = time.monotonic() - started
    if len({r["server_response_id"] for r in records}) != len(records):
        raise RuntimeError("Server reused response IDs across independent requests")
    n, valid = len(records), sum(r["success"] for r in records)
    return {
        "n": n, "elapsed_seconds": elapsed,
        "latency_seconds": quantiles([r["latency_seconds"] for r in records]),
        "ttft_seconds": quantiles([r["ttft_seconds"] for r in records]),
        "output_tokens": sum(r["output_tokens"] for r in records),
        "output_tokens_per_second": sum(r["output_tokens"] for r in records) / elapsed,
        "requests_per_second": n / elapsed, "effective_answers_per_second": valid / elapsed,
        "success_fraction": valid / n, "json_valid_fraction": sum(r["json_valid"] for r in records) / n,
        "truncation_fraction": sum(r["truncated"] for r in records) / n,
        "gpu_hours_per_million_requests": args.gpus * elapsed / n * 1e6 / 3600,
        "gpu_hours_per_million_effective_answers": args.gpus * elapsed / valid * 1e6 / 3600 if valid else None,
    }

async def benchmark(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "requests.jsonl").exists():
        raise FileExistsError("Refusing to overwrite HTTP evidence")
    counts, rows = {}, []
    for row in load_rows(args.data):
        domain = row["domain"]
        if counts.get(domain, 0) < args.per_domain:
            if args.teacher_prompts:
                row = {**row, "system": row.get("teacher_system", TEACHER_SYSTEM)}
            rows.append(row)
            counts[domain] = counts.get(domain, 0) + 1
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.request_timeout),
                                    trust_env=False) as session:
        async with session.get(args.endpoint.rstrip("/") + "/v1/models") as response:
            response.raise_for_status()
            models = await response.json()
        if args.model not in {m["id"] for m in models["data"]}:
            raise RuntimeError("Served model identity differs from the benchmark")
        for row in rows[:5]:
            await stream_request(session, args.endpoint, args.model, row, args.max_new_tokens)
        summaries = {}
        for concurrency in (1, 4, 16):
            summaries[str(concurrency)] = await sweep(session, args, rows, concurrency, output)
            print(json.dumps({"event": "http_sweep", "concurrency": concurrency,
                              **summaries[str(concurrency)]}), flush=True)
    report = {
        "backend": "vllm_streaming_http", "concurrency": summaries,
        "boundary": "localhost HTTP dispatch through stream DONE; includes server queue, excludes retrieval",
        "load_model": "continuously replenished closed clients; excludes intentional client-side waiting",
        "ttft": "first nonempty content delta; role and usage chunks excluded",
        "precision": "BF16", "gpu_count": args.gpus, "max_new_tokens": args.max_new_tokens,
        "per_domain": counts, "enable_thinking": False, "warmup_requests": min(5, len(rows)),
        "price": "multiply GPU hours by an explicit currency/GPU-hour price",
        "identity_checks": "one server response ID per request; exact served model name",
        "prompt_role": "teacher" if args.teacher_prompts else "student",
    }
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="servedistill")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-domain", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--sweep-timeout", type=int, default=900)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--teacher-prompts", action="store_true")
    args = parser.parse_args()
    if min(args.per_domain, args.max_new_tokens, args.gpus) < 1:
        parser.error("Sizes and GPU count must be positive")
    asyncio.run(benchmark(args))

if __name__ == "__main__":
    main()
