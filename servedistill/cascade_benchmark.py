"""Measured student/teacher cascade with label-free protocol fallback."""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp

from .data import TEACHER_SYSTEM, load_rows, parse_response
from .http_benchmark import quantiles, stream_request


def protocol_valid(row, text):
    """Read the output contract and candidate IDs, never relevance-label values."""
    value = parse_response(text)
    task = row.get("task")
    if task == "esci_ranking":
        if value is None or set(value) != {"ranking"}:
            return False
        ranking = value["ranking"]
        ids = set(row["labels"].keys())
        return (isinstance(ranking, list)
                and all(isinstance(item, str) for item in ranking)
                and len(ranking) == len(ids) and set(ranking) == ids)
    if task == "esci_classification":
        return value is not None and set(value) == {"label"} and value["label"] in ("E", "S", "C", "I")
    if task == "esci_substitute":
        return (value is not None and set(value) == {"substitute"}
                and value["substitute"] in ("yes", "no"))
    return value is not None and set(value) == {"answer"} and isinstance(value["answer"], (str, int, float))


async def request(session, args, row):
    started = time.monotonic()
    student = await stream_request(session, args.student_endpoint, args.student_model,
                                   row, args.max_new_tokens)
    fallback = student["truncated"] or not protocol_valid(row, student["text"])
    final = student
    teacher = None
    if fallback:
        teacher_row = {**row, "system": row.get("teacher_system", TEACHER_SYSTEM)}
        teacher = await stream_request(session, args.teacher_endpoint, args.teacher_model,
                                       teacher_row, args.max_new_tokens)
        final = teacher
    elapsed = time.monotonic() - started
    return {
        **final, "latency_seconds": elapsed, "fallback": fallback,
        "student_latency_seconds": student["latency_seconds"],
        "teacher_latency_seconds": teacher["latency_seconds"] if teacher else 0.0,
        "student_response_id": student["server_response_id"],
        "teacher_response_id": teacher["server_response_id"] if teacher else None,
        "final_protocol_valid": not final["truncated"] and protocol_valid(row, final["text"]),
        # A complete JSON object is validated before any answer becomes visible.
        "first_visible_answer_seconds": elapsed,
    }


async def benchmark(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "requests.jsonl").exists():
        raise FileExistsError("Refusing to overwrite cascade evidence")
    rows, counts = [], {}
    for row in load_rows(args.data):
        domain = row["domain"]
        if counts.get(domain, 0) < args.per_domain:
            rows.append(row)
            counts[domain] = counts.get(domain, 0) + 1
    reports = {}
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=args.request_timeout), trust_env=False) as session:
        for endpoint, model in ((args.student_endpoint, args.student_model),
                                (args.teacher_endpoint, args.teacher_model)):
            async with session.get(endpoint.rstrip("/") + "/v1/models") as response:
                response.raise_for_status()
                available = await response.json()
            if model not in {m["id"] for m in available["data"]}:
                raise RuntimeError("Cascade model identity mismatch")
        for row in rows[:5]:
            await request(session, args, row)
        for concurrency in (1, 4, 16):
            queue, records = asyncio.Queue(), []
            for row in rows:
                queue.put_nowait(row)
            async def worker():
                while not queue.empty():
                    row = queue.get_nowait()
                    record = {**await request(session, args, row), "concurrency": concurrency}
                    records.append(record)
                    with (output / "requests.jsonl").open("a") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            started = time.monotonic()
            await asyncio.wait_for(asyncio.gather(*(worker() for _ in range(concurrency))),
                                   args.sweep_timeout)
            elapsed = time.monotonic() - started
            identifiers = [r[key] for r in records
                           for key in ("student_response_id", "teacher_response_id") if r[key]]
            if len(identifiers) != len(set(identifiers)):
                raise RuntimeError("Cascade endpoint reused response IDs")
            n, valid = len(records), sum(r["success"] for r in records)
            reports[str(concurrency)] = {
                "n": n, "elapsed_seconds": elapsed,
                "latency_seconds": quantiles([r["latency_seconds"] for r in records]),
                "first_visible_answer_seconds": quantiles([r["first_visible_answer_seconds"] for r in records]),
                "requests_per_second": n / elapsed,
                "effective_answers_per_second": valid / elapsed,
                "success_fraction": valid / n,
                "fallback_fraction": sum(r["fallback"] for r in records) / n,
                "protocol_valid_fraction": sum(r["final_protocol_valid"] for r in records) / n,
                "truncation_fraction": sum(r["truncated"] for r in records) / n,
                "gpu_hours_per_million_requests": 2 * elapsed / n * 1e6 / 3600,
                "gpu_hours_per_million_effective_answers": 2 * elapsed / valid * 1e6 / 3600 if valid else None,
            }
            print(json.dumps({"event": "cascade_sweep", "concurrency": concurrency,
                              **reports[str(concurrency)]}), flush=True)
    report = {
        "backend": "vllm_streaming_http_cascade", "concurrency": reports,
        "routing": "malformed output contract or truncation; no correctness labels or confidence oracle",
        "answer_delivery": "complete JSON buffered until validation; first visible answer equals completion",
        "load_model": "continuously replenished closed clients",
        "gpu_count": 2, "allocation": "one dedicated student GPU and one dedicated teacher GPU",
        "boundary": "localhost sequential HTTP cascade; retrieval excluded",
        "price": "measured allocated GPU hours multiplied by an explicit currency/GPU-hour price",
        "per_domain": counts, "max_new_tokens": args.max_new_tokens, "warmup_requests": min(5, len(rows)),
    }
    (output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-endpoint", required=True)
    parser.add_argument("--teacher-endpoint", required=True)
    parser.add_argument("--student-model", default="servedistill")
    parser.add_argument("--teacher-model", default="servedistill")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-domain", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--sweep-timeout", type=int, default=900)
    args = parser.parse_args()
    if min(args.per_domain, args.max_new_tokens) < 1:
        parser.error("Sizes must be positive")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
