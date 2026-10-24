"""Joint query/document scoring reference using official Qwen3 reranker prompts."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from .data import canonical, grade, load_rows
from .runtime import Runtime, append_json, seed_all, summarize, write_json

PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on '
          'the Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
          '<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INSTRUCTION = ('Given a product search query, rank products that match the requested product '
               'and its constraints. Prefer exact matches to substitutes, complementary '
               'accessories, and irrelevant products.')

class PairReranker:
    def __init__(self, path, batch_size=8):
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = Runtime.load_model(path, "cuda:0")
        self.batch_size = batch_size
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.yes = self.tokenizer.encode("yes", add_special_tokens=False)
        self.no = self.tokenizer.encode("no", add_special_tokens=False)
        if len(self.yes) != 1 or len(self.no) != 1:
            raise ValueError("Reranker binary labels are not single token events")

    @torch.inference_mode()
    def score(self, query, products):
        values = []
        for start in range(0, len(products), self.batch_size):
            batch = products[start:start + self.batch_size]
            encoded = []
            for product in batch:
                facts = {key: value for key, value in product.items() if key != "id"}
                text = ("<Instruct>: " + INSTRUCTION + "\n<Query>: " + query
                        + "\n<Document>: " + canonical(facts))
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) + len(self.prefix) + len(self.suffix) > 4096:
                    raise ValueError("Reranker input exceeds the frozen context budget")
                encoded.append(self.prefix + tokens + self.suffix)
            inputs = self.tokenizer.pad({"input_ids": encoded}, padding=True, return_tensors="pt").to(self.model.device)
            logits = self.model(**inputs, logits_to_keep=1).logits[:, -1].float()
            binary = logits[:, [self.no[0], self.yes[0]]].softmax(-1)[:, 1]
            values.extend(binary.cpu().tolist())
        return values

def products(row):
    marker = "Products: "
    start = row["prompt"].index(marker) + len(marker)
    values, _ = json.JSONDecoder().raw_decode(row["prompt"][start:])
    if {p["id"] for p in values} != set(row["labels"]):
        raise ValueError("Displayed candidates do not match grading IDs")
    return values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    seed_all(20261024)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    runtime = PairReranker(args.model, args.batch_size)
    cold = time.monotonic() - started
    reports = {}
    for split in ("test", "ood"):
        path = args.output / (split + ".jsonl")
        if path.exists():
            raise FileExistsError("Reranker evaluation evidence already exists")
        rows = [r for r in load_rows(args.data / (split + ".jsonl")) if r["domain"] == "search"]
        runtime.score(rows[0]["query"], products(rows[0]))
        records, latencies = [], []
        for row in rows:
            candidates = products(row)
            torch.cuda.synchronize()
            started = time.monotonic()
            scores = runtime.score(row["query"], candidates)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], candidates[i]["id"]))
            text = canonical({"ranking": [candidates[i]["id"] for i in order]})
            record = {"sample_id": row["sample_id"], "input_sha256": row["input_sha256"],
                      "domain": "search", "locale": row["locale"], "text": text,
                      "output_tokens": 0, "truncated": False, "pair_scores": scores,
                      "latency_seconds": elapsed, **grade(row, text)}
            records.append(record)
            latencies.append(elapsed)
            append_json(path, record)
        reports[split] = {**summarize(records), "n_queries": len(rows),
                         "latency_seconds": {"p" + str(p): float(np.percentile(latencies, p)) for p in (50,95,99)},
                         "elapsed_seconds": sum(latencies),
                         "gpu_hours_per_million_queries": sum(latencies)/len(rows)*1e6/3600}
    result = {"model": "Qwen/Qwen3-Reranker-0.6B", "backend": "transformers_sdpa_bf16",
              "scope": "off-the-shelf joint pair scoring; search ranking only",
              "latency_boundary": "in-process scoring of all displayed candidates; excludes retrieval/HTTP",
              "candidate_batch_size": args.batch_size, "cold_load_seconds": cold,
              "device": torch.cuda.get_device_name(0), "splits": reports}
    write_json(args.output / "summary.json", result)
    print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
