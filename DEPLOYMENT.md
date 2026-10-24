# Deployment reference

Merge the dev-selected adapter into a BF16 language checkpoint and run the same frozen final quality evaluation before serving. Training uses an FP32 student for the measured probability contract; this is independent of deployment precision.

The in-process Hugging Face benchmark is a reproducible reference, not an online SLA. It reports static batches at 1/4/16 requests, per-batch completion latency, first-token callback timing, valid output fraction and GPU hours per million requests. Include retrieval, HTTP, queueing and fallback costs when validating a real service.

`scripts/run_reference_server.sh` starts a localhost vLLM endpoint from `MODEL_PATH`. Use a vLLM version with native Qwen3.5 support and its supported PyTorch/Transformers stack; do not install it over the training environment. Send `chat_template_kwargs={"enable_thinking":false}` and enforce the same output budget as evaluation. The serving stack is being validated separately; this script is not evidence that an HTTP service has passed.

No GPU price is assumed. Multiply measured GPU hours by an explicit unit price; preserve teacher labeling, teacher scoring, failed jobs, loading and idle residency in training cost. Report effective answers separately from requests.

## Streaming HTTP measurement

Install the serving stack in a separate Python environment (`vllm==0.19.1`, `torch==2.10.0`, `transformers==5.6.2`, `numpy==2.2.6`). The resolved binary wheelhouse is pinned by SHA-256. `scripts/run_reference_server.sh` enables native Qwen3.5 language-only serving.

```bash
# In the training environment, export the checkpoint selected on dev.
python -m servedistill.export --model /tmp/student/model \
  --adapter results/run/train/checkpoint-0040 \
  --selection results/run/train/selected.json --output /tmp/merged-student
# In the serving environment.
MODEL_PATH=/tmp/merged-student bash scripts/run_reference_server.sh
python -m servedistill.http_benchmark --data data_search/test.jsonl \
  --output results/http --per-domain 30
```

The client measures the first nonempty content delta and the complete stream through `[DONE]`; it rejects mixed response IDs, missing usage, and interrupted streams. Concurrency 1/4/16 uses continuously replenished clients. This is a localhost HTTP benchmark, including engine queues and protocol handling, with retrieval excluded. GPU hours per million correct answers expose quality-related cost; total requests and correct answers are distinct denominators. Teacher fallback measurements use `--teacher-prompts` to preserve the qualified teacher prompt.

HTTP measurements are in progress. Source support and passing CPU protocol tests do not establish GPU engine startup or latency.

## Reusing the business checkpoint for general restoration

```bash
python -m servedistill.pipeline --student /tmp/student/model --teacher /tmp/teacher/model \
  --data data_search --config configs/matrix/restore_general_opd_seed17.json \
  --output results/restore_general_opd_seed17 \
  --initial-adapter results/business_sft_seed17/train/checkpoint-0040 \
  --initial-selection results/business_sft_seed17/train/selected.json
```

Replace the example checkpoint with the actual dev-selected directory. The pipeline saves the selection and adapter file hash before training; it starts a new optimizer for the second stage. Cold resume of the same stage instead restores Adam, RNG and controller state with `--resume`. These two operations have different experimental meanings.
