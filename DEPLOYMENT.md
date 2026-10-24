# Deployment reference

Merge the dev-selected adapter into a BF16 language checkpoint and run the same frozen final quality evaluation before serving. Training uses an FP32 student for the measured probability contract; this is independent of deployment precision.

The in-process Hugging Face benchmark is a reproducible reference, not an online SLA. It reports static batches at 1/4/16 requests, per-batch completion latency, first-token callback timing, valid output fraction and GPU hours per million requests. Include retrieval, HTTP, queueing and fallback costs when validating a real service.

`scripts/run_reference_server.sh` starts a localhost vLLM endpoint from `MODEL_PATH`. Use a vLLM version with native Qwen3.5 support and its supported PyTorch/Transformers stack; do not install it over the training environment. Send `chat_template_kwargs={"enable_thinking":false}` and enforce the same output budget as evaluation. The serving stack is being validated separately; this script is not evidence that an HTTP service has passed.

No GPU price is assumed. Multiply measured GPU hours by an explicit unit price; preserve teacher labeling, teacher scoring, failed jobs, loading and idle residency in training cost. Report effective answers separately from requests.
