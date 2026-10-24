#!/usr/bin/env bash
set -euo pipefail
: "${MODEL_PATH:?Set MODEL_PATH to the verified local model directory}"
exec vllm serve "$MODEL_PATH" --served-model-name servedistill --dtype bfloat16 \
  --tensor-parallel-size 1 --max-model-len 4096 --gpu-memory-utilization 0.8 \
  --host 127.0.0.1 --port 8000
