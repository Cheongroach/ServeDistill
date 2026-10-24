# ServeDistill

**On-policy distillation for business decisions, capability retention, and small-model serving.**

ServeDistill tests whether OPD is worth using for product-search relevance, reranking, capability retention, and small-model serving. The main benchmark uses real Amazon ESCI queries, product texts, and independently annotated labels. Fictional local-services tasks provide additional checks of budget, refunds, inventory, and clarification decisions.

Current status: the 26-run, two-seed training matrix is running on H20; two general-restoration stages and native vLLM HTTP baselines are also being executed. Four nonzero FP32 pilot updates, frozen teacher checks, and matching adapter/Adam/RNG cold-restoration fingerprints passed. Maximum cached/recomputed log-probability error was 4.55e-5; the earlier BF16 gate rejected a batch at 0.194. **No final training improvement or serving speedup has been established yet**.

The primary public models are **Qwen3.5-9B → Qwen3.5-2B**, with Qwen3.5-4B as a serving/quality reference. These are public post-trained checkpoints, used through their text-only language components. Prompt-routed teachers are not presented as independently trained domain experts. Thinking is explicitly disabled for both teacher and student.

- Main tasks: search reranking, E/S/C/I relevance classification, substitute identification.
- Extensions: O2O coupon redemption, recommendation, and ads prediction; see the dataset selection notes for readiness and teacher requirements.
- Retention: a public GSM8K subset and deterministic instruction tasks, with a deliberately limited general-capability claim.
- Evidence: task-level paired confidence intervals, per-domain failures, effective training tokens, complete GPU hours, and measured inference latency.
- Reproducibility: model/file hashes, tokenizer contracts, rollout versions, deterministic grading, and bounded execution.
- Data: public ESCI search data plus clearly labeled fictional local-services constraints.

See the [experiment protocol](EXPERIMENT_PROTOCOL.md) for objectives, loss definitions, ablations, qualification gates, and acceptance criteria.
See [dataset selection](DATASET_SELECTION.md) for the search, coupon, recommendation, and ads benchmark choices.

## Branches

`main` contains runnable code, tests, experiment configurations, and reproducible reports. `manuscript` contains research notes, design decisions, and a sanitized conversation/design history. Resource identifiers, credentials, internal endpoints, and unrelated workspace files are excluded from public history.

## Interpretation

OPD uses student-generated trajectories and teacher feedback at the prefixes the student visits. Dense token supervision and optimizer update frequency are separate choices. A successful simulation is evidence for a business pilot, not a claim about real traffic, conversion, or revenue.

No fixed GPU price is assumed. Monetary estimates use measured GPU hours and an explicit user-supplied unit price.

## Run

Use a CUDA environment with a compatible PyTorch build. Keep the accelerator packages supplied by that environment; install the Python dependencies from `pyproject.toml`. Qwen3.5 requires the newer Transformers implementation.

```bash
pip install -e '.[data]'
python -m servedistill.assets --model Qwen/Qwen3.5-2B --output /tmp/student
python -m servedistill.assets --model Qwen/Qwen3.5-9B --output /tmp/teacher
python -m servedistill.data --output data
python -m servedistill.search_data --cache /tmp/esci --output data_search --general data
python -m servedistill.evaluate --student /tmp/student/model --teacher /tmp/teacher/model \
  --data data_search --output results/qualification --phase qualification
python -m servedistill.train --student /tmp/student/model --teacher /tmp/teacher/model \
  --data data_search --config configs/pilot.json --output results/pilot
```

Every checkpoint includes the LoRA adapter, Adam state, RNG state, controller state, immutable configuration, and tensor fingerprints. Use `--stop-after N` and then `--resume results/pilot/checkpoint-NNNN` in a fresh process to audit restoration.

Training uses an FP32 student with the PyTorch DeltaNet path and a frozen BF16 teacher after the BF16 probability gate failed. Serving is measured separately in BF16. Install `.[kernels]` for the BF16 inference path; causal-conv1d is optional and its presence is recorded.
