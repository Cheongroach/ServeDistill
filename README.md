# ServeDistill

**On-policy distillation for business decisions, capability retention, and small-model serving.**

ServeDistill tests whether OPD is worth using for product-search relevance, reranking, capability retention, and small-model serving. The main benchmark uses real Amazon ESCI queries, product texts, and independently annotated labels. Fictional local-services tasks provide additional checks of budget, refunds, inventory, and clarification decisions.

Start with the Chinese [54-step tutorial](docs/一步步认识ServeDistill项目-friendly.md).
The standalone [performance monitoring manual](PERFORMANCE_MONITORING.md) covers runnable
Nsight Systems, Nsight Compute, PyTorch profiler, GPU/host telemetry, DCGM, perf,
Compute Sanitizer, serving measurements, and validity checks.

As of 2026-10-27, **stable quality superiority remains unestablished**. The frozen
router confirmation failed; its test and replication are now observed data.
The four-seed DEV stability study selected last2 averaging, but only two of four
pairs improved over offline SFT. The [preserved results](reports/advantage/ROUTER_STABILITY_DEV.md)
include failures and allocation costs.

A new [DEV-only protocol](configs/anchor-development-v1.json) compares AOPD with a
fixed gold CE anchor, pure AOPD, oracle SFT, and offline SFT. All methods use one
common adapter, fresh Adam, paired prompt schedules, and the same Teacher320
for on-policy feedback and unfiltered offline labels. Four one-GPU jobs are running;
final DEV comparisons remain pending. This round respects **at most 8 concurrent
H20 GPUs**. Submission and CPU tests are setup evidence; completed model and quality
acceptance require terminal artifacts and independent scoring.

The [professional profiler verification](reports/monitoring/STATUS.md) collected
real Nsight Systems and PyTorch GPU traces after cold restoration. All 91 indexed
output files passed SHA256 checks, and adapter/Adam drift passed the frozen diagnostic
tolerances while exact hashes differed. NCU 2026.1.1 failed on driver 535.261.03;
a separate one-GPU compatibility run is queued. Its absence remains a failed counter
capture. The first suite consumed **0.527 H20 GPU hours**, including preparation and
failure, separate from historical and DEV allocations. CPU regressions passed **136
tests**. Profiling diagnostics do not establish speed or quality superiority.

The following paragraphs describe older, closed studies. Their historical GPU
counts do not change the current eight-GPU limit.

The [four-seed AOPD router confirmation](reports/advantage/CONFIRMATION_AOPD_DOMAIN_ROUTER.md) completed all 12 training arms and both independent holdout splits. It did not establish superiority over offline SFT: business macro differences were −0.147 pp on test and −0.202 pp on replication, with both 95% intervals crossing zero. Seed 401 also failed the 99% JSON-validity gate. The [confirmation cost audit](reports/advantage/CONFIRMATION_AOPD_DOMAIN_ROUTER_COST.json) retains all 12 trials and 46.548 H20 GPU hours; this is separate from the earlier studies below. Further method development uses DEV, and a new confirmation requires unseen data.

The original study completed **28 training stages × 80 updates**, two seeds, final test/replication, frozen dev selection, BF16 native vLLM serving, protocol fallback, and 32 CPU regressions. All original-study GPU allocations are released: **59.75 H20 GPU hours**, peak 32 GPUs, 4.21 hours from first allocation to final release.

The dev-selected OPD-family method (`full_kl`) reached a two-seed business macro score of **49.25% vs 25.86%** initially; the limited general score increased **17.75 pp**. Substitute macro-F1 fell **5.48 pp**, failing the final 5 pp retention gate. Dev selection favored offline SFT; this study does **not establish OPD superiority or production readiness**. Fictional local-services output-format transfer also failed. These negative outcomes are part of the result.

Read the [final report](reports/FINAL_REPORT.md), [interview notes](reports/INTERVIEW_NOTES.md), [measured statistics](reports/results.json), and [complete allocation cost](reports/cost-audit.json). [Task score evidence](reports/evidence/task-metrics.jsonl.gz) supports independent paired-statistics checks.

A separate [H20 profiling study](reports/profiling/REPORT.md) includes real Nsight CUDA/NVTX traces, PyTorch input shapes, a microbatch 1/4 diagnostic, and reproducible commands. Kernel count fell 63.9% in the traced comparison; runtime ratios remain diagnostic because profiler overhead and shape compilation affect timing. Both observation allocations are released: **0.87 additional H20 GPU hours**, or **60.62 GPU hours** including the original study. See the [observation cost audit](reports/profiling/cost-audit.json).

The [expanded experiment](reports/expanded/REPORT.md) completed **32 frozen workload banks, 384 unprofiled timing measurements, and 8 arms × 240 updates**, with **881,401 effective training response tokens**. Learner throughput at microbatch 4 was **2.98×** microbatch 1 (paired-bank 95% CI **[2.88, 3.10]**); observed complete training was **1.98×** faster. Each final checkpoint covered 6,600 test and 2,200 independent replication queries. All three business domains passed the predeclared −5 pp m4-versus-m1 noninferiority criterion, but **every final substitute classifier predicts only `no`, with zero positive recall**; relevance also favors E/I. Execution gains do not establish useful absolute business quality. [Class evidence](reports/expanded/classification-audit.json) retains these failures. Microbatch 8 added 5.19 GiB allocated memory without established additional throughput benefit over 4. All expansion resources are released: **43.64 additional H20 GPU hours**, **104.27 project GPU hours** in total. The [observation guide](reports/expanded/OBSERVATION_GUIDE.md) explains timing boundaries, CUDA attribution, parameter effects, and exercises for checking the saved evidence.

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

`main` contains runnable code, tests, experiment configurations, and reproducible
reports. `doc` contains the latest research notes, manuscripts, and sanitized
conversation excerpts. The existing `manuscript` branch preserves its historical
notes. Resource identifiers, credentials, internal endpoints, and unrelated
workspace files are excluded from public history.

Check out `doc` alongside `main` without switching the current directory:

```bash
bash scripts/open_reading_workspace.sh ../ServeDistill-doc
code ServeDistill.code-workspace
```

The repository includes an exact source snapshot for the frozen DEV protocol:

```bash
python scripts/materialize_experiment.py --output ../ServeDistill-anchor-v1
```

Run the protocol from that generated source directory with its hash-bound input
assets. This avoids silently substituting newer monitoring code into the
submitted experiment.

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
