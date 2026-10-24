# ServeDistill design notes

2026-10-24. Companion to main; decisions and bounded evidence, not a claim of final business success.

The intended user is a business algorithm team deciding whether to use on-policy distillation to deploy a smaller model or recover an existing capability. Success requires per-domain outcomes, capability regression checks, and full teacher/student costs. Multiexpert model merging is one possible use, not the whole project.

The primary benchmark moved from fictional local-services constraints to real Amazon ESCI query/product labels. Ranking, E/S/C/I classification and substitute identification provide distinct outcome rewards. Coupon, recommendation and advertising datasets are extensions requiring their own domain teachers and independent temporal/propensity assumptions; anonymous feature IDs are not converted into invented LLM semantics.

Qwen3.5-9B and Qwen3.5-2B were chosen as contemporary teacher/student, with Qwen3.5-4B as a serving reference. Both run with thinking disabled and a shared token event space. Gated DeltaNet and full attention projections receive LoRA. Prompt-routed teachers remain public checkpoints with specialized prompts, not independently trained domain experts.

The BF16 rollout/recompute gate rejected a batch at log-probability error 0.193709 after matching cold-restoration fingerprints. A targeted precision comparison found maximum error 0.148020 for BF16, 0.089907 for an FP32 output projection, and 0.000046253 for an FP32 student with the PyTorch DeltaNet path. The conservative main training path uses the latter. Four nonzero updates and adapter/Adam/RNG restoration subsequently passed. Precision-path diagnostic timing is not a fair benchmark because kernels warmed sequentially.

At 100 dev tasks per domain, teacher has strong headroom for ranking, instruction and the limited GSM8K sample; relevance macro-F1 also improves. Substitute macro-F1 falls from 0.4409 to 0.4012, with a paired interval spanning zero. This domain is retained as a negative control, not advertised as transferable headroom.

A fixed-state sampled-token reverse-KL gradient differs from full trajectory differentiation. The student generates first; the teacher scores only visited prefixes; the optimizer updates once per batch. GRPO trajectory advantages and OPD token log-ratios stay separate. Mixed loss weights, alternating updates, bounded dev-controlled weights, batch bucketing and length normalization receive independent ablations.

Two seeds and 80 optimizer updates are frozen before final model comparisons. Actual response tokens and GPU residency are reported separately; equal updates cannot be described as equal tokens or equal teacher cost. The general-recovery stage starts from the corresponding dev-selected business-only SFT checkpoint.

The public reranker reference is Qwen3-Reranker-0.6B, a specialized joint query/document scorer. Its frozen search test NDCG was 0.8807. This observation does not pick an OPD hyperparameter or establish the final student comparison. Serving metrics distinguish in-process static batching, scheduler timing and real HTTP boundaries.

Final deployment selection must use dev and an explicit general-regression constraint. Final test/OOD labels cannot control weights, reward or checkpoint choice. The extra official-test replication is called replication, not proven distribution shift.
