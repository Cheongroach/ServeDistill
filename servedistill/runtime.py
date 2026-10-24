"""Hugging Face reference runtime with audited response-token boundaries."""
from __future__ import annotations

import contextlib
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .data import BUSINESS, SYSTEM, TEACHER_SYSTEM, grade


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def append_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def action_mask(tokens: torch.Tensor, eos_ids: list[int]) -> torch.Tensor:
    """Include the first EOS action, exclude padding and later EOS values."""
    eos = torch.zeros_like(tokens, dtype=torch.bool)
    for token in eos_ids:
        eos |= tokens == token
    before = eos.long().cumsum(-1) - eos.long()
    return before.eq(0)


def summarize(records: list[dict]) -> dict:
    domains = defaultdict(list)
    for record in records:
        domains[record["domain"]].append(record)
    scores = {}
    for domain, values in domains.items():
        scores[domain] = {
            "n": len(values),
            **{key: sum(float(row[key]) for row in values) / len(values)
               for key in ("success", "json_valid", "constraint_violation", "truncated")},
            "mean_output_tokens": np.mean([row["output_tokens"] for row in values]).item(),
        }
        scores[domain]["selection_score"] = scores[domain]["success"]
        if values[0].get("task") == "esci_ranking":
            eligible = [row for row in values if row["ndcg_eligible"]]
            scores[domain]["ndcg_n"] = len(eligible)
            scores[domain]["ndcg"] = float(np.mean([r["ndcg"] for r in eligible])) if eligible else 0.0
            scores[domain]["mrr"] = float(np.mean([r["mrr"] for r in values]))
            scores[domain]["selection_score"] = scores[domain]["ndcg"]
        elif values[0].get("task") in ("esci_classification", "esci_substitute"):
            from sklearn.metrics import f1_score
            options = ["E", "S", "C", "I"] if domain == "relevance" else ["yes", "no"]
            truth = [row["target_class"] for row in values]
            predicted = [row["predicted_class"] for row in values]
            scores[domain]["macro_f1"] = f1_score(
                truth, predicted, labels=options, average="macro", zero_division=0)
            scores[domain]["selection_score"] = scores[domain]["macro_f1"]
    business = [value["selection_score"] for domain, value in scores.items()
                if domain not in ("math", "instruction")]
    general = [scores[d]["success"] for d in ("math", "instruction") if d in scores]
    return {
        "domains": scores, "business_macro": float(np.mean(business)) if business else None,
        "general_macro": float(np.mean(general)) if general else None,
        "overall": sum(row["success"] for row in records) / len(records),
    }


def stratified(rows: list[dict], per_domain: int) -> list[dict]:
    counts = defaultdict(int)
    result = []
    for row in rows:
        if counts[row["domain"]] < per_domain:
            counts[row["domain"]] += 1
            result.append(row)
    return result


class Float32Output(torch.nn.Module):
    """Compute vocabulary projection without BF16 output-logit rounding."""
    def __init__(self, original):
        super().__init__()
        self.weight = torch.nn.Parameter(original.weight.float(), requires_grad=False)

    def forward(self, hidden):
        return torch.nn.functional.linear(hidden.float(), self.weight)


class Runtime:
    def __init__(self, student_path: str, teacher_path: str | None = None,
                 adapter_path: str | None = None, train: bool = False,
                 student_device: str = "cuda:0", teacher_device: str = "cuda:0",
                 train_precision: str = "float32_torch"):
        started = time.monotonic()
        self.tokenizer = AutoTokenizer.from_pretrained(student_path, local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.student = self.load_model(student_path, student_device)
        if train:
            from peft import LoraConfig, get_peft_model
            hybrid = self.student.config.model_type == "qwen3_5_text"
            self.student = get_peft_model(self.student, LoraConfig(
                task_type="CAUSAL_LM", r=16, lora_alpha=32, lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"] + (
                                    ["in_proj_qkv", "in_proj_z", "out_proj"] if hybrid else []),
            ))
            if adapter_path:
                from peft import set_peft_model_state_dict
                from safetensors.torch import load_file
                set_peft_model_state_dict(
                    self.student, load_file(str(Path(adapter_path) / "adapter_model.safetensors")))
            self.student.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            self.student.enable_input_require_grads()
            base = self.student.get_base_model()
            if train_precision == "float_head":
                base.lm_head = Float32Output(base.lm_head)
            elif train_precision == "float32_torch":
                base.float()
                from transformers.models.qwen3_5.modeling_qwen3_5 import (
                    torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule,
                    torch_causal_conv1d_update)
                for layer in base.model.layers:
                    attention = getattr(layer, "linear_attn", None)
                    if attention is not None:
                        attention.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
                        attention.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
                        attention.causal_conv1d_fn = None
                        attention.causal_conv1d_update = torch_causal_conv1d_update
            elif train_precision != "bf16":
                raise ValueError("Unknown training precision")
        elif adapter_path:
            from peft import PeftModel
            self.student = PeftModel.from_pretrained(self.student, adapter_path).merge_and_unload()
        self.student.eval()  # Keep dropout off in both rollout and learner recomputation.
        self.teacher = None
        self.teacher_tokenizer = None
        if teacher_path:
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path, local_files_only=True)
            self.teacher_tokenizer.padding_side = "left"
            if self.teacher_tokenizer.pad_token_id is None:
                self.teacher_tokenizer.pad_token_id = self.teacher_tokenizer.eos_token_id
            self.verify_tokenizers()
            self.teacher = self.load_model(teacher_path, teacher_device)
            self.teacher.requires_grad_(False)
            self.teacher.eval()
        self.eos_ids = self.student.generation_config.eos_token_id
        if isinstance(self.eos_ids, int):
            self.eos_ids = [self.eos_ids]
        if not self.eos_ids:
            self.eos_ids = [self.tokenizer.eos_token_id]
        self.load_seconds = time.monotonic() - started
        self.contract = {
            "tokenizer_vocab_sha256": hashlib.sha256(
                json.dumps(self.tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
            "eos_ids": self.eos_ids, "pad_id": self.tokenizer.pad_token_id,
            "student_vocab": self.student.config.vocab_size,
            "teacher_vocab": self.teacher.config.vocab_size if self.teacher else None,
            "temperature": 1.0, "top_p": 1.0, "top_k": 0,
            "response_mask": "first EOS included; later padding excluded",
            "attention": "SDPA full attention",
            "delta_kernels": {
                name: str(getattr(layer.linear_attn, name).__module__)
                for layer in (self.student.get_base_model() if train else self.student).model.layers
                if hasattr(layer, "linear_attn")
                for name in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule")},
            "precision": train_precision if train else "bf16",
            "load_seconds": self.load_seconds,
            "enable_thinking": False, "architecture": self.student.config.model_type,
            "loading": getattr(self.student.get_base_model() if train else self.student,
                               "_servedistill_loading", {}),
        }

    @staticmethod
    def load_model(path: str, device: str):
        raw = json.loads((Path(path) / "config.json").read_text())
        options = dict(
            local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
            attn_implementation="sdpa", device_map={"": device},
            output_loading_info=True,
        )
        if raw.get("model_type") in ("qwen3_5", "qwen3_5_text"):
            from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
            # This business experiment uses text. The native class explicitly
            # filters vision/MTP keys and retains all language parameters.
            config = Qwen3_5TextConfig.from_pretrained(path, local_files_only=True)
            model, loading = Qwen3_5ForCausalLM.from_pretrained(path, config=config, **options)
        else:
            model, loading = AutoModelForCausalLM.from_pretrained(path, **options)
        errors = {key: sorted(loading.get(key)) if isinstance(loading.get(key), set)
                  else loading.get(key) for key in
                  ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
                  if loading.get(key)}
        if errors:
            raise RuntimeError(f"Model weights did not load exactly: {errors}")
        model._servedistill_loading = {
            "missing_keys": [], "unexpected_keys": [], "mismatched_keys": [],
            "text_only": raw.get("model_type") in ("qwen3_5", "qwen3_5_text"),
        }
        return model.eval()

    def verify_tokenizers(self) -> None:
        a, b = self.tokenizer, self.teacher_tokenizer
        if a.get_vocab() != b.get_vocab():
            raise ValueError("Teacher and student token event spaces differ")
        if a.all_special_ids != b.all_special_ids:
            raise ValueError("Teacher/student special token IDs differ")
        # Qwen3.5 sizes have different thinking defaults in the template source.
        # Compare the actual shared mode, rather than rejecting equivalent renders.
        probes = [
            [{"role": "system", "content": "Return JSON only."},
             {"role": "user", "content": text}]
            for text in ("Search products.", "搜索商品。", "商品を検索。")]
        for messages in probes:
            options = dict(tokenize=True, add_generation_prompt=True, enable_thinking=False)
            if a.apply_chat_template(messages, **options) != b.apply_chat_template(messages, **options):
                raise ValueError("Teacher/student rendered chat token sequences differ")

    def prompts(self, rows: list[dict], teacher: bool = False) -> torch.Tensor:
        tokenizer = self.teacher_tokenizer if teacher else self.tokenizer
        texts = [tokenizer.apply_chat_template([
            {"role": "system", "content": (
                row.get("teacher_system", TEACHER_SYSTEM) if teacher
                else row.get("system", SYSTEM))},
            {"role": "user", "content": row["prompt"]},
        ], tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in rows]
        model = self.teacher if teacher else self.student
        return tokenizer(texts, padding=True, return_tensors="pt").to(model.device)

    def rollout(self, rows: list[dict], max_new_tokens: int = 128,
                sample: bool = True, teacher: bool = False, audit: bool = True) -> dict:
        model = self.teacher if teacher else self.student
        tokenizer = self.teacher_tokenizer if teacher else self.tokenizer
        inputs = self.prompts(rows, teacher)
        config = GenerationConfig(
            do_sample=sample, temperature=1.0, top_p=1.0, top_k=0,
            max_new_tokens=max_new_tokens, repetition_penalty=1.0,
            eos_token_id=self.eos_ids, pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        torch.cuda.synchronize(model.device)
        started = time.monotonic()
        with torch.inference_mode():
            result = model.generate(
                **inputs, generation_config=config, return_dict_in_generate=True,
                output_scores=audit,
            )
        torch.cuda.synchronize(model.device)
        elapsed = time.monotonic() - started
        response = result.sequences[:, inputs.input_ids.shape[1]:].clone()
        mask = action_mask(response, self.eos_ids)
        old_logp = None
        if audit:
            old_logp = torch.stack([
                scores.float().log_softmax(-1).gather(1, response[:, index, None]).squeeze(-1)
                for index, scores in enumerate(result.scores)
            ], -1).clone()
        # Clones outside inference_mode can be used by autograd for indexed gathers.
        return {
            "rows": rows, "tokens": response.cpu().clone(), "mask": mask.cpu().clone(),
            "behavior_logp": old_logp.cpu().clone() if old_logp is not None else None,
            "texts": tokenizer.batch_decode(response, skip_special_tokens=True),
            "elapsed_seconds": elapsed,
        }

    def score(self, rows: list[dict], response: torch.Tensor, mask: torch.Tensor,
              teacher: bool = False, reference: bool = False,
              entropy: bool = False, keep_logits: bool = False) -> dict:
        model = self.teacher if teacher else self.student
        inputs = self.prompts(rows, teacher)
        response = response.to(model.device)
        mask = mask.to(model.device)
        input_ids = torch.cat([inputs.input_ids, response], -1)
        attention = torch.cat([inputs.attention_mask, mask.long()], -1)
        positions = attention.long().cumsum(-1) - 1
        positions.masked_fill_(attention == 0, 0)
        adapter_context = model.disable_adapter() if reference else contextlib.nullcontext()
        grad_context = torch.no_grad() if teacher or reference else contextlib.nullcontext()
        was_training = model.training
        # Checkpointing activates in train mode; all dropout probabilities are zero.
        model.train(not (teacher or reference))
        try:
            with adapter_context, grad_context:
                options = {}
                response_only = model.config.model_type == "qwen3_5_text"
                if response_only:
                    options["logits_to_keep"] = torch.arange(
                        inputs.input_ids.shape[1] - 1, input_ids.shape[1] - 1,
                        device=model.device)
                output = model(
                    input_ids=input_ids, attention_mask=attention, position_ids=positions,
                    use_cache=False, **options,
                )
                logits = (output.logits if response_only else
                          output.logits[:, inputs.input_ids.shape[1] - 1:-1]).float()
                logp = logits.gather(-1, response.unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
                result = {"logp": logp}
                if entropy:
                    log_probs = logits.log_softmax(-1)
                    result["entropy"] = -(log_probs.exp() * log_probs).sum(-1)
                if keep_logits:
                    result["logits"] = logits
        finally:
            model.train(was_training)
        return result

    def evaluate(self, rows: list[dict], output: Path, label: str,
                 batch_size: int = 8, max_new_tokens: int = 128,
                 teacher: bool = False, version: int = 0) -> dict:
        records = []
        path = output / f"{label}.jsonl"
        if path.exists():
            raise FileExistsError("Refusing to overwrite evaluation evidence: " + str(path))
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            rollout = self.rollout(batch, max_new_tokens, sample=False,
                                   teacher=teacher, audit=False)
            for row, text, tokens, mask in zip(
                    batch, rollout["texts"], rollout["tokens"], rollout["mask"]):
                ended = any(int(token) in self.eos_ids for token in tokens[mask])
                record = {
                    "sample_id": row["sample_id"], "domain": row["domain"],
                    "locale": row.get("locale", "general"),
                    "input_sha256": row["input_sha256"], "text": text,
                    "output_tokens": int(mask.sum()), "truncated": not ended,
                    "version": version, "batch_seconds": rollout["elapsed_seconds"],
                    **grade(row, text),
                }
                records.append(record)
                append_json(path, record)
        report = summarize(records)
        write_json(output / f"{label}.summary.json", report)
        print(json.dumps({"event": "evaluation", "label": label, **report}), flush=True)
        return report
