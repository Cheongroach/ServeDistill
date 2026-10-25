"""Paired, unprofiled learner steps on frozen trajectories and identical Adam state."""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import random
import time

import torch
import numpy as np
from servedistill.data import grade, load_rows
from servedistill.losses import DevWeightController, grpo_advantages
from servedistill.runtime import Runtime, seed_all, write_json, append_json
from servedistill.scheduling import Config
from servedistill.train import (apply_rollout, collate_rollouts, load_checkpoint,
                                parameter_state, tensor_fingerprint)


def bank_order(values, worker_index, workers=2):
    """Disjoint prompt sets; spread each bank across domains and real token-length ranks."""
    buckets = defaultdict(list)
    for row, length in values:
        buckets[row['domain']].append((length, row))
    selected = []
    per_domain = 26
    for domain in sorted(buckets):
        pool = sorted(buckets[domain], key=lambda pair: (pair[0], pair[1]['sample_id']))
        local = pool[worker_index::workers]
        for i in range(per_domain):
            selected.append(local[min(len(local)-1, int((i+.5)*len(local)/per_domain))][1])
    selected = selected[:128]
    if len({row['sample_id'] for row in selected}) != 128:
        raise ValueError('Bank sampling lacks distinct prompts')
    rng = random.Random(20261025 + worker_index)
    rng.shuffle(selected)
    return [selected[i:i+8] for i in range(0, len(selected), 8)]


def orders(bank, repetitions=3):
    values = [1, 2, 4, 8]
    return [values[(bank+rep) % 4:] + values[:(bank+rep) % 4]
            if rep % 2 == 0 else list(reversed(values[(bank+rep) % 4:] + values[:(bank+rep) % 4]))
            for rep in range(repetitions)]


class PhaseTimer:
    def __init__(self, nvtx=False):
        self.nvtx = nvtx
        self.times = defaultdict(float)

    @contextmanager
    def span(self, name):
        if self.nvtx:
            torch.cuda.nvtx.range_push(name)
        started = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] += time.perf_counter()-started
            if self.nvtx:
                torch.cuda.nvtx.range_pop()


def main():
    parser = argparse.ArgumentParser()
    for key in ('student', 'teacher', 'checkpoint', 'data', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--worker-index', type=int, default=0)
    parser.add_argument('--banks', type=int, default=16)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--trace-bank', type=int)
    parser.add_argument('--trace-microbatch', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_all(20261025+args.worker_index)
    runtime = Runtime(str(args.student), str(args.teacher), str(args.checkpoint), train=True)
    state = torch.load(args.checkpoint/'training.pt', map_location='cpu', weights_only=False)
    config = Config(**state['config'])
    config.prompts_per_step = 8
    params = list(parameter_state(runtime).values())
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=0.)
    parent_config = Config(**state['config'])
    restored = load_checkpoint(runtime, optimizer, args.checkpoint, parent_config,
                               DevWeightController(config.opd_weight))
    write_json(args.output/'parent-restoration.json', restored['restoration'])
    write_json(args.output/'runtime-contract.json', runtime.contract)
    saved = [p.detach().clone() for p in params]
    adam = copy.deepcopy(state['optimizer'])
    rng = dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())
    del state, restored

    def reset():
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for p, value in zip(params, saved):
                p.copy_(value)
        optimizer.load_state_dict(copy.deepcopy(adam))
        for values in optimizer.state.values():
            for key, value in values.items():
                if isinstance(value, torch.Tensor):
                    values[key] = value.to(runtime.student.device)
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['cpu'])
        torch.cuda.set_rng_state_all(rng['cuda'])
        torch.cuda.synchronize()

    if args.trace_bank is None:
        rows = load_rows(args.data/'train.jsonl')
        lengths = [(row, len(runtime.tokenizer.encode(row['prompt']))) for row in rows]
        banks = bank_order(lengths, args.worker_index)
        write_json(args.output/'bank-prompt-ids.json', [[r['sample_id'] for r in bank] for bank in banks])
    else:
        banks = [None]*args.banks
    for index, prompts in enumerate(banks[:args.banks]):
        bank_file = args.output/f'bank-{index:02d}.pt'
        if args.trace_bank is not None and index != args.trace_bank:
            continue
        reset()
        if bank_file.exists():
            rollout = torch.load(bank_file, map_location='cpu', weights_only=False)
        else:
            rollout = collate_rollouts(runtime, prompts, config)
            torch.save(rollout, bank_file)
            write_json(args.output/f'bank-{index:02d}-contract.json', {
                'sample_ids': [r['sample_id'] for r in rollout['rows']],
                'token_sha256': tensor_fingerprint(rollout['tokens']),
                'mask_sha256': tensor_fingerprint(rollout['mask']),
                'behavior_logp_sha256': tensor_fingerprint(rollout['behavior_logp']),
                'effective_tokens': int(rollout['mask'].sum()),
                'prompt_token_lengths': [len(runtime.tokenizer.encode(r['prompt'])) for r in rollout['rows']],
                'response_width': rollout['tokens'].shape[1], 'rollout_seconds': rollout['elapsed_seconds'],
            })
        reward = torch.tensor([grade(r,t)['reward'] for r,t in zip(rollout['rows'],rollout['texts'])])
        advantages, _ = grpo_advantages(reward, rollout['groups'])
        sequence = [(True,-1,m) for m in (1,2,4,8)] + [
            (False,rep,m) for rep,order in enumerate(orders(index,args.repetitions)) for m in order]
        if args.trace_bank is not None:
            sequence = [(True,-1,args.trace_microbatch), (False,0,args.trace_microbatch)]
        vectors = {}
        for warm, rep, microbatch in sequence:
            reset()
            config.microbatch = microbatch
            timer = PhaseTimer(nvtx=args.trace_bank is not None and not warm)
            totals = {'teacher':0.,'learner':0.}
            torch.cuda.reset_peak_memory_stats()
            if timer.nvtx:
                torch.cuda.profiler.start()
                torch.cuda.nvtx.range_push('fixed_bank_update')
            began = time.perf_counter()
            result = apply_rollout(runtime, optimizer, config, DevWeightController(1.),
                rollout, advantages, params, saved, 41, args.output, totals, timer.span)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-began
            if timer.nvtx:
                torch.cuda.nvtx.range_pop()
                torch.cuda.profiler.stop()
            audit, error, norm, delta = result
            gradient = torch.cat([p.grad.detach().float().cpu().reshape(-1) for p in params if p.grad is not None])
            update = torch.cat([(p.detach().float()-old.float()).cpu().reshape(-1) for p,old in zip(params,saved)])
            if rep == 0 and not warm:
                vectors[microbatch] = (gradient, update)
            record = {'worker_index':args.worker_index, 'bank':index, 'warmup':warm,
                'repetition':rep, 'microbatch':microbatch, 'profiled':timer.nvtx,
                'learner_update_seconds':elapsed,'effective_response_tokens':int(rollout['mask'].sum()),
                'response_tokens_per_second':int(rollout['mask'].sum())/elapsed,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                'phase_host_seconds':dict(timer.times), 'behavior_max_error':error,
                'gradient_norm_preclip':norm, 'parameter_delta_l2':delta, 'loss':audit['loss'],
                'parent_reset':True, 'response_width':rollout['tokens'].shape[1]}
            append_json(args.output/'measurements.jsonl',record)
            print(json.dumps(record),flush=True)
            del gradient,update
        if args.trace_bank is None:
            reference_grad, reference_update = vectors[1]
            for microbatch, (gradient, update) in vectors.items():
                append_json(args.output/'native-numerics.jsonl',{
                    'bank':index, 'microbatch':microbatch,
                    'gradient_cosine_postclip':torch.nn.functional.cosine_similarity(reference_grad,gradient,dim=0).item(),
                    'gradient_relative_l2_postclip':((gradient-reference_grad).norm()/reference_grad.norm()).item(),
                    'update_relative_l2':((update-reference_update).norm()/reference_update.norm()).item()})
        del vectors
        if index < 2 and args.trace_bank is None:
            # Fix Teacher logits for one four-sequence slice, isolating Teacher batch effects.
            subset = {k: (v[:4] if isinstance(v,torch.Tensor) else v[:4] if isinstance(v,list) else v)
                      for k,v in rollout.items()}
            reset()
            frozen = runtime.score(subset['rows'],subset['tokens'],subset['mask'],teacher=True,keep_logits=True)
            frozen = {k:v.detach().cpu() for k,v in frozen.items()}
            comparisons = {}
            for microbatch in (1,4):
                reset();config.microbatch=microbatch
                totals = {'teacher':0.,'learner':0.}
                result=apply_rollout(runtime,optimizer,config,DevWeightController(1.),subset,
                    advantages[:4],params,saved,41,args.output,totals,PhaseTimer().span,frozen)
                comparisons[microbatch]=torch.cat([p.grad.detach().float().cpu().reshape(-1)
                                                  for p in params if p.grad is not None])
            a,b=comparisons[1],comparisons[4]
            append_json(args.output/'frozen-teacher-numerics.jsonl',{
                'bank':index,'sequences':4,'gradient_cosine':torch.nn.functional.cosine_similarity(a,b,dim=0).item(),
                'gradient_relative_l2':((b-a).norm()/a.norm()).item(),
                'teacher_logits_sha256':tensor_fingerprint(frozen['logits']),
                'boundary':'fixed Teacher four-sequence diagnostic; not native throughput'})
            del frozen,comparisons
    write_json(args.output/'completed.json',{'completed':True,'banks':args.banks,
        'repetitions':args.repetitions,'worker_index':args.worker_index,
        'trace_only':args.trace_bank is not None,'fixed_parent_state':True})


if __name__=='__main__':
    main()
