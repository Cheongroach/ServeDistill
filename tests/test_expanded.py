import ast
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from servedistill.scheduling import Config
from servedistill.losses import DevWeightController
from servedistill.train import apply_rollout, load_checkpoint, save_checkpoint, tensor_fingerprint
from scripts.benchmark_microbatch import bank_order, orders


class ToyRuntime:
    def __init__(self):
        self.weight=torch.nn.Parameter(torch.tensor([.2,-.1,.4]))
        self.student=type('Student',(),{'device':torch.device('cpu')})()
        self.teacher=torch.nn.Linear(1,1).requires_grad_(False)
        self.teacher_calls=0

    def score(self,rows,tokens,mask,teacher=False,reference=False,entropy=False,keep_logits=False):
        if teacher:
            self.teacher_calls+=1
            logits=torch.tensor([.1,.3,-.2]).expand(*tokens.shape,3)
        elif reference:
            logits=torch.tensor([.0,.0,.0]).expand(*tokens.shape,3)
        else:logits=self.weight.expand(*tokens.shape,3)
        normalizer=logits.logsumexp(-1)
        logs=logits-normalizer.unsqueeze(-1)
        result={'logp':logs.gather(-1,tokens.unsqueeze(-1)).squeeze(-1),
                'log_normalizer':normalizer}
        if entropy:result['entropy']=-(logs.exp()*logs).sum(-1)
        if keep_logits:result['logits']=logits
        return result


class ExpandedTests(unittest.TestCase):
    def update(self,microbatch,frozen=False,objective='full_kl'):
        runtime=ToyRuntime()
        tokens=torch.tensor([[0,1,2],[1,2,0],[2,0,1],[1,0,2]])
        mask=torch.tensor([[1,1,0],[1,0,0],[1,1,1],[1,1,0]],dtype=torch.bool)
        rows=[{'sample_id':str(i)} for i in range(4)]
        with torch.no_grad():old=runtime.score(rows,tokens,mask)['logp'].clone()
        rollout={'rows':rows,'tokens':tokens,'mask':mask,'behavior_logp':old}
        params=[runtime.weight];before=[runtime.weight.detach().clone()]
        opt=torch.optim.AdamW(params,lr=2e-5,weight_decay=0.)
        cfg=Config(objective=objective,microbatch=microbatch,aopd_top_k=3)
        teacher=runtime.score(rows,tokens,mask,teacher=True,keep_logits=True) if frozen else None
        runtime.teacher_calls=0
        with tempfile.TemporaryDirectory() as tmp,patch('torch.cuda.synchronize'):
            result=apply_rollout(runtime,opt,cfg,DevWeightController(1.),rollout,
                torch.zeros(4),params,before,41,Path(tmp),{'teacher':0.,'learner':0.},
                lambda _:nullcontext(),teacher)
        return result,runtime.weight.detach(),runtime.weight.grad,runtime.teacher_calls

    def test_token_weighted_update_matches_across_microbatch(self):
        a=self.update(1);b=self.update(4)
        self.assertAlmostEqual(a[0][0]['loss'],b[0][0]['loss'],places=6)
        torch.testing.assert_close(a[1],b[1],atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(a[2],b[2],atol=1e-7,rtol=1e-6)
        self.assertEqual(a[3],4);self.assertEqual(b[3],1)

    def test_aopd_update_matches_across_microbatch(self):
        a=self.update(1,objective='aopd');b=self.update(4,objective='aopd')
        self.assertAlmostEqual(a[0][0]['loss'],b[0][0]['loss'],places=6)
        torch.testing.assert_close(a[1],b[1],atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(a[2],b[2],atol=1e-7,rtol=1e-6)
        self.assertAlmostEqual(a[0][0]['aopd_intervention_fraction'],
                               b[0][0]['aopd_intervention_fraction'],places=6)

    def test_frozen_teacher_has_no_recomputed_teacher_calls(self):
        a=self.update(1);b=self.update(1,True)
        self.assertEqual(b[3],0)
        torch.testing.assert_close(a[2],b[2],atol=1e-7,rtol=1e-6)

    def test_bank_workers_are_disjoint_and_orders_complete(self):
        values=[({'domain':domain,'sample_id':f'{domain}-{i}'},i)
                for domain in ['search','relevance','substitute','math','instruction'] for i in range(200)]
        a=bank_order(values,0);b=bank_order(values,1)
        ids=lambda banks:{r['sample_id'] for bank in banks for r in bank}
        self.assertEqual(len(a),16);self.assertEqual(len(ids(a)),128)
        self.assertFalse(ids(a)&ids(b))
        for index in range(16):
            for order in orders(index):self.assertEqual(sorted(order),[1,2,4,8])

    def test_schedule_configurations_have_matching_pairs(self):
        root=Path(__file__).resolve().parents[1]/'configs/expanded'
        for seed in [17,29,43,71]:
            a=json.loads((root/f'full_kl_m1_seed{seed}.json').read_text())
            b=json.loads((root/f'full_kl_m4_seed{seed}.json').read_text())
            self.assertEqual(a.pop('microbatch'),1);self.assertEqual(b.pop('microbatch'),4)
            self.assertEqual(a,b);self.assertEqual(a['steps'],240)

    def test_common_parent_restores_real_adam_state_and_detects_changed_adapter(self):
        class Student(torch.nn.Linear):
            device = torch.device("cpu")

            def save_pretrained(self, path):
                torch.save(self.state_dict(), path / "adapter.pt")

        runtime = type("Runtime", (), {"student": Student(2, 2)})()
        config = Config(seed=17)
        optimizer = torch.optim.AdamW(runtime.student.parameters(), lr=2e-5)
        runtime.student(torch.ones(1, 2)).square().sum().backward()
        optimizer.step()
        cuda_rng = [torch.tensor([1, 2, 3], dtype=torch.uint8)]
        with tempfile.TemporaryDirectory() as directory, \
                patch("torch.cuda.get_rng_state_all", return_value=cuda_rng), \
                patch("torch.cuda.set_rng_state_all"):
            root = Path(directory)
            saved = save_checkpoint(runtime, optimizer, config, DevWeightController(1.),
                                    40, root, .5, 100, [])
            original_rng = tensor_fingerprint(np.random.get_state())
            optimizer.param_groups[0]["lr"] = 1.
            np.random.seed(99)
            restored = load_checkpoint(runtime, optimizer, root / "checkpoint-0040",
                                       config, DevWeightController(1.))
            self.assertTrue(restored["restoration"]["passed"])
            self.assertEqual(optimizer.param_groups[0]["lr"], 2e-5)
            self.assertEqual(tensor_fingerprint(np.random.get_state()), original_rng)
            self.assertEqual(restored["restoration"]["optimizer_sha256"], saved["optimizer_sha256"])
            with torch.no_grad():
                runtime.student.weight.add_(.1)
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                load_checkpoint(runtime, optimizer, root / "checkpoint-0040",
                                config, DevWeightController(1.))


if __name__=='__main__':unittest.main()
