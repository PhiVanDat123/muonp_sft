"""CPU optimizer integration checks, without optional training dependencies.

Run: python -m unittest discover -s tests -p test_muon_p.py
"""
import copy
from contextlib import nullcontext
import importlib
from pathlib import Path
import sys
import types
import unittest
from types import SimpleNamespace

import torch

# Load optimizer modules without alignment.__init__'s dataset/TRL dependencies.
package = types.ModuleType('_optimizer_checks')
package.__path__ = [str(Path(__file__).resolve().parents[1] / 'alignment')]
sys.modules[package.__name__] = package
module = importlib.import_module('_optimizer_checks.muon_p')
factory = importlib.import_module('_optimizer_checks.optimizers').create_custom_optimizer


class MuonPTests(unittest.TestCase):
    def args(self, **overrides):
        values = dict(custom_optimizer='muon_p', learning_rate=0.01,
                      adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8,
                      weight_decay=0.1, muon_backend='halfpower_svd')
        values.update(overrides)
        return SimpleNamespace(**values)

    def model(self):
        return torch.nn.ModuleDict(dict(
            embed_tokens=torch.nn.Embedding(8, 4),
            proj=torch.nn.Linear(4, 4), lm_head=torch.nn.Linear(4, 8)))

    def test_factory_partition_adamw_and_resume(self):
        torch.manual_seed(7)
        model = self.model()
        opt = factory(model, self.args())
        matrix = next(g for g in opt.param_groups if g['use_muon'])
        self.assertEqual([id(p) for p in matrix['params']], [id(model['proj'].weight)])
        self.assertAlmostEqual(matrix['lr'], 0.003)
        aux = next(g for g in opt.param_groups if not g['use_muon'])
        reference = [torch.nn.Parameter(p.detach().clone()) for p in aux['params']]
        adam = torch.optim.AdamW(reference, lr=.01, betas=(.9,.95), eps=1e-8, weight_decay=.1)
        for _ in range(3):
            for p in model.parameters():
                p.grad = torch.randn_like(p)
            for p, q in zip(aux['params'], reference):
                q.grad = p.grad.clone()
            opt.step()
            adam.step()
        for p, q in zip(aux['params'], reference):
            torch.testing.assert_close(p, q)
        clone = copy.deepcopy(model)
        resumed = factory(clone, self.args())
        resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
        scheduler = torch.optim.lr_scheduler.LambdaLR(resumed, lambda _: .5)
        for p, q in zip(model.parameters(), clone.parameters()):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        for g, h in zip(opt.param_groups, resumed.param_groups):
            g['lr'] = h['lr']
        opt.step()
        resumed.step()
        scheduler.step()
        for p, q in zip(model.parameters(), clone.parameters()):
            torch.testing.assert_close(p, q)

    def test_backends_shapes_zero_and_missing_gradients(self):
        for backend in module.zeropower_backends:
            for shape in [(4, 5), (12, 4), (4, 12)]:
                p = torch.nn.Parameter(torch.randn(*shape))
                opt = module.MuonP([p], backend=backend)
                # Use eager kernels on CPU, preserving polynomial arithmetic.
                with nullcontext():
                    p.grad = torch.randn_like(p)
                    opt.step()
                    self.assertTrue(torch.isfinite(p).all())
                    p.grad = None
                    before = p.detach().clone()
                    opt.step()
                    torch.testing.assert_close(p, before)
                zero = torch.nn.Parameter(torch.zeros(*shape))
                zopt = module.MuonP([zero], backend=backend)
                zero.grad = torch.zeros_like(zero)
                with nullcontext():
                    zopt.step()
                self.assertTrue(torch.isfinite(zero).all())

    def test_validation_and_empty_matrix_group(self):
        with self.assertRaises(ValueError):
            factory(self.model(), self.args(muon_exponent='2/9', muon_backend='halfpower'))
        with self.assertRaises(ValueError):
            factory(self.model(), self.args(muon_backend='bad'))
        model = torch.nn.LayerNorm(4)
        opt = factory(model, self.args())
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        opt.step()


if __name__ == '__main__':
    unittest.main()
