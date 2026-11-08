"""Muon^p matrix updates ported from ../muon-p/dual_optimizer.py.

Uses a single standard Optimizer state dict for Trainer/Accelerate checkpoints.
Matrix updates retain upstream momentum, QKV splitting and RMS scaling;
auxiliary parameters use AdamW. DDP computes updates on each replica.
"""
import torch
from . import muon_p_utils as utils

@utils.compile_if_cuda
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7, exponent="1/2", c=None, d=None):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


@utils.compile_if_cuda
def halfpower_via_newtonschulz5(G, steps=6, eps=1e-7, exponent="1/2", c=None, d=None):
    assert len(G.shape) == 2
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    Y = X
    for _ in range(steps):
        if exponent == "1/2":
            Y = utils.poly_half(X, Y, c=c, d=d)
        elif exponent == "1/3":
            Y = utils.poly_third(X, Y, c)
        elif exponent == "1/5":
            Y = utils.poly_fifth(X, Y, c)
        elif exponent == "1/7":
            Y = utils.poly_seventh(X, Y, c)
        elif exponent == "3/5":
            Y = utils.poly_three_fifth(X, Y, c)
        elif exponent == "1/15":
            Y = utils.poly_fifteenth(X, Y, c)
        elif exponent == "13/15":
            Y = utils.poly_thirteen_fifteenth(X, Y, c)
        else:
            raise ValueError(f"Invalid exponent: {exponent}")
    if G.size(0) > G.size(1):
        Y = Y.T
    return Y.to(G.dtype)


def zeropower_via_svd(G, steps=None, exponent="0", c=None, d=None):
    """SVD-based orthogonalization (UV^T). `steps`, `c`, `d` are ignored —
    accepted only to match the signature expected by Muon.step."""
    return utils.zeropower_via_svd(G, steps=steps, exponent=exponent, halfpower_c=c)


def halfpower_via_svd(G, steps=None, exponent="1/3", c=None, d=None):
    """SVD-based fractional-power: U S^exponent V^T. `steps` and `d` are
    ignored; `c` (== halfpower_c) is forwarded but unused by this backend."""
    return utils.halfpower_via_svd(G, exponent=exponent, steps=steps, halfpower_c=c)


zeropower_backends = dict(
    newtonschulz5=zeropower_via_newtonschulz5,
    halfpower=halfpower_via_newtonschulz5,
    zeropower_svd=zeropower_via_svd,
    halfpower_svd=halfpower_via_svd,
)


class MuonP(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-Schulz.

    Internally runs standard SGD-momentum, then performs an
    orthogonalization post-processing step on each 2-D parameter update.
    """

    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 backend='halfpower', backend_steps=6, exponent="1/3", ns_c=None, ns_d=None):
        if backend not in zeropower_backends:
            raise ValueError(f"Unknown MuonP backend: {backend}")
        if backend == 'halfpower' and exponent not in {'1/2', '1/3', '1/5', '1/7', '3/5', '1/15', '13/15'}:
            raise ValueError(f"Unsupported polynomial exponent: {exponent}")
        power = utils._parse_exponent(exponent)
        if not 0 <= power <= 1:
            raise ValueError("MuonP exponent must be between 0 and 1")
        if lr < 0 or not 0 <= momentum < 1 or backend_steps < 1:
            raise ValueError("Invalid MuonP learning rate, momentum, or backend_steps")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        backend=backend, backend_steps=backend_steps, exponent=exponent,
                        ns_c=ns_c, ns_d=ns_d)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if not group.get('use_muon', True):
                for p in group['params']:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if g.is_sparse:
                        raise RuntimeError('MuonP does not support sparse gradients')
                    state = self.state[p]
                    if not state:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    state['step'] += 1
                    beta1, beta2 = group['betas']
                    state['exp_avg'].lerp_(g, 1 - beta1)
                    state['exp_avg_sq'].lerp_(g.square(), 1 - beta2)
                    avg = state['exp_avg'] / (1 - beta1 ** state['step'])
                    var = state['exp_avg_sq'] / (1 - beta2 ** state['step'])
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                    p.addcdiv_(avg, var.sqrt().add_(group['eps']), value=-group['lr'])
                continue
            lr = group['lr']
            momentum = group['momentum']
            zeropower_backend = zeropower_backends[group['backend']]
            exponent = group['exponent']
            ns_c = group['ns_c']
            ns_d = group['ns_d']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse or g.ndim != 2:
                    raise RuntimeError('MuonP requires dense 2-D matrix gradients')
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)
                if g.size(0) == 3 * g.size(1):  # split grouped QKV parameters
                    g_list = []
                    for g1 in g.split(g.size(1)):
                        g_list.append(zeropower_backend(g1, steps=group['backend_steps'], exponent=exponent, c=ns_c, d=ns_d))
                    g = torch.cat(g_list)
                    legacy_scale = g.size(1)**0.5
                elif g.size(1) == 3 * g.size(0):  # split grouped QKV parameters
                    g_list = []
                    for g1 in g.split(g.size(0), dim=1):
                        g_list.append(zeropower_backend(g1, steps=group['backend_steps'], exponent=exponent, c=ns_c, d=ns_d))
                    g = torch.cat(g_list, dim=1)
                    legacy_scale = g.size(0)**0.5
                else:
                    g = zeropower_backend(g, steps=group['backend_steps'], exponent=exponent, c=ns_c, d=ns_d)
                    legacy_scale = max(g.size(0), g.size(1))**0.5
                # Scaling rule:
                #   - newtonschulz5 / zeropower_svd produce an orthogonal output
                #     (singular values ≈ 1), so the legacy `sqrt(max(m,n))` /
                #     `sqrt(n)` correctly gives `update.square().mean() == 1`.
                #   - halfpower / halfpower_svd produce U S^p V^T whose Frobenius
                #     norm shrinks with p, so the legacy scale silently down-weights
                #     the update by a p-dependent factor. We instead measure ||g||_F
                #     directly so update RMS = 1 holds for any exponent.
                if group['backend'] in ('halfpower', 'halfpower_svd'):
                    scale = (g.size(0) * g.size(1))**0.5 / (g.norm() + 1e-7)
                else:
                    scale = legacy_scale
                p.add_(g * scale, alpha=-lr)
        return loss


