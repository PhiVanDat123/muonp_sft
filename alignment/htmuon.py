import torch
import torch.distributed as dist


def _dist_is_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _get_world_size() -> int:
    return dist.get_world_size() if _dist_is_ready() else 1


def _get_rank() -> int:
    return dist.get_rank() if _dist_is_ready() else 0


def msign_generalized_ht_v2(g: torch.Tensor, p: float = 0.25) -> torch.Tensor:
    orig_dtype = g.dtype
    u32, s32, vh32 = torch.linalg.svd(g.to(torch.float32), full_matrices=False)
    sp32 = s32.pow(p)
    out32 = (u32 * sp32.view(*([1] * (u32.ndim - 2)), 1, -1)) @ vh32
    return out32.to(orig_dtype)


def muon_generalized_ht_update_v2(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    power: float = 0.25,
    beta: float = 0.95,
    nesterov: bool = True,
) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    original_shape = update.shape
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = msign_generalized_ht_v2(update, power)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update.reshape(original_shape)


def adam_update(
    grad: torch.Tensor,
    buf1: torch.Tensor,
    buf2: torch.Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> torch.Tensor:
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class HTMuonWithAuxAdam(torch.optim.Optimizer):
    def __init__(self, param_groups, power: float = 0.25):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())
        self.power = power

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        world_size = _get_world_size()
        rank = _get_rank()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                if len(params) == 0:
                    continue
                pad_len = (world_size - len(params) % world_size) % world_size
                params_pad = params + [torch.empty_like(params[-1]) for _ in range(pad_len)]

                for base_i in range(0, len(params_pad), world_size):
                    local_i = base_i + rank
                    if local_i < len(params):
                        p = params[local_i]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_generalized_ht_update_v2(
                            p.grad,
                            state["momentum_buffer"],
                            self.power,
                            beta=group["momentum"],
                        )
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape_as(p), alpha=-group["lr"])

                    if _dist_is_ready():
                        dist.all_gather(
                            params_pad[base_i : base_i + world_size],
                            params_pad[base_i + rank],
                        )
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
