import logging

import torch
import torch.distributed as dist

from .muon import (
    MuonWithAuxAdam,
    MuonWithAuxAdam_sigmoid,
    SingleDeviceMuonWithAuxAdam,
    SingleDeviceMuonWithAuxAdam_sigmoid,
)
from .htmuon import HTMuonWithAuxAdam
from .muon_p import MuonP


logger = logging.getLogger(__name__)

MUON_OPTIMIZERS = {
    "muon",
    "soren",
    "htmuon",
    "muon_p",
    "muon-p",
}


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _split_muon_params(model: torch.nn.Module):
    muon_params = []
    adam_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lower_name = name.lower()
        use_muon = (
            param.ndim >= 2
            and "embed" not in lower_name
            and "lm_head" not in lower_name
            and "norm" not in lower_name
        )
        if use_muon:
            muon_params.append(param)
        else:
            adam_params.append(param)

    return muon_params, adam_params


def create_custom_optimizer(model: torch.nn.Module, training_args):
    custom_optimizer = getattr(training_args, "custom_optimizer", None)
    if custom_optimizer is None:
        return None

    custom_optimizer = custom_optimizer.lower()
    if custom_optimizer not in MUON_OPTIMIZERS:
        raise ValueError(f"Unknown custom optimizer: {custom_optimizer}")

    muon_params, adam_params = _split_muon_params(model)
    if custom_optimizer in {"muon_p", "muon-p"}:
        # The upstream spectral kernels operate on matrices only.
        adam_params.extend(p for p in muon_params if p.ndim != 2)
        muon_params = [p for p in muon_params if p.ndim == 2]
    param_groups = []

    if adam_params:
        param_groups.append(
            {
                "params": adam_params,
                "lr": training_args.learning_rate,
                "betas": (training_args.adam_beta1, training_args.adam_beta2),
                "eps": training_args.adam_epsilon,
                "weight_decay": training_args.weight_decay,
                "use_muon": False,
            }
        )
    if muon_params:
        param_groups.append(
            {
                "params": muon_params,
                "lr": training_args.learning_rate,
                "momentum": getattr(training_args, "muon_momentum", 0.95),
                "weight_decay": training_args.weight_decay,
                "use_muon": True,
            }
        )

    if not param_groups:
        raise ValueError("No trainable parameters found for custom optimizer.")

    logger.info(
        "Using custom optimizer %s with %d muon tensors, %d aux-adam tensors, lr=%s, weight_decay=%s",
        custom_optimizer,
        len(muon_params),
        len(adam_params),
        training_args.learning_rate,
        training_args.weight_decay,
    )

    if custom_optimizer in {"muon_p", "muon-p"}:
        factor = getattr(training_args, "muon_lr_factor", 0.3)
        if factor < 0:
            raise ValueError("muon_lr_factor must be nonnegative")
        for group in param_groups:
            if group["use_muon"]:
                group["lr"] *= factor
                # Upstream Muon^p has no matrix weight decay.
                group["weight_decay"] = 0.0
        return MuonP(
            param_groups,
            lr=training_args.learning_rate,
            momentum=getattr(training_args, "muon_momentum", 0.95),
            backend=getattr(training_args, "muon_backend", "halfpower"),
            backend_steps=getattr(training_args, "muon_backend_steps", 6),
            exponent=getattr(training_args, "muon_exponent", "1/3"),
            ns_c=getattr(training_args, "halfpower_c", None),
            ns_d=getattr(training_args, "halfpower_d", None),
        )

    if custom_optimizer == "soren":
        optimizer_cls = MuonWithAuxAdam_sigmoid if _is_distributed() else SingleDeviceMuonWithAuxAdam_sigmoid
        return optimizer_cls(param_groups)
    if custom_optimizer == "muon":
        optimizer_cls = MuonWithAuxAdam if _is_distributed() else SingleDeviceMuonWithAuxAdam
        return optimizer_cls(param_groups)

    htmuon_power = getattr(training_args, "htmuon_power", 0.25)
    logger.info("Using HTMuon power=%s", htmuon_power)
    return HTMuonWithAuxAdam(param_groups, power=htmuon_power)
