from typing import Literal

import torch

Tensor = torch.Tensor


def to_tensor(x, dtype=None, device=None):
    if isinstance(x, torch.Tensor):
        tensor = x
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device=device)
        return tensor
    return torch.as_tensor(x, dtype=dtype, device=device)


def zeros(x, dtype=torch.float32, device=None):
    return torch.zeros(x, dtype=dtype, device=device)


def sum(x, axis=None, dtype=None, keepdim=False, name=None):
    return torch.sum(x, dim=axis, dtype=dtype, keepdim=keepdim)


def abs(x):
    return torch.abs(x)


def load(f, **configs):
    return torch.load(f, map_location=configs.pop("map_location", "cpu"), **configs)


def concat(x, axis=0, name=None):
    return torch.cat(list(x), dim=axis)


def save(obj, path, protocol: Literal[2, 3, 4, 5] = 4, **configs):
    return torch.save(obj, path, pickle_protocol=protocol, **configs)


def save_state_dict(state, path):
    return save(state, path)


def load_state_dict(state, path):
    return load(path)


def mse_fn(reduction="mean"):
    return torch.nn.MSELoss(reduction=reduction)


def adamw_fn(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    parameters=None,
    weight_decay=0.01,
    use_lowprecision_moment=False,
    lr_ratio=None,
    apply_decay_param_fun=None,
    grad_clip=None,
    lazy_mode=False,
    multi_precision=False,
    amsgrad=False,
    name=None,
):
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
    )


def lr_schedular_fn(
    scheduler_name,
    learning_rate,
    T_max,
    eta_min=0.000001,
    last_epoch=-1,
    verbose=False,
    optimizer=None,
):
    if scheduler_name == "CosineAnnealingLR":
        if last_epoch != -1:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch
        )
    elif scheduler_name is None:
        scheduler = None
    else:
        raise ValueError("Unknown lr scheduler")
    return optimizer, scheduler


def mean(x, axis=None, keepdim=False, name=None):
    return torch.mean(x, dim=axis, keepdim=keepdim)
