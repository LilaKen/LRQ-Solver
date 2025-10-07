import paddle
from typing import Literal
import paddle.distributed as dist
from math import ceil
Tensor = paddle.Tensor
def to_tensor(x):
    if isinstance(x, paddle.Tensor):
        return x
    return paddle.to_tensor(x)

def zeros(x):
    return paddle.zeros(x)

def sum(x, axis=None, dtype=None, keepdim=False, name=None):
    return paddle.sum(x, axis, dtype, keepdim, name)

def abs(x):
    return paddle.abs(x)

def load(f, **configs):
    return paddle.load(f, **configs)

def concat(x, axis=0, name=None):
    return paddle.concat(x, axis, name)

def save(obj, path, protocol: Literal[2, 3, 4] = 4, **configs):
    return paddle.save(obj, path, protocol, **configs)

def save_state_dict(state, path):
    dist.save_state_dict(state, path)

def load_state_dict(state, path):
    return dist.load_state_dict(state, path)

def mse_fn(reduction='mean'):
    return paddle.nn.MSELoss(reduction)

def adamw_fn(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, parameters=None, weight_decay=0.01,use_lowprecision_moment=False, lr_ratio=None, apply_decay_param_fun=None, grad_clip=None, lazy_mode=False, multi_precision=False, amsgrad=False, name=None):
    return paddle.optimizer.AdamW(learning_rate, beta1, beta2, epsilon, parameters, weight_decay, use_lowprecision_moment, lr_ratio, apply_decay_param_fun, grad_clip, lazy_mode, multi_precision, amsgrad, name)


class CosineAnnealingLR(paddle.optimizer.lr.CosineAnnealingDecay):
    def __init__(self, learning_rate, T_max, eta_min=0.000001, last_epoch=-1, verbose=False):
        super().__init__(learning_rate, T_max, eta_min, last_epoch, verbose)


def lr_schedular_fn(scheduler_name, learning_rate, T_max, eta_min=0.000001, last_epoch=-1, verbose=False, optimizer = None):
    if scheduler_name == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(learning_rate, T_max, eta_min, last_epoch, verbose)
        optimizer.set_lr_scheduler(scheduler)
    elif scheduler_name is None:
        scheduler = None
    else:
        raise ValueError("Unknown lr scheduler")
    return optimizer, scheduler

def mean(x, axis=None, keepdim=False, name=None):
    return paddle.mean(x, axis=None, keepdim=False, name=None)