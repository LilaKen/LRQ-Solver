import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_dist_env(config):
    if getattr(config, "enable_mp", False) or getattr(config, "enable_pp", False):
        raise NotImplementedError("Torch version currently supports DP/DDP only.")
    if not getattr(config, "enable_dp", False):
        return
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def setup_module(config, model, optimizer=None):
    init_dist_env(config)
    device = get_device()
    if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    model = model.to(device)

    if getattr(config, "enable_dp", False) and get_world_size() > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
        )
    return model, optimizer


def setup_dataloaders(config, dataloader, dataloader_fn):
    if getattr(config, "enable_mp", False) or getattr(config, "enable_pp", False):
        raise NotImplementedError("Torch version currently supports DP/DDP only.")

    if getattr(config, "enable_dp", False) and get_world_size() > 1:
        if getattr(config, "mode", None) == "test":
            indices = list(range(get_rank(), len(dataloader.dataset), get_world_size()))
            return DataLoader(
                Subset(dataloader.dataset, indices),
                num_workers=config.num_workers,
                batch_size=config.test_batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=dataloader.collate_fn,
            )

        sampler = DistributedSampler(
            dataloader.dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            drop_last=True,
        )
        return dataloader_fn(
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            sampler=sampler,
            drop_last=True,
        )
    return dataloader
