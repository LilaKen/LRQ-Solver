import logging
import time
from pathlib import Path

import hydra
import numpy as np
import tensorboardX
import torch
from omegaconf import DictConfig

import ppcfd.utils.op as op
from ppcfd.utils.parallel import get_device
from ppcfd.utils.parallel import setup_dataloaders
from ppcfd.utils.parallel import setup_module

log = logging.getLogger(__name__)


class Loss_logger:
    def __init__(self, output_dir, mode, simulation_type, out_keys, loss_fn):
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.simulation_type = simulation_type
        self.out_keys = out_keys
        self.loss_fn = loss_fn
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard = tensorboardX.SummaryWriter(
            log_dir=self.output_dir / "tensorboard"
        )
        self.train_losses = []
        self.val_losses = []
        self.log_file = self.output_dir / f"{mode}.log"
        log.info(f"Log file will be saved at: {str(self.log_file)}")
        logging.basicConfig(
            filename=self.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

    def record_train_loss(self, loss):
        self.train_losses.append(loss)
        self.tensorboard.add_scalar("Train_Loss", loss, len(self.train_losses))

    def record_val_loss(self, loss):
        self.val_losses.append(loss)
        self.tensorboard.add_scalar("Validation_Loss", loss, len(self.val_losses))

    def record_metric(
        self,
        epoch,
        train_loss,
        val_loss,
        val_loss_mae,
        lr,
        train_time,
        val_time,
        **metrics,
    ):
        logging.info(
            f"Epoch {epoch} | Train Loss: {train_loss:.4e} | "
            f"Val Loss: MSE {val_loss:.4e}, MAE {val_loss_mae:.2e} | "
            f"LR: {lr:.1e} | Train Time: {train_time:.2f}s, Val Time: {val_time:.2f}s"
        )
        self.tensorboard.add_scalar("Learning_Rate", lr, epoch)
        self.tensorboard.add_scalar("Epoch_Time", train_time + val_time, epoch)

        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value:.4e}")
            self.tensorboard.add_scalar(metric_name, metric_value, epoch)


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    if isinstance(data, dict):
        return {key: to_device(value, device) for key, value in data.items()}
    if isinstance(data, tuple):
        return tuple(to_device(value, device) for value in data)
    if isinstance(data, list):
        return [to_device(value, device) for value in data]
    return data


def module_state_dict(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def load_torch_checkpoint(checkpoint, model, device):
    if str(checkpoint).endswith(".pdparams"):
        raise RuntimeError(
            "This Torch version cannot load Paddle .pdparams checkpoints directly. "
            "Train a Torch checkpoint or convert the weights to .pt/.pth first."
        )
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)


def cuda_memory_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


@hydra.main(
    version_base=None, config_path="./configs", config_name="lrqsolver_beam.yaml"
)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    loss_logger = Loss_logger(
        cfg.output_dir, cfg.mode, cfg.simulation_type, cfg.out_keys, cfg.loss_fn
    )
    datamodule = hydra.utils.instantiate(cfg.data_module)
    model = hydra.utils.instantiate(cfg.model)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Total number of parameters: {total_params / 1e6:.2f} M")

    if cfg.mode == "train":
        train(cfg, model, datamodule, loss_logger)
    elif cfg.mode == "test":
        test_dataloader = datamodule.test_dataloader(
            batch_size=cfg.test_batch_size, num_workers=cfg.num_workers
        )
        model.eval()
        test(cfg, model, test_dataloader, loss_logger)


def train(cfg, model, datamodule, loss_logger):
    device = get_device()
    model, _ = setup_module(cfg, model, None)
    loss_fn = torch.nn.MSELoss()
    loss_fn_mae = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    optimizer, scheduler = op.lr_schedular_fn(
        scheduler_name=cfg.lr_schedular,
        learning_rate=cfg.lr,
        T_max=cfg.num_epochs,
        optimizer=optimizer,
    )
    train_loader = datamodule.train_dataloader(
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=cfg.num_workers,
    )
    val_loader = datamodule.val_dataloader(
        batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )
    train_loader = setup_dataloaders(cfg, train_loader, datamodule.train_dataloader)
    best_val_loss = float(1.0)

    for ep in range(cfg.num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        for batch_data in train_loader:
            branch, trunk, label, mask, _ = to_device(batch_data, device)
            pred = model((trunk, branch))
            loss = loss_fn(pred, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if cfg.lr_schedular is not None:
                scheduler.step()

        t0 = time.time()

        if ep % cfg.val_freq == 0:
            val_loss = 0.0
            val_loss_mae = 0.0
            with torch.no_grad():
                for batch_data in val_loader:
                    branch, trunk, label, mask, _ = to_device(batch_data, device)
                    pred = model((trunk, branch))
                    val_loss += loss_fn(pred, label).item()
                    val_loss_mae += loss_fn_mae(pred, label).item()
            val_loss /= len(val_loader)
            val_loss_mae /= len(val_loader)
        else:
            val_loss = best_val_loss
            val_loss_mae = 0.0

        t1 = time.time()
        train_loss /= len(train_loader)
        loss_logger.record_train_loss(train_loss)
        loss_logger.record_val_loss(val_loss)
        train_time = t0 - start_time
        val_time = t1 - t0
        current_lr = optimizer.param_groups[0]["lr"]
        loss_logger.record_metric(
            ep, train_loss, val_loss, val_loss_mae, current_lr, train_time, val_time
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                module_state_dict(model), str(Path(cfg.output_dir) / "best_model.pt")
            )

        if (ep + 1) % 100 == 0 or (ep + 1) == cfg.num_epochs:
            torch.save(
                module_state_dict(model),
                str(Path(cfg.output_dir) / f"model_ep{ep + 1}.pt"),
            )

    log.info(
        f"Training finished. time: {float(time.time() - t0) / 3600:.2e} h, "
        f"max gpu memory = {cuda_memory_gb():.2f} GB"
    )


def test(cfg, model, test_dataloader, loss_logger):
    assert cfg.checkpoint is not None
    device = get_device()
    model, _ = setup_module(cfg, model, None)
    log.info(f"Loading model weights from: {cfg.checkpoint}")
    load_torch_checkpoint(cfg.checkpoint, model, device)

    model.eval()
    test_loss = []
    start_time = time.time()
    loss_fn = torch.nn.L1Loss(reduction="none")
    ds = test_dataloader.dataset

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            branch, trunk, label, mask, case_id = to_device(batch_data, device)
            mask = mask.bool()
            pred = model((trunk, branch))
            pred_denorm = ds.inverse_transform(pred)
            label_denorm = ds.inverse_transform(label)
            mae_loss = loss_fn(pred_denorm, label_denorm)
            mae_loss = mae_loss[mask].mean()
            test_loss.append(mae_loss)
            log.info(
                f"Batch={batch_idx + 1}, "
                f"Batch Mean MAE: {mae_loss.mean().item() / 1e6:.2f} MPa"
            )

    time_cost = time.time() - start_time
    avg_mae = torch.stack(test_loss).mean().item()
    max_gpu_cache = cuda_memory_gb()

    log.info("Test completed! Statistics:")
    log.info(f"  MAE: {avg_mae / 1e6:.4f} MPa")
    log.info(f"  Time Cost: {time_cost:.2f} s")
    log.info(f"  Max GPU Memory: {max_gpu_cache:.2f} GB")
    return avg_mae


if __name__ == "__main__":
    main()
