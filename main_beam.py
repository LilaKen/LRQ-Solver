import logging
import time
from pathlib import Path

import hydra
import numpy as np
import paddle
import tensorboardX
from omegaconf import DictConfig
from paddle.io import BatchSampler

import ppcfd.utils.op as op
from ppcfd.utils.parallel import setup_dataloaders
from ppcfd.utils.parallel import setup_module
from visual_beam import visual_beam

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
    paddle.seed(seed)
    np.random.seed(seed)


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

    if cfg.mode == "train":
        train(cfg, model, datamodule, loss_logger)
    elif cfg.mode == "test":
        test_dataloader = datamodule.test_dataloader(
            batch_size=cfg.test_batch_size, num_workers=cfg.num_workers
        )
        model.eval()
        test(cfg, model, test_dataloader, loss_logger)


def train(cfg, model, datamodule, loss_logger):
    loss_fn = paddle.nn.MSELoss()
    loss_fn_mae = paddle.nn.L1Loss()
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    optimizer, scheduler = op.lr_schedular_fn(
        scheduler_name=cfg.lr_schedular,
        learning_rate=cfg.lr,
        T_max=cfg.num_epochs,
        optimizer=optimizer,
    )
    train_sampler = BatchSampler(
        datamodule.train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    train_loader = datamodule.train_dataloader(
        num_workers=cfg.num_workers, batch_sampler=train_sampler, return_list=True
    )
    val_loader = datamodule.val_dataloader(
        batch_size=cfg.batch_size, num_workers=cfg.num_workers, return_list=True
    )
    train_loader = setup_dataloaders(cfg, train_loader, datamodule)
    model, optimizer = setup_module(cfg, model, optimizer)
    best_val_loss = float(1.0)

    for ep in range(cfg.num_epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        for step, batch_data in enumerate(train_loader):
            branch, trunk, label, mask, _ = batch_data
            pred = model((trunk, branch))
            loss = loss_fn(pred, label)
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if cfg.lr_schedular is not None:
                scheduler.step()

        t0 = time.time()

        if ep % cfg.val_freq == 0:
            val_loss = 0.0
            val_loss_mae = 0.0
            with paddle.no_grad():
                for step, batch_data in enumerate(val_loader):
                    branch, trunk, label, mask, _ = batch_data
                    pred = model((trunk, branch))
                    val_loss += loss_fn(pred, label).item()
                    val_loss_mae += loss_fn_mae(pred, label).item()
            val_loss /= len(val_loader)
            val_loss_mae /= len(val_loader)
        else:
            val_loss = best_val_loss

        t1 = time.time()
        train_loss /= len(train_loader)
        loss_logger.record_train_loss(train_loss)
        loss_logger.record_val_loss(val_loss)
        train_time = t0 - start_time
        val_time = t1 - t0
        current_lr = optimizer.get_lr()
        loss_logger.record_metric(
            ep, train_loss, val_loss, val_loss_mae, current_lr, train_time, val_time
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep + 1
            paddle.save(
                model.state_dict(), str(Path(cfg.output_dir) / "best_model.pdparams")
            )

        if (ep + 1) % 100 == 0 or (ep + 1) == cfg.num_epochs:
            paddle.save(
                model.state_dict(),
                str(Path(cfg.output_dir) / f"model_ep{ep + 1}.pdparams"),
            )

    log.info(
        f"Training finished. time: {float(time.time() - t0)/3600:.2e} h, "
        f"max gpu memory = {paddle.device.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB"
    )


def test(cfg, model, test_dataloader, loss_logger):
    assert cfg.checkpoint is not None
    log.info(f"Loading model weights from: {cfg.checkpoint}")
    model_state_dict = paddle.load(cfg.checkpoint)
    model.set_state_dict(model_state_dict)

    model.eval()
    test_loss = []
    start_time = time.time()
    loss_fn = paddle.nn.L1Loss(reduction="none")
    ds = test_dataloader.dataset
    # labels = np.load("./data/3d_beam/stress_targets.npz")

    for batch_idx, batch_data in enumerate(test_dataloader):
        branch, trunk, label, mask, case_id = batch_data
        mask = mask.astype("bool")
        pred = model((trunk, branch))
        pred_denorm = ds.inverse_transform(pred)
        label_denorm = ds.inverse_transform(label)
        mae_loss = loss_fn(pred_denorm, label_denorm)
        mae_loss = paddle.mean(mae_loss[mask], axis=0)
        test_loss.append(mae_loss)
        # log.info(f"Test Case ID {case_id.item()}")
        # label_i = labels[f"Job_{case_id.item()}"].reshape(-1)
        # pred_i = pred_denorm[0,:label_i.shape[0],0].numpy()
        # visual_beam(label_i, pred_i, case_id.item())
        log.info(
            f"Batch={batch_idx + 1}, "
            f"Batch Mean MAE: {mae_loss.mean().item() / 1e6:.2f} MPa"
        )

    time_cost = time.time() - start_time
    test_loss = paddle.to_tensor(test_loss)
    avg_mae = paddle.mean(test_loss).item()
    max_gpu_cache = paddle.device.cuda.max_memory_allocated() / 1024**3

    log.info(f"Test completed! Statistics:")
    log.info(f"  MAE: {avg_mae / 1e6:.4f} MPa")
    log.info(f"  Time Cost: {time_cost:.2f} s")
    log.info(f"  Max GPU Memory: {max_gpu_cache:.2f} GB")
    return avg_mae


if __name__ == "__main__":
    main()
