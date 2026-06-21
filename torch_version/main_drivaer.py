import logging
import json
import os
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import tensorboardX
import torch

import ppcfd.utils.op as op
import ppcfd.utils.parallel as parallel
from ppcfd.utils.loss import LpLoss
from ppcfd.utils.metric import R2Score
from ppcfd.utils.parallel import get_device

log = logging.getLogger(__name__)


def tensor_default(value):
    return torch.tensor([[value]], dtype=torch.float32)


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def tensor_item(value, index=0):
    if isinstance(value, torch.Tensor):
        return value.detach().reshape(-1)[index].cpu().item()
    return np.asarray(value).reshape(-1)[index].item()


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


def cuda_memory_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def reset_cuda_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def reduce_scalar(value, op="max"):
    value = float(value)
    if parallel.get_world_size() == 1:
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=get_device())
    reduce_op = {
        "max": torch.distributed.ReduceOp.MAX,
        "sum": torch.distributed.ReduceOp.SUM,
    }[op]
    torch.distributed.all_reduce(tensor, op=reduce_op)
    return float(tensor.item())


def barrier():
    if parallel.get_world_size() > 1:
        torch.distributed.barrier()


def batch_size_from_inputs(inputs):
    if isinstance(inputs, torch.Tensor):
        return int(inputs.shape[0])
    if isinstance(inputs, (tuple, list)):
        return batch_size_from_inputs(inputs[0])
    if isinstance(inputs, dict):
        first_value = next(iter(inputs.values()))
        return batch_size_from_inputs(first_value)
    raise TypeError(f"Unsupported input type for batch size: {type(inputs)!r}")


def write_runtime_metrics(config, section, metrics):
    if not getattr(config, "record_runtime_metrics", True):
        return
    if parallel.get_rank() != 0:
        return

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "runtime_metrics.json"
    payload = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            log.warning(f"Could not parse existing runtime metrics file: {path}")

    payload.setdefault(
        "run",
        {
            "mode": config.mode,
            "model_name": config.model_name,
            "num_points": config.data_module.num_points,
            "world_size": parallel.get_world_size(),
            "batch_size_per_rank": config.batch_size,
            "test_batch_size_per_rank": config.test_batch_size,
        },
    )
    payload[section] = metrics
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def profile_forward_flops(config, model, dataloader, device, profile_name):
    if not getattr(config, "record_flops", True):
        return None

    barrier()
    stats = None
    if parallel.get_rank() == 0:
        try:
            data = next(iter(dataloader))
            data = to_device(data, device)
            target_model = model.module if hasattr(model, "module") else model
            was_training = target_model.training
            target_model.eval()
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
                torch.cuda.synchronize()

            with torch.no_grad():
                with torch.profiler.profile(
                    activities=activities,
                    with_flops=True,
                    profile_memory=False,
                ) as prof:
                    _ = target_model(data["inputs"])
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

            if was_training:
                target_model.train()

            forward_flops = sum(
                getattr(event, "flops", 0) or 0 for event in prof.key_averages()
            )
            batch_size = batch_size_from_inputs(data["inputs"])
            stats = {
                "profile_name": profile_name,
                "profile_batch_size": batch_size,
                "forward_flops_per_batch": int(forward_flops),
                "forward_flops_per_sample": float(forward_flops / max(batch_size, 1)),
                "flops_note": (
                    "Counted with torch.profiler with_flops=True; only operators "
                    "supported by PyTorch profiler are included."
                ),
            }
            log.info(
                f"{profile_name} FLOPs profile | "
                f"forward/batch={forward_flops:.4e}, "
                f"forward/sample={stats['forward_flops_per_sample']:.4e}"
            )
        except Exception as exc:
            stats = {
                "profile_name": profile_name,
                "flops_error": str(exc),
            }
            log.warning(f"Failed to profile {profile_name} FLOPs: {exc}")

    barrier()
    reset_cuda_peak_memory()
    return stats


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@dataclass
class AeroDynamicCoefficients:
    c_p_pred: op.Tensor = field(default_factory=lambda: tensor_default(1.0))
    c_f_pred: op.Tensor = field(default_factory=lambda: tensor_default(1.0))
    c_d_pred: op.Tensor = field(default_factory=lambda: tensor_default(1.0))
    c_l_pred: op.Tensor = field(default_factory=lambda: tensor_default(1.0))
    c_p_true: op.Tensor = field(default_factory=lambda: tensor_default(1.0))
    c_f_true: op.Tensor = field(default_factory=lambda: tensor_default(1.0))
    c_l_true: op.Tensor = field(default_factory=lambda: tensor_default(1.0))
    c_d_true: op.Tensor = field(default_factory=lambda: tensor_default(1.0))
    mre_cp: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    mre_cf: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    mre_cd: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    mre_cl: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    cd_starccm: op.Tensor = field(default_factory=lambda: tensor_default(1.0))
    reference_area: object = field(default_factory=lambda: tensor_default(-1.0))
    batch_size: int = 1

    def __post_init__(self):
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, torch.Tensor) and attr_value.numel() == 1:
                vars(self)[attr_name] = attr_value.reshape(1, 1).repeat(
                    self.batch_size, 1
                )


@dataclass
class AeroDynamicLoss:
    total_loss: list = field(default_factory=list)
    l2_p: list = field(default_factory=list)
    mse_cd: list = field(default_factory=list)
    mre_cd: list = field(default_factory=list)
    mre_cp: list = field(default_factory=list)


@dataclass
class AeroDynamicMetrics:
    csv_title: list = field(
        default_factory=lambda: [
            [
                "file_name",
                "cp pred",
                "cf pred",
                "cp true",
                "cf true",
                "cd starccm+",
                "cd pred",
                "cd true",
                "frontal area",
            ],
        ]
    )
    physics_loss: list = field(default_factory=list)
    mse_cd: list = field(default_factory=list)
    l2_p: list = field(default_factory=list)
    l2_wss: list = field(default_factory=list)
    l2_vel: list = field(default_factory=list)
    mse_p: list = field(default_factory=list)
    mse_wss: list = field(default_factory=list)
    mse_vel: list = field(default_factory=list)
    mre_cp: list = field(default_factory=list)
    mre_cf: list = field(default_factory=list)
    mre_cd: list = field(default_factory=list)
    mre_cl: list = field(default_factory=list)
    cp_r2_score: float = 0.0
    cf_r2_score: float = 0.0
    cl_r2_score: float = 0.0
    cd_r2_score: float = 0.0


@dataclass
class AeroDynamicPhysicsField:
    physics_field: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    u: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    v: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    w: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    p: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    wss: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    wss_x: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    cd: op.Tensor = field(default_factory=lambda: tensor_default(0.0))


@dataclass
class StructuralCoefficients:
    mass: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    safety_factor: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    max_displacement: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    contact_pressure: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    max_mises_stress: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    max_shear_stress: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    total_strain_energy: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    max_principal_stress: op.Tensor = field(default_factory=lambda: tensor_default(0.0))
    max_von_mises_strain: op.Tensor = field(default_factory=lambda: tensor_default(0.0))


@dataclass
class StructuralMetrics:
    csv_title: list = field(default_factory=lambda: ["file_name", "mean L-2 error"])
    l2: list = field(default_factory=list)


@dataclass
class StructuralLoss:
    l2: list = field(default_factory=list)


class Car_Loss:
    def __init__(self, config, data_to_dict, mean_std_dict=None):
        self.config = config
        self.cx_list = []
        self.mse_loss = torch.nn.MSELoss()
        self.metric = AeroDynamicMetrics()
        self.data_to_dict = data_to_dict
        self.mean_std_dict = mean_std_dict or {}

    def __call__(self, data, output, loss_fn, loss_cd_fn, cal_metric=False):
        config = self.config
        batch_size = output.shape[0]
        zero = torch.zeros(batch_size, 1, dtype=output.dtype, device=output.device)
        mse_cd_loss, loss_p, loss_wss, loss_vel = [zero] * 4
        pred, true = self.denormalize(data, output, config.mode)
        cx = self.calculate_coefficient(
            data,
            pred=pred,
            true=true,
            mass_density=config.mass_density,
            flow_speed=config.flow_speed,
        )
        data["coefficient"] = cx

        if self.config.mode == "inference":
            return cx, pred

        if "pressure" in config.out_keys:
            loss_p = loss_fn(pred.p, true.p)
        if "wss" in config.out_keys:
            loss_wss = loss_fn(pred.wss, true.wss)
        if "vel" in config.out_keys:
            loss_vel = loss_fn(pred.u, true.u)
        if "Cd" in self.config.out_keys or config.cd_finetune:
            mse_cd_loss = loss_cd_fn(cx.c_d_pred, cx.c_d_true)

        physics_loss = loss_fn(true.physics_field, pred.physics_field)
        return_list = [
            physics_loss,
            mse_cd_loss,
            loss_p,
            loss_wss,
            loss_vel,
        ]
        if cal_metric:
            return self.update(*return_list, cx)
        return return_list

    def update(self, physics_loss, mse_cd_loss, loss_p, loss_wss, loss_vel, cx):
        m = self.metric
        m.physics_loss.append(to_numpy(physics_loss))
        m.mse_cd.append(to_numpy(mse_cd_loss))
        m.l2_p.append(to_numpy(loss_p))
        m.l2_wss.append(to_numpy(loss_wss))
        m.l2_vel.append(to_numpy(loss_vel))
        m.mre_cp.append(to_numpy(cx.mre_cp))
        m.mre_cf.append(to_numpy(cx.mre_cf))
        m.mre_cd.append(to_numpy(cx.mre_cd))
        m.mre_cl.append(to_numpy(cx.mre_cl))
        return m

    def integral_over_cells(
        self,
        reference_area,
        surface_normals,
        areas,
        mass_density,
        flow_speed,
        x_direction=1,
    ):
        flow_normals = torch.zeros_like(surface_normals)
        flow_normals[..., 0] = x_direction
        const = 2.0 / (mass_density * flow_speed**2 * reference_area)
        const = torch.as_tensor(
            const, dtype=surface_normals.dtype, device=surface_normals.device
        ).reshape(-1, 1, 1)
        direction = torch.sum(surface_normals * flow_normals, dim=-1, keepdim=True)
        c_p = const * direction * areas
        c_f = (const * flow_normals * areas)[..., 0:1]
        return c_p, c_f

    def calculate_coefficient(
        self,
        data,
        pred,
        true,
        mass_density,
        flow_speed,
        x_direction=1,
    ):
        cx = AeroDynamicCoefficients(batch_size=data["inputs"][0].shape[0])
        if "Cd" in self.config.out_keys:
            cx.c_d_pred = pred.cd
            cx.c_d_true = true.cd
            cx.mre_cd = torch.abs(cx.c_d_pred - cx.c_d_true) / (
                torch.abs(cx.c_d_true) + 1e-8
            )
            inputs, targets, others = self.data_to_dict(data)
            cx.reference_area = others["reference_area"]
            return cx

        inputs, targets, others = self.data_to_dict(data)
        cx.cd_starccm = others.get("Cd", tensor_default(1.0))
        cx.reference_area = others["reference_area"]
        if "pressure" in self.config.out_keys or "wss" in self.config.out_keys:
            cp, cf = self.integral_over_cells(
                others["reference_area"],
                targets["normal"],
                targets["areas"],
                mass_density,
                flow_speed,
                x_direction,
            )

            if "pressure" in self.config.out_keys:
                cx.c_p_pred = torch.sum(cp * pred.p, dim=1)
                cx.c_p_true = torch.sum(cp * true.p, dim=1)
                cx.mre_cp = torch.abs(cx.c_p_pred - cx.c_p_true) / (
                    torch.abs(cx.c_p_true) + 1e-8
                )

            if "wss" in self.config.out_keys:
                cx.c_f_pred = torch.sum(cf * pred.wss_x, dim=(1, 2))
                cx.c_f_true = torch.sum(cf * true.wss_x, dim=(1, 2))
                cx.mre_cf = torch.abs(cx.c_f_pred - cx.c_f_true) / (
                    torch.abs(cx.c_f_true) + 1e-8
                )

            if {"pressure", "wss"}.issubset(self.config.out_keys):
                cx.c_d_pred = cx.c_p_pred + cx.c_f_pred
                cx.c_d_true = cx.c_p_true + cx.c_f_true
                cx.mre_cd = torch.abs(cx.c_d_pred - cx.c_d_true) / (
                    torch.abs(cx.c_d_true) + 1e-8
                )
        return cx

    def denormalize(self, data, outputs, mode, eps=1e-6):
        _, targets, _ = self.data_to_dict(data)
        config = self.config
        mean_std_dict = self.mean_std_dict
        channels = 0
        true, pred = AeroDynamicPhysicsField(), AeroDynamicPhysicsField()
        label_list, pred_list = [], []
        assert len(config.out_keys) != 0, "config.out_keys must be not empty"
        if "pressure" in config.out_keys:
            mean = mean_std_dict["p_mean"]
            std = mean_std_dict["p_std"]
            index = config.out_keys.index("pressure")
            n = config.out_channels[index]
            p_pred = outputs[..., channels : channels + n] * (std + eps) + mean
            pred_list.append(p_pred)
            pred.p = p_pred
            if mode in ["test", "train"]:
                p_true = targets["pressure"]
                label_list.append(p_true)
                true.p = p_true
            channels += n
        if "wss" in config.out_keys:
            mean = mean_std_dict["wss_mean"]
            std = mean_std_dict["wss_std"]
            index = config.out_keys.index("wss")
            n = config.out_channels[index]
            wss_pred = outputs[..., channels : channels + n] * (std + eps) + mean
            wss_x_pred = wss_pred[..., 0:1]
            pred_list.append(wss_pred)
            pred.wss = wss_pred
            pred.wss_x = wss_x_pred
            if mode in ["test", "train"]:
                wss_true = targets["wss"]
                label_list.append(wss_true)
                true.wss = wss_true
                true.wss_x = wss_true[..., 0:1]
            channels += n
        if "vel" in config.out_keys:
            mean = mean_std_dict.get("v_mean", [1.0])[0]
            std = mean_std_dict.get("v_std", [0.0])[0]
            index = config.out_keys.index("vel")
            n = config.out_channels[index]
            vel_pred = outputs[..., channels : channels + n] * mean + std
            pred_list.append(vel_pred)
            pred.u = vel_pred
            if mode in ["test", "train"]:
                vel_true = targets[..., channels : channels + n]
                label_list.append(vel_true)
                true.u = vel_true
            channels += n
        if ["Cd"] == list(config.out_keys):
            index = config.out_keys.index("Cd")
            n = config.out_channels[index]
            label = targets["Cd"]
            mean = mean_std_dict.get("cd_mean", 0.0)
            std = mean_std_dict.get("cd_std", 1.0)
            cd_pred = outputs[..., channels : channels + n] * std + mean
            cd_true = label[..., channels : channels + n]
            pred_list.append(cd_pred)
            label_list.append(cd_true)
            true.cd = cd_true
            pred.cd = cd_pred
        pred.physics_field = torch.cat(pred_list, dim=-1)
        if mode in ["test", "train"]:
            true.physics_field = torch.cat(label_list, dim=-1)
        return pred, true


class Structural_Loss:
    def __init__(self, config):
        self.config = config
        self.structural_loss = StructuralLoss()
        self.structural_metric = StructuralMetrics()

    def __call__(
        self, inputs, outputs, targets, others, loss_fn, loss_cd_fn, cal_metric=False
    ):
        targets["coefficient"] = StructuralCoefficients()
        loss_list = []
        for k in self.config.out_keys:
            _targets = targets[k]
            l2_loss = loss_fn(outputs, _targets)
            loss_list.append(l2_loss)
            if cal_metric:
                self.structural_metric.l2.append(l2_loss.item())
            else:
                self.structural_loss.l2.append(l2_loss.item())
            output_dir = Path(self.config.output_dir) / "test_case"
            output_dir.mkdir(parents=True, exist_ok=True)
            batch_size = inputs["centroids"].shape[0]
            for i in range(batch_size):
                if k == "stress":
                    output_df = pd.DataFrame(
                        {
                            "x": to_numpy(inputs["centroids"][i, :, 0]),
                            "y": to_numpy(inputs["centroids"][i, :, 1]),
                            "z": to_numpy(inputs["centroids"][i, :, 2]),
                            k: to_numpy(targets[k][i, :, 0]),
                            "output": to_numpy(outputs[i, :, 0]),
                        }
                    )
                elif k == "natural_frequency":
                    output_df = pd.DataFrame(
                        {
                            k: to_numpy(targets[k][i, :, 0]),
                            "output": to_numpy(outputs[i, :, 0]),
                        }
                    )
                else:
                    raise NotImplementedError

                output_df.to_csv(
                    output_dir / f"{others['file_name'][i]}_test_train.csv",
                    index=False,
                )
        if cal_metric:
            return self.structural_metric
        return loss_list


class Loss_logger:
    def __init__(
        self, output_dir, mode, simulation_type, out_keys, loss_fn, test_batch_size
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.out_keys = out_keys
        self.loss_fn = loss_fn
        self.test_batch_size = test_batch_size
        if "Structural" == simulation_type:
            self.metric = StructuralMetrics()
            self.loss = StructuralLoss()
        elif "AeroDynamic" == simulation_type:
            self.metric = AeroDynamicMetrics()
            self.loss = AeroDynamicLoss()
        else:
            raise ValueError("simulation_type must be Structural or AeroDynamic")

        tensorboard = tensorboardX.SummaryWriter(
            os.path.join(output_dir, "tensorboard")
        )
        log.info(f"Working directory : {os.getcwd()}")
        log.info(f"Output directory  : {output_dir}")
        self.tensorboard = tensorboard

        self.csv_list = [self.metric.csv_title]
        self.cx_test_list = []

    def get_iters(self, iters):
        self.iters = iters

    def record_metric_csv(self):
        m = self.metric
        rows = m.csv_title[1:]
        cd_pred = None
        cd_true = None

        if isinstance(m, AeroDynamicMetrics) and self.mode == "test":
            cd_pred = self._cat_test_tensor("c_d_pred")
            cd_true = self._cat_test_tensor("c_d_true")

        if self.mode == "test" and parallel.get_world_size() > 1:
            payload = {"rows": rows, "cd_pred": cd_pred, "cd_true": cd_true}
            gathered = [None for _ in range(parallel.get_world_size())]
            torch.distributed.all_gather_object(gathered, payload)
            if parallel.get_rank() != 0:
                return None

            rows = [row for rank_payload in gathered for row in rank_payload["rows"]]
            if isinstance(m, AeroDynamicMetrics):
                cd_pred = torch.cat(
                    [rank_payload["cd_pred"] for rank_payload in gathered], dim=0
                )
                cd_true = torch.cat(
                    [rank_payload["cd_true"] for rank_payload in gathered], dim=0
                )

        if parallel.get_rank() == 0:
            df = pd.DataFrame(rows, columns=m.csv_title[0])
            df.to_csv(self.output_dir / "test.csv", mode="w", index=False)
        if self.mode != "test":
            return None
        if parallel.get_rank() != 0:
            return None

        if isinstance(m, StructuralMetrics):
            mean_l2 = float(np.mean(m.l2))
            log.info(f"Mean Relative L-2 Error [Stress]: {mean_l2:.2f}")
            return {"mean_relative_l2": mean_l2}

        if cd_pred is None:
            cd_pred = self._cat_test_tensor("c_d_pred")
        if cd_true is None:
            cd_true = self._cat_test_tensor("c_d_true")

        mse = float(torch.mean((cd_pred - cd_true) ** 2))
        mae = float(torch.mean(torch.abs(cd_pred - cd_true)))
        max_ae = float(torch.max(torch.abs(cd_pred - cd_true)))

        eps = 1e-8
        mre = float(torch.mean(torch.abs(cd_pred - cd_true) / (torch.abs(cd_true) + eps))) * 100
        rel_l2 = float(torch.linalg.vector_norm(cd_pred - cd_true, ord=2) / (torch.linalg.vector_norm(cd_true, ord=2) + eps))

        r2_metric = R2Score()
        r2 = float(r2_metric({"Cd": cd_pred}, {"Cd": cd_true})["Cd"])

        m.cd_mse = mse
        m.cd_mae = mae
        m.cd_max_ae = max_ae
        m.cd_mre = mre
        m.cd_rel_l2 = rel_l2
        m.cd_r2_score = r2
        case_number = len(cd_true)
        log.info(
            f"Cd summary over {case_number} cases | "
            f"MSE={mse:.4e}  MAE={mae:.4e}  MaxAE={max_ae:.4e}  "
            f"MRE={mre:.2f}%  RelL2={rel_l2:.4f}  R²={r2:.4f}"
        )
        return {
            "case_number": case_number,
            "cd_mse": mse,
            "cd_mae": mae,
            "cd_max_ae": max_ae,
            "cd_mre_percent": mre,
            "cd_rel_l2": rel_l2,
            "cd_r2_score": r2,
        }

    def _cat_test_tensor(self, name):
        tensors = [
            getattr(cx, name).detach().cpu().reshape(-1) for cx in self.cx_test_list
        ]
        if not tensors:
            return torch.empty(0)
        return torch.cat(tensors, dim=0)

    def record_metric(self, file_name, cx, metric, iter):
        self.cx_test_list.append(cx)
        self.metric = metric
        batch_size = len(file_name)
        if isinstance(metric, AeroDynamicMetrics):
            physics_field = self.out_keys
            for i, f in enumerate(file_name):
                mre_cp = tensor_item(cx.mre_cp, i) * 100
                mre_cf = tensor_item(cx.mre_cf, i) * 100
                mre_cd = tensor_item(cx.mre_cd, i) * 100
                mre_cl = tensor_item(cx.mre_cl, i) * 100
                physics_loss = np.asarray(self.metric.physics_loss[-1]).reshape(-1)[i]
                if self.mode == "test":
                    f_log = f"{f}".ljust(25)
                    log.info(
                        f"Case [{self.test_batch_size * iter + i}] {f_log} "
                        f"{self.loss_fn}: {physics_field} {physics_loss:.2e} "
                        f"MRE: [Cd] {mre_cd:.2f}%"
                    )
                reference_area = (
                    cx.reference_area[i]
                    if isinstance(cx.reference_area, list)
                    else tensor_item(cx.reference_area, i)
                )
                self.metric.csv_title.append(
                    [
                        file_name[i],
                        tensor_item(cx.c_p_pred, i),
                        tensor_item(cx.c_f_pred, i),
                        tensor_item(cx.c_p_true, i),
                        tensor_item(cx.c_f_true, i),
                        tensor_item(cx.cd_starccm, i),
                        tensor_item(cx.c_d_pred, i),
                        tensor_item(cx.c_d_true, i),
                        reference_area,
                    ]
                )
        elif isinstance(metric, StructuralMetrics):
            if self.mode == "test":
                log.info(
                    f"Case {file_name}\t, Mean L-2 Relative Error [Stress]: {metric.l2[-1]:.2f}"
                )
            self.csv_list.append([file_name, metric.l2[-1]])
        else:
            raise ValueError("metric must be StructuralMetrics or AeroDynamicMetrics")

    def record_tensorboard(self, ep, time_cost, lr):
        loss = self.loss
        m = self.metric
        if isinstance(loss, AeroDynamicLoss) and isinstance(m, AeroDynamicMetrics):
            physics_field_str = self.out_keys[0]
            loss_function_str = self.loss_fn

            loss_l2_p = np.concatenate(loss.l2_p[-self.iters :]).mean()
            loss_mse_cd = np.concatenate(loss.mse_cd[-self.iters :]).mean()
            loss_mre_cp = np.concatenate(loss.mre_cp[-self.iters :]).mean()
            physics_loss = np.concatenate(m.physics_loss).mean()
            metric_mse_cd = np.concatenate(m.mse_cd).mean()
            metric_mre_cd = np.concatenate(m.mre_cd).mean() * 100.0
            metric_mre_cp = np.concatenate(m.mre_cp).mean() * 100.0

            self.tensorboard.add_scalar(
                f"Train_{physics_field_str}_{loss_function_str}", loss_l2_p, ep
            )
            self.tensorboard.add_scalar(
                f"Valid_{physics_field_str}_{loss_function_str}", physics_loss, ep
            )
            self.tensorboard.add_scalar("Train_Cd_MSE", loss_mse_cd, ep)
            self.tensorboard.add_scalar("Valid_Cd_MSE", metric_mse_cd, ep)
            log.info(
                f"Epoch {ep}, lr: {lr:.1e}, "
                f"[Train] MSE: [Cd] {loss_mse_cd:.1e} "
                f"[Valid] MSE: [Cd] {metric_mse_cd:.1e}, MRE: [Cd] {metric_mre_cd:.2f}% "
                f"time {(time_cost):.2f}s"
            )
        elif isinstance(m, StructuralMetrics) and isinstance(loss, StructuralLoss):
            self.tensorboard.add_scalar("Train_L2", np.mean(loss.l2), ep)
            self.tensorboard.add_scalar("Test_L2", np.mean(m.l2), ep)
            log.info(
                f"Epoch {ep} Times {(time_cost):.2f}s, lr:{lr:.1e}, "
                f"[Train] Mean Relative L2 loss:{np.mean(m.l2):.4f} "
                f"[Valid] Mean Relative L2 loss:{np.mean(loss.l2):.4f}"
            )
        else:
            raise ValueError("loss/metric type mismatch")


def load_checkpoint(config, model, optimizer=None):
    assert config.checkpoint is not None, "checkpoint must be given."

    checkpoint_path = Path(config.checkpoint)
    if checkpoint_path.suffix == ".pdparams":
        raise RuntimeError(
            "This Torch version cannot load Paddle .pdparams checkpoints directly. "
            "Train a Torch checkpoint or convert the weights to .pt/.pth first."
        )
    if checkpoint_path.suffix == "":
        checkpoint_path = checkpoint_path.with_suffix(".pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint file not found at: {checkpoint_path}")

    log.info(f"Loading model checkpoint from: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    target_model = model.module if hasattr(model, "module") else model
    epoch = -1
    if isinstance(state, dict) and "model_state_dict" in state:
        target_model.load_state_dict(state["model_state_dict"])
        epoch = state.get("epoch", -1)
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
    else:
        target_model.load_state_dict(state)
    log.info("Checkpoint loading completed.")
    return epoch


def save_ckpt(config, ep, model, optimizer, model_name):
    if ((ep + 1) % 1 == 0) or ((ep + 1) == config.num_epochs):
        torch.save(
            {
                "epoch": ep,
                "model_state_dict": module_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"{model_name}.pt",
        )


@torch.no_grad()
def test(config, model, test_dataloader, loss_logger, data_loader_fn, ep=None):
    if config.mode == "test":
        model, _ = parallel.setup_module(config, model)
        load_checkpoint(config, model)
        full_test = True
    else:
        full_test = ((ep + 1) % config.val_freq == 0) or ((ep + 1) == config.num_epochs)
    model.eval()
    loss_cd_fn = op.mse_fn("none")
    if config.loss_fn == "MSE":
        loss_fn = op.mse_fn("none")
    elif config.loss_fn == "L2":
        loss_fn = LpLoss(p=2, enable_dp=config.enable_dp, reduction=False)
    else:
        raise ValueError(f"Invalid loss function. {config.loss_fn}")
    test_ds = test_dataloader.dataset
    if config.simulation_type == "AeroDynamic":
        simulation_loss = Car_Loss(config, test_ds.data_to_dict, test_ds.mean_std_dict)
    elif config.simulation_type == "Structural":
        simulation_loss = Structural_Loss(config)
    else:
        raise ValueError(f"Invalid simulation type. {config.simulation_type}")

    device = get_device()
    test_dataloader = parallel.setup_dataloaders(
        config, test_dataloader, data_loader_fn
    )
    flops_stats = None
    if config.mode == "test":
        flops_stats = profile_forward_flops(
            config, model, test_dataloader, device, "test_forward"
        )

    total_inference_time = 0
    num_batches = 0
    num_samples = 0
    for i, data in enumerate(test_dataloader):
        data = to_device(data, device)
        batch_size = batch_size_from_inputs(data["inputs"])
        start_time = time.time()
        outputs = model(data["inputs"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_inference_time += time.time() - start_time
        num_batches += 1
        num_samples += batch_size
        metric = simulation_loss(data, outputs, loss_fn, loss_cd_fn, cal_metric=True)
        inputs, targets, others = test_ds.data_to_dict(data)
        loss_logger.record_metric(
            others["file_name"],
            data.get("coefficient", AeroDynamicCoefficients()),
            metric,
            i,
        )
        if (full_test is False) and (i > 5):
            break

    metric_summary = loss_logger.record_metric_csv()
    if config.mode == "test":
        local_memory_gb = cuda_memory_gb()
        global_total_time = reduce_scalar(total_inference_time, "sum")
        global_wall_time = reduce_scalar(total_inference_time, "max")
        global_batches = int(reduce_scalar(num_batches, "sum"))
        global_samples = int(reduce_scalar(num_samples, "sum"))
        global_memory_gb = reduce_scalar(local_memory_gb, "max")
        avg_batch_time = global_total_time / max(global_batches, 1)
        avg_sample_time = global_total_time / max(global_samples, 1)
        throughput = global_samples / max(global_wall_time, 1e-12)

        runtime_summary = {
            "world_size": parallel.get_world_size(),
            "local_batches": num_batches,
            "global_batches": global_batches,
            "global_samples": global_samples,
            "inference_total_device_time_sec_sum": global_total_time,
            "inference_wall_time_sec_max_rank": global_wall_time,
            "inference_avg_time_per_batch_sec": avg_batch_time,
            "inference_avg_time_per_sample_sec": avg_sample_time,
            "inference_throughput_samples_per_sec": throughput,
            "peak_gpu_memory_gb_max_rank": global_memory_gb,
            "memory_note": (
                "Peak memory is torch.cuda.max_memory_allocated(), max over ranks."
            ),
        }
        if flops_stats:
            runtime_summary.update(flops_stats)
            if "forward_flops_per_sample" in flops_stats:
                runtime_summary["test_forward_flops_total"] = int(
                    flops_stats["forward_flops_per_sample"] * global_samples
                )
        if metric_summary:
            runtime_summary.update(metric_summary)
        write_runtime_metrics(config, "test", runtime_summary)

        if parallel.get_rank() == 0:
            log.info(
                "Runtime summary | "
                f"cases={global_samples}, "
                f"wall_time={global_wall_time:.4e}s, "
                f"avg_batch={avg_batch_time:.4e}s, "
                f"avg_sample={avg_sample_time:.4e}s, "
                f"throughput={throughput:.2f} samples/s, "
                f"max_gpu_memory={global_memory_gb:.2f} GB"
            )
        denom = max(num_batches, 1)
        log.info(
            f"Test finished. time: {total_inference_time / denom:.2e} seconds, "
            f"max gpu memory = {cuda_memory_gb():.2f} GB"
        )
        model.train()


def ranking_loss(pred, true, margin=0.01):
    pred = pred.reshape(-1)
    true = true.reshape(-1)
    true_order = torch.argsort(true, dim=0)
    n = len(true_order)
    if n == 1:
        return pred.new_zeros(())

    loss = pred.new_zeros(())
    for i in range(n):
        for j in range(i + 1, n):
            idx_i = true_order[i]
            idx_j = true_order[j]
            diff = pred[idx_j] - pred[idx_i]
            loss = loss + torch.clamp(margin - diff, min=0.0)
    return loss / (n * (n - 1) / 2)


def train(config, model, datamodule, loss_logger):
    best_val_mse = float("inf")
    model, _ = parallel.setup_module(config, model)
    model.train()

    optimizer = op.adamw_fn(
        parameters=model.parameters(),
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
    )
    if config.checkpoint is not None:
        log.info(f"loading checkpoint from: {config.checkpoint}")
        ep_start = load_checkpoint(config, model, optimizer) + 1
        learning_rate = config.last_lr or config.lr
    else:
        ep_start = 0
        learning_rate = config.lr
    optimizer, scheduler = op.lr_schedular_fn(
        scheduler_name=config.lr_schedular,
        learning_rate=learning_rate,
        T_max=config.num_epochs,
        optimizer=optimizer,
        last_epoch=ep_start - 1,
    )
    train_dataloader = datamodule.train_dataloader(
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.num_workers,
    )
    eval_dataloader = datamodule.val_dataloader(
        batch_size=config.test_batch_size, num_workers=config.num_workers
    )
    train_dataloader = parallel.setup_dataloaders(
        config, train_dataloader, datamodule.train_dataloader
    )

    loss_cd_fn = op.mse_fn("none")
    if config.loss_fn == "MSE":
        loss_fn = op.mse_fn()
    elif config.loss_fn == "L2":
        loss_fn = LpLoss(p=2, enable_dp=config.enable_dp, reduction=False)
    else:
        raise ValueError(f"Invalid loss function. {config.loss_fn}")
    data_to_dict = train_dataloader.dataset.data_to_dict
    mean_std_dict = train_dataloader.dataset.mean_std_dict
    car_loss = Car_Loss(config, data_to_dict, mean_std_dict)
    structural_loss = Structural_Loss(config)
    log.info(
        f"iters per epochs = {len(train_dataloader)}, "
        f"num_train = {config.data_module.n_train_num}, "
        f"total batch size = {config.batch_size * parallel.get_world_size()}"
    )
    loss_logger.get_iters(len(train_dataloader))

    device = get_device()
    flops_stats = profile_forward_flops(
        config, model, train_dataloader, device, "train_forward"
    )
    t0 = time.time()
    reset_cuda_peak_memory()
    for ep in range(ep_start, config.num_epochs):
        t1 = time.time()
        for data in train_dataloader:
            data = to_device(data, device)
            outputs = model(data["inputs"])
            inputs, targets, others = data_to_dict(data)

            if config.simulation_type == "AeroDynamic":
                rank_loss = 2e-3 * ranking_loss(outputs, targets["Cd"])
                physics_loss, mse_cd_loss, loss_p, loss_wss, loss_vel = car_loss(
                    data, outputs, loss_fn, loss_cd_fn
                )
                if config.cd_finetune is True:
                    train_loss = physics_loss + config.cd_loss_weight * mse_cd_loss.mean()
                else:
                    train_loss = physics_loss + rank_loss
                cx = data.get("coefficient", AeroDynamicCoefficients())
                loss_logger.loss.l2_p.append(to_numpy(loss_p))
                loss_logger.loss.mse_cd.append(to_numpy(mse_cd_loss))
                loss_logger.loss.mre_cp.append(to_numpy(cx.mre_cp))
            elif config.simulation_type == "Structural":
                loss_list = structural_loss(
                    inputs, outputs, targets, others, loss_fn, loss_cd_fn
                )
                loss_logger.loss.l2.append(loss_list[0].item())
                physics_loss = loss_list[0]
                train_loss = physics_loss
            else:
                raise ValueError(f"Invalid simulation type. {config.simulation_type}")

            optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            optimizer.step()

        if config.lr_schedular is not None:
            scheduler.step()
        test(config, model, eval_dataloader, loss_logger, datamodule.val_dataloader, ep)
        val_mse = np.concatenate(loss_logger.metric.mse_cd).mean()
        loss_logger.record_tensorboard(
            ep,
            (time.time() - t1),
            optimizer.param_groups[0]["lr"],
        )
        if best_val_mse > val_mse:
            best_val_mse = val_mse
            save_ckpt(
                config,
                ep,
                model,
                optimizer,
                f"{config.output_dir}/best_{config.model_name}",
            )
        save_ckpt(
            config,
            ep,
            model,
            optimizer,
            f"{config.output_dir}/{config.model_name}_{ep}",
        )
    train_time_sec = time.time() - t0
    global_train_time_sec = reduce_scalar(train_time_sec, "max")
    global_memory_gb = reduce_scalar(cuda_memory_gb(), "max")
    epochs_ran = config.num_epochs - ep_start
    train_summary = {
        "world_size": parallel.get_world_size(),
        "epochs_ran": epochs_ran,
        "steps_per_epoch_per_rank": len(train_dataloader),
        "train_time_sec": global_train_time_sec,
        "train_time_hours": global_train_time_sec / 3600,
        "train_avg_epoch_time_sec": global_train_time_sec / max(epochs_ran, 1),
        "peak_gpu_memory_gb_max_rank": global_memory_gb,
        "memory_note": (
            "Peak memory is torch.cuda.max_memory_allocated(), max over ranks."
        ),
    }
    if flops_stats:
        train_summary.update(flops_stats)
        if "forward_flops_per_batch" in flops_stats:
            train_steps_per_rank = len(train_dataloader) * max(epochs_ran, 0)
            world_size = parallel.get_world_size()
            forward_flops_per_batch = flops_stats["forward_flops_per_batch"]
            train_summary["train_forward_flops_per_step_global"] = int(
                forward_flops_per_batch * world_size
            )
            train_summary["train_forward_backward_flops_per_step_global_estimate"] = int(
                forward_flops_per_batch * world_size * 3
            )
            train_summary["train_forward_backward_flops_total_estimate"] = int(
                forward_flops_per_batch * world_size * train_steps_per_rank * 3
            )
            train_summary["train_flops_estimate_note"] = (
                "Training FLOPs total estimates forward+backward as 3x one "
                "profiled forward pass per rank."
            )
    write_runtime_metrics(config, "train", train_summary)
    log.info(
        f"Training finished. time: {global_train_time_sec / 3600:.2e} hours, "
        f"Max GPU Memory = {global_memory_gb:.2f} GB"
    )


@hydra.main(
    version_base=None, config_path="./configs", config_name="lrqsolver_drivaerpp.yaml"
)
def main(config):
    loss_logger = Loss_logger(
        config.output_dir,
        config.mode,
        config.simulation_type,
        config.out_keys,
        config.loss_fn,
        config.test_batch_size,
    )
    set_seed(config.seed)
    datamodule = hydra.utils.instantiate(config.data_module)
    model = hydra.utils.instantiate(config.model)

    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Total number of parameters: {total_params / 1e6:.2f} M")

    if config.mode == "train":
        train(config, model, datamodule, loss_logger)
    elif config.mode == "test":
        test_dataloader = datamodule.test_dataloader(
            batch_size=config.test_batch_size, num_workers=config.num_workers
        )
        model.eval()
        test(config, model, test_dataloader, loss_logger, datamodule.test_dataloader)


if __name__ == "__main__":
    main()
