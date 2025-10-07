# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import hydra
import numpy as np
import paddle
import pandas as pd
import tensorboardX
from paddle import profiler
from paddle.io import BatchSampler

import ppcfd.utils.op as op
import ppcfd.utils.parallel as parallel
from ppcfd.utils.loss import LpLoss
from ppcfd.utils.metric import R2Score

log = logging.getLogger(__name__)


def set_seed(seed: int = 0):
    paddle.seed(seed)
    np.random.seed(seed)


@dataclass
class AeroDynamicCoefficients:
    c_p_pred: op.Tensor = op.to_tensor(1.0)
    c_f_pred: op.Tensor = op.to_tensor(1.0)
    c_d_pred: op.Tensor = op.to_tensor(1.0)
    c_l_pred: op.Tensor = op.to_tensor(1.0)
    c_p_true: op.Tensor = op.to_tensor(1.0)
    c_f_true: op.Tensor = op.to_tensor(1.0)
    c_l_true: op.Tensor = op.to_tensor(1.0)
    c_d_true: op.Tensor = op.to_tensor(1.0)
    mre_cp: op.Tensor = op.to_tensor(0.0)
    mre_cf: op.Tensor = op.to_tensor(0.0)
    mre_cd: op.Tensor = op.to_tensor(0.0)
    mre_cl: op.Tensor = op.to_tensor(0.0)
    cd_starccm: op.Tensor = op.to_tensor(1.0)
    reference_area: op.Tensor = op.to_tensor(-1.0)
    batch_size: int = 1

    def __post_init__(self):
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, paddle.Tensor):
                vars(self)[attr_name] = paddle.tile(attr_value, [self.batch_size, 1])


@dataclass
class AeroDynamicLoss:
    total_loss: list = field(default_factory=lambda: [])
    l2_p: list = field(default_factory=lambda: [])
    mse_cd: list = field(default_factory=lambda: [])
    mre_cd: list = field(default_factory=lambda: [])
    mre_cp: list = field(default_factory=lambda: [])


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
    physics_loss: list = field(default_factory=lambda: [])
    mse_cd: list = field(default_factory=lambda: [])
    l2_p: list = field(default_factory=lambda: [])
    l2_wss: list = field(default_factory=lambda: [])
    l2_vel: list = field(default_factory=lambda: [])
    mse_p: list = field(default_factory=lambda: [])
    mse_wss: list = field(default_factory=lambda: [])
    mse_vel: list = field(default_factory=lambda: [])
    mre_cp: list = field(default_factory=lambda: [])
    mre_cf: list = field(default_factory=lambda: [])
    mre_cd: list = field(default_factory=lambda: [])
    mre_cl: list = field(default_factory=lambda: [])
    cp_r2_score: float = 0.0
    cf_r2_score: float = 0.0
    cl_r2_score: float = 0.0
    cd_r2_score: float = 0.0


@dataclass
class AeroDynamicPhysicsField:
    physics_field: op.Tensor = op.to_tensor(0.0)
    u: op.Tensor = op.to_tensor(0.0)
    v: op.Tensor = op.to_tensor(0.0)
    w: op.Tensor = op.to_tensor(0.0)
    p: op.Tensor = op.to_tensor(0.0)
    wss: op.Tensor = op.to_tensor(0.0)
    wss_x: op.Tensor = op.to_tensor(0.0)
    cd: op.Tensor = op.to_tensor(0.0)


@dataclass
class StructuralCoefficients:
    mass: op.Tensor = op.to_tensor(0.0)
    safety_factor: op.Tensor = op.to_tensor(0.0)
    max_displacement: op.Tensor = op.to_tensor(0.0)
    contact_pressure: op.Tensor = op.to_tensor(0.0)
    max_mises_stress: op.Tensor = op.to_tensor(0.0)
    max_shear_stress: op.Tensor = op.to_tensor(0.0)
    total_strain_energy: op.Tensor = op.to_tensor(0.0)
    max_principal_stress: op.Tensor = op.to_tensor(0.0)
    max_von_mises_strain: op.Tensor = op.to_tensor(0.0)


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
        self.mse_loss = paddle.nn.MSELoss()
        self.metric = AeroDynamicMetrics()
        self.data_to_dict = data_to_dict
        self.mean_std_dict = mean_std_dict

    def __call__(self, data, output, loss_fn, loss_cd_fn, cal_metric=False):
        config = self.config
        bs = config.test_batch_size if cal_metric else 1
        mse_cd_loss, loss_p, loss_wss, loss_vel = [op.zeros([bs, 1])] * 4
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
            metrics_updated = self.update(*return_list, cx)
            return metrics_updated
        else:
            return return_list

    def update(self, physics_loss, mse_cd_loss, loss_p, loss_wss, loss_vel, cx):
        m = self.metric
        m.physics_loss.append(physics_loss.numpy())
        m.mse_cd.append(mse_cd_loss.numpy())
        m.l2_p.append(loss_p.numpy())
        m.l2_wss.append(loss_wss.numpy())
        m.l2_vel.append(loss_vel.numpy())
        m.mre_cp.append(cx.mre_cp.numpy())
        m.mre_cf.append(cx.mre_cf.numpy())
        m.mre_cd.append(cx.mre_cd.numpy())
        m.mre_cl.append(cx.mre_cl.numpy())
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
        flow_normals = op.zeros(surface_normals.shape)
        flow_normals[..., 0] = x_direction
        const = 2.0 / (mass_density * flow_speed**2 * reference_area)
        const = op.to_tensor(const)
        const = const.reshape([-1, 1, 1])
        direction = op.sum(surface_normals * flow_normals, axis=-1, keepdim=True)
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
            cx.mre_cd = abs(cx.c_d_pred - cx.c_d_true) / abs(cx.c_d_true)
            inputs, targets, others = self.data_to_dict(data)
            cx.reference_area = others["reference_area"]
            return cx
        else:
            inputs, targets, others = self.data_to_dict(data)
            cx.cd_starccm = others.get("Cd", 1.0)
            cx.reference_area = others["reference_area"]
            if "pressure" in self.config.out_keys or "wss" in self.config.out_keys:
                # 2. Prepare Discreted Integral over Car Surface
                cp, cf = self.integral_over_cells(
                    others["reference_area"],
                    targets["normal"],
                    targets["areas"],
                    mass_density,
                    flow_speed,
                    x_direction,
                )

                # 3. Calculate coefficient and MRE
                if "pressure" in self.config.out_keys:
                    cx.c_p_pred = op.sum(cp * pred.p, axis=[1])
                    cx.c_p_true = op.sum(cp * true.p, axis=[1])
                    cx.mre_cp = abs(cx.c_p_pred - cx.c_p_true) / abs(cx.c_p_true)

                if "wss" in self.config.out_keys:
                    cx.c_f_pred = op.sum(cf * pred.wss_x, axis=[1, 2])
                    cx.c_f_true = op.sum(cf * true.wss_x, axis=[1, 2])
                    cx.mre_cf = abs(cx.c_f_pred - cx.c_f_true) / abs(cx.c_f_true)

                if {"pressure", "wss"}.issubset(self.config.out_keys):
                    cx.c_d_pred = cx.c_p_pred + cx.c_f_pred
                    cx.c_d_true = cx.c_p_true + cx.c_f_true
                    cx.mre_cd = abs(cx.c_d_pred - cx.c_d_true) / abs(cx.c_d_true)
                    cx.c_l_pred = cx.c_p_pred  # tofix
                    cx.c_l_true = cx.c_p_true  # tofix
                    cx.mre_cl = abs(
                        op.to_tensor([1e-5] * inputs["centroids"].shape[0])
                    ) / abs(cx.c_l_true)
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
                wss_x_true = wss_true[..., 0:1]
                true.wss = wss_true
                true.wss_x = wss_x_true
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
        if ["Cd"] == config.out_keys:
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
        pred.physics_field = op.concat(pred_list, axis=-1)
        if mode in ["test", "train"]:
            true.physics_field = op.concat(label_list, axis=-1)
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
                            "x": inputs["centroids"][i, :, 0].numpy(),
                            "y": inputs["centroids"][i, :, 1].numpy(),
                            "z": inputs["centroids"][i, :, 2].numpy(),
                            k: targets[k][i, :, 0].numpy(),
                            "output": outputs[i, :, 0].numpy(),  # 假设output是可迭代对象
                        }
                    )
                elif k == "natural_frequency":
                    output_df = pd.DataFrame(
                        {
                            k: targets[k][i, :, 0].numpy(),
                            "output": outputs[i, :, 0].numpy(),  # 假设output是可迭代对象
                        }
                    )
                else:
                    raise NotImplementedError

                # 保存为CSV
                output_df.to_csv(
                    output_dir / f"{others['file_name'][i]}_test_train.csv",
                    index=False,  # 不保存行索引
                )
        if cal_metric:
            return self.structural_metric
        else:
            return loss_list


class Loss_logger:
    def __init__(
        self, output_dir, mode, simulation_type, out_keys, loss_fn, test_batch_size
    ):
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.out_keys = out_keys  # str list
        self.loss_fn = loss_fn  # str
        self.test_batch_size = test_batch_size
        if "Structural" == simulation_type:
            self.metric = StructuralMetrics()
            self.loss = StructuralLoss()
        elif "AeroDynamic" == simulation_type:
            self.metric = AeroDynamicMetrics()
            self.loss = AeroDynamicLoss()
        else:
            raise ValueError("loss_fn must be StructuralMetrics or AeroDynamicMetrics")

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
        df = pd.DataFrame(m.csv_title[1:], columns=m.csv_title[0])
        df.to_csv(self.output_dir / "test.csv", mode="w", index=False)
        if self.mode != "test":
            return

        if isinstance(m, StructuralMetrics):
            log.info(f"Mean Relative L-2 Error [Stress]: {np.mean(m.l2):.2f}")
            return

        cd_pred = paddle.concat([cx.c_d_pred for cx in self.cx_test_list])
        cd_true = paddle.concat([cx.c_d_true for cx in self.cx_test_list])

        # Calculate MSE MAE MAX-AE
        mse = float(paddle.mean((cd_pred - cd_true) ** 2))
        mae = float(paddle.mean(paddle.abs(cd_pred - cd_true)))
        max_ae = float(paddle.max(paddle.abs(cd_pred - cd_true)))

        # MRE
        eps = 1e-8
        mre = (
            float(
                paddle.mean(paddle.abs(cd_pred - cd_true) / (paddle.abs(cd_true) + eps))
            )
            * 100
        )

        # Relative L2
        l2_num = paddle.norm(cd_pred - cd_true, p=2)
        l2_den = paddle.norm(cd_true, p=2)
        rel_l2 = float(l2_num / (l2_den + eps))

        # R²
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

    def record_metric(self, file_name, cx, metric, iter):
        self.cx_test_list.append(cx)
        self.metric = metric
        B = self.test_batch_size
        if isinstance(metric, AeroDynamicMetrics):
            physics_field = self.out_keys
            for i, f in enumerate(file_name):
                mre_cp = cx.mre_cp.numpy()[i][0] * 100
                mre_cf = cx.mre_cf.numpy()[i][0] * 100
                mre_cd = cx.mre_cd.numpy()[i][0] * 100
                mre_cl = cx.mre_cl.numpy()[i][0] * 100
                physics_loss = self.metric.physics_loss[-1][i][0]
                if self.mode == "test":
                    f = f"{f}".ljust(25)
                    log.info(
                        f"Case [{B * iter + i}] {f} {self.loss_fn}: {physics_field} {physics_loss:.2e} "
                        f"MRE: [Cd] {mre_cd:.2f}%"
                        # f"MRE: [Cp] {mre_cp:.2f}%, [Cf] {mre_cf:.2f}%, [Cd] {mre_cd:.2f}%, [Cl] {mre_cl:.2f}%, "
                        # f"L2: [P] {metric.l2_p[-1]:.4f}, [WSS] {metric.l2_wss[-1]:.4f}, [VEL] {metric.l2_vel[-1]:.4f}"
                    )
                self.metric.csv_title.append(
                    [
                        file_name[i],
                        cx.c_p_pred[i].item(),
                        cx.c_f_pred[i].item(),
                        cx.c_p_true[i].item(),
                        cx.c_f_true[i].item(),
                        cx.cd_starccm[i].item(),
                        cx.c_d_pred[i].item(),
                        cx.c_d_true[i].item(),
                        cx.reference_area[i],
                    ]
                )
        elif isinstance(metric, StructuralMetrics):
            if self.mode == "test":
                log.info(
                    f"Case {file_name}\t, Mean L-2 Relative Error [Stress]: {metric.l2[-1]:.2f}"
                )
            self.csv_list.append([file_name, metric.l2[-1]])
        else:
            raise ValueError("loss_fn must be StructuralMetrics or AeroDynamicMetrics")

    def record_tensorboard(
        self,
        ep,
        time_cost,
        lr,
    ):
        loss = self.loss
        m = self.metric
        if isinstance(loss, AeroDynamicLoss) and isinstance(m, AeroDynamicMetrics):
            physics_field_str = self.out_keys[0]
            loss_function_str = self.loss_fn

            # Loss averaged in 1 epoch
            loss_l2_p = np.concat(loss.l2_p[-self.iters :]).mean()
            loss_mse_cd = np.concat(loss.mse_cd[-self.iters :]).mean()
            loss_mre_cp = np.concat(loss.mre_cp[-self.iters :]).mean()
            # Metric averaged in 1 epoch
            physics_loss = np.concat(m.physics_loss).mean()
            metric_mse_cd = np.concat(m.mse_cd).mean()
            metric_mre_cd = np.concat(m.mre_cd).mean() * 100.0
            metric_mre_cp = np.concat(m.mre_cp).mean() * 100.0

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
                f"[Tain]  MSE: [Cd] {loss_mse_cd:.1e} "
                f"[Valid] MSE: [Cd] {metric_mse_cd:.1e}, MRE: [Cd] {metric_mre_cd:.2f}% "
                f"time {(time_cost):.2f}s"
            )
        elif isinstance(m, StructuralMetrics) and isinstance(loss, StructuralLoss):
            self.tensorboard.add_scalar("Train_L2", np.mean(loss.l2), ep)
            self.tensorboard.add_scalar("Test_L2", np.mean(m.l2), ep)
            log.info(
                f"Epoch {ep}  Times {(time_cost):.2f}s, lr:{lr:.1e}, "
                f"[Tain]  Mean Relative L2 loss:{np.mean(m.l2):.4f}   "
                f"[Valid] Mean Relative L2 loss:{np.mean(loss.l2):.4f}"
            )
        else:
            raise ValueError("loss_fn must be StructuralMetrics or AeroDynamicMetrics")


def init_profiler(enable_profiler):
    if enable_profiler:
        prof = profiler.Profiler(
            targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
            timer_only=True,
            scheduler=(3, 7),
            on_trace_ready=profiler.export_chrome_tracing("./log"),
        )
        prof.start()
        return prof
    else:
        return None


def update_profiler(enable_profiler, prof, ep):
    if enable_profiler:
        prof.step()
        if ep == 20:
            prof.stop()
            prof.summary(
                sorted_by=profiler.SortedKeys.GPUTotal,
                op_detail=True,
                thread_sep=False,
                time_unit="ms",
            )
            exit()
    return prof


def load_checkpoint(config, model, optimizer=None):
    """
    load [.pdparams] into model
    load [.pdopt] into optimizer

    Args:
    config (object): yaml configuration files
    model (paddle.nn.Layer): paddle model
    optimizer (paddle.optimizer.Optimizer, optional): paddle optimizer
    """
    assert config.checkpoint is not None, "checkpoint must be given."

    checkpoint_base_path = Path(config.checkpoint)
    ckpt_path = checkpoint_base_path.with_suffix(".pdparams")
    opt_path = checkpoint_base_path.with_suffix(".pdopt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Model checkpoint file (.pdparams) not found at: {ckpt_path}"
        )

    log.info(f"Loading model checkpoint from: {ckpt_path}")
    try:
        pdparams = op.load(str(ckpt_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load model checkpoint from {ckpt_path}: {e}")

    try:
        if isinstance(pdparams, dict) and "model_state_dict" in pdparams:
            model.set_state_dict(pdparams["model_state_dict"])
            log.info("Loaded model state dict from 'model_state_dict' key.")
        else:
            model.set_state_dict(pdparams)
            log.info("Loaded model state dict directly from checkpoint file.")
    except Exception as e:
        raise RuntimeError(f"Failed to set model state dict: {e}")
    if optimizer is not None:
        if os.path.exists(opt_path):
            log.info(f"Loading optimizer checkpoint from: {opt_path}")
            try:
                pdopt = op.load(str(opt_path))
                optimizer.set_state_dict(pdopt)
                log.info("Loaded optimizer state dict.")
            except Exception as e:
                log.info(
                    f"Warning: Failed to load optimizer checkpoint from {opt_path}: {e}. "
                    f"Optimizer state not loaded."
                )
        else:
            log.info(
                f"Optimizer provided, but optimizer checkpoint file (.pdopt) "
                f"not found at: {opt_path}. Optimizer state not loaded."
            )
    else:
        log.info("No optimizer provided. Only model state loaded.")
    log.info("Checkpoint loading completed (ep_start logic removed).")
    return None


def save_ckpt(config, ep, model, optimizer, model_name):
    if ((ep + 1) % 1 == 0) or ((ep + 1) == config.num_epochs):
        if config.enable_mp is True or config.enable_pp is True:
            op.save_state_dict(model.state_dict(), f"{model_name}.pdparams")
            state = optimizer.state_dict()
            if config.lr_schedular is not None:
                state.pop("LR_Scheduler")
            op.save_state_dict(state, f"{model_name}.pdopt")
        else:
            op.save(model.state_dict(), f"{model_name}.pdparams")
            op.save(optimizer.state_dict(), f"{model_name}.pdopt")


@paddle.no_grad()
def test(config, model, test_dataloader, loss_logger, data_loader_fn, ep=None):
    """
    Model Test Function

    Args:
    config (Config): yaml configuration files
    model (nn.Module): model need test
    test_dataloader (DataLoader): dataloader for test data
    loss_logger (LossLogger): loss logger
    data_loader_fn (Callable): data loader function
    ep (int, optional): current epoch. Defaults to None.

    Returns:
    None
    """
    if config.mode == "test":
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

    # parallel
    config.enable_dp = False
    model, _ = parallel.setup_module(config, model)
    test_dataloader = parallel.setup_dataloaders(
        config, test_dataloader, data_loader_fn
    )
    total_inference_time = 0
    t0 = time.time()
    for i, data in enumerate(test_dataloader):
        with paddle.no_grad():
            start_time = time.time()
            outputs = model(data["inputs"])
            end_time = time.time()
            inference_time = end_time - start_time
            total_inference_time += inference_time
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

    loss_logger.record_metric_csv()
    if config.mode == "test":
        log.info(
            f"Test finished. time: {total_inference_time/1154:.2e} seconds, "
            f"max gpu memory = {paddle.device.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB"
        )
        model.train()


def ranking_loss(pred, true, margin=0.01):
    """
    pred: predicted drag in batch
    true_order: true drag in batch
    margin: minimum interval threshold
    """
    # get label index ranking
    true_order = paddle.argsort(true, axis=0)
    loss = 0.0
    n = len(true_order)

    if n == 1:  # counter non-full batch
        return paddle.zeros([1, 1])

    # loop and calculate loss of the pred index ranking
    for i in range(n):
        for j in range(i + 1, n):
            idx_i = true_order[i]  # smaller index
            idx_j = true_order[j]  # larger index

            # rank diff
            diff = pred[idx_j] - pred[idx_i]
            # Hinge Loss
            loss += paddle.clip(margin - diff, min=0.0)

    # averaged ranking loss
    return loss / (n * (n - 1) / 2)


def train(config, model, datamodule, loss_logger):
    """
    Training by PaddlePaddle Deeplearning Framework

    Args:
        config (dict): configuration parameters
        model (paddle.nn.Layer): model to train

    Returns:
        None
    """
    best_val_mse = float("inf")
    model.train()

    optimizer = op.adamw_fn(
        parameters=model.parameters(),
        learning_rate=config.lr,
        weight_decay=config.weight_decay,
    )
    if config.checkpoint is not None:
        log.info(f"loading checkpoint from: {config.checkpoint}")
        ep_start = load_checkpoint(config, model, optimizer) + 1
        learning_rate = config.last_lr
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
    train_sampler = BatchSampler(
        datamodule.train_data,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    train_dataloader = datamodule.train_dataloader(
        num_workers=config.num_workers,
        batch_sampler=train_sampler,
    )
    eval_dataloader = datamodule.val_dataloader(
        batch_size=config.test_batch_size, num_workers=config.num_workers
    )
    train_dataloader = parallel.setup_dataloaders(
        config, train_dataloader, datamodule.train_dataloader
    )
    model, optimizer = parallel.setup_module(config, model, optimizer)

    loss_cd_fn = op.mse_fn("none")
    if config.loss_fn == "MSE":
        loss_fn = op.mse_fn()
    elif config.loss_fn == "L2":
        loss_fn = LpLoss(p=2, enable_dp=config.enable_dp, reduction=False)
    else:
        raise ValueError(f"Invalid loss function. {config.loss_fn}")
    data_to_dict = train_dataloader.dataset.data_to_dict
    mean_std_dict = train_dataloader.dataset.mean_std_dict
    car_loss = Car_Loss(config, train_dataloader.dataset.data_to_dict, mean_std_dict)
    structural_loss = Structural_Loss(config)
    log.info(
        f"iters per epochs = {len(train_dataloader)}, "
        f"num_train = {config.data_module.n_train_num}, "
        f"total bacth size = {config.batch_size * paddle.distributed.get_world_size()}"
    )
    loss_logger.get_iters(len(train_dataloader))

    best_val_mse = float("inf")
    t0 = time.time()
    prof = init_profiler(config.enable_profiler)
    for ep in range(ep_start, config.num_epochs):
        t1 = time.time()
        for _, data in enumerate(train_dataloader):
            outputs = model(data["inputs"])
            if config.enable_profiler:
                paddle.device.synchronize()
            inputs, targets, others = data_to_dict(data)

            if config.simulation_type == "AeroDynamic":
                rank_loss = 2e-3 * ranking_loss(outputs, targets["Cd"])
                physics_loss, mse_cd_loss, loss_p, loss_wss, loss_vel = car_loss(
                    data, outputs, loss_fn, loss_cd_fn
                )
                if config.cd_finetune is True:
                    train_loss = physics_loss + config.cd_loss_weight * mse_cd_loss
                else:
                    train_loss = physics_loss + rank_loss
                cx = data.get("coefficient", AeroDynamicCoefficients())
                loss_logger.loss.l2_p.append(loss_p.numpy())
                loss_logger.loss.mse_cd.append(mse_cd_loss.numpy())
                loss_logger.loss.mre_cp.append(cx.mre_cp.numpy())
            elif config.simulation_type == "Structural":
                loss_list = structural_loss(
                    inputs, outputs, targets, others, loss_fn, loss_cd_fn
                )
                loss_logger.loss.l2.append(loss_list[0])
                physics_loss = loss_list[0]
                train_loss = physics_loss
            else:
                raise ValueError(f"Invalid simulation type. {config.simulation_type}")
            prof = update_profiler(config.enable_profiler, prof, ep)
            optimizer.clear_grad()
            train_loss.backward()
            if config.enable_profiler:
                paddle.device.synchronize()
            optimizer.step()

        if config.lr_schedular is not None:
            scheduler.step()
        test(config, model, eval_dataloader, loss_logger, datamodule.val_dataloader, ep)
        val_mse = np.concat(loss_logger.metric.mse_cd).mean()
        loss_logger.record_tensorboard(
            ep,
            (time.time() - t1),
            optimizer.get_lr(),
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
    log.info(
        f"Training finished. time: {float(time.time() - t0)/3600:.2e} hours, "
        f"Max GPU Memory = {paddle.device.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB"
    )


@hydra.main(
    version_base=None, config_path="./configs", config_name="lqrsolver_drivaerpp.yaml"
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
