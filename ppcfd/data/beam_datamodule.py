import os
import pickle
import time
from typing import Callable
from typing import Optional

import numpy as np
import paddle
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import MinMaxScaler

from ppcfd.data.base_datamodule import BaseDataModule


class BeamDataset(paddle.io.Dataset):
    def __init__(
        self,
        input_param_path: str,
        output_npz_path: str,
        ids: np.ndarray,
        clip_max: float = 4.5e8,
        scaler_dir: str = "scalers",
        transform: Optional[Callable] = None,
        gen_scaler: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        np.random.seed(seed)
        self.scaler_dir = scaler_dir
        self.gen_scaler = gen_scaler

        self.design_params = np.load(input_param_path)  # (N_total, 4)
        out_npz = np.load(output_npz_path)
        self.output_pos = out_npz["a"]  # (N_total, N_pts, 4)
        self.output_vals = out_npz["b"]  # (N_total, N_pts, 1)
        N_total, N_pts, _ = self.output_pos.shape

        if "mask" in out_npz:
            self.mask = (
                out_npz["mask"].astype(np.bool_).reshape((N_total, N_pts, 1))
            )  # (N_total, N_pts)
        else:
            self.mask = np.ones((N_total, N_pts, 1), dtype=np.bool_)

        self.ids = ids
        self.transform = transform
        t1 = time.time()
        self._clip_stress(clip_max)
        if gen_scaler:
            self._normalize()
        else:
            self._load_and_transform_scalers()
        t2 = time.time()
        print(
            f"Time cost of Data Pre-processing: {(t2 - t1):.2f}s",
        )

    def _clip_stress(self, clip_max):
        self.output_vals[..., 0] = np.clip(self.output_vals[..., 0], 0.0, clip_max)

    def _normalize(self):
        for name, arr in [
            ("xyz", self.output_pos),
            ("stress", self.output_vals),
            ("param", self.design_params),
        ]:
            scaler = MinMaxScaler()
            flat = (
                arr.reshape(-1, arr.shape[-1])
                if name != "stress"
                else arr.reshape(-1, 1)
            )
            scaler.fit(flat)
            setattr(self, f"{name}_scaler", scaler)
            new_arr = scaler.transform(flat).reshape(arr.shape)
            if name == "xyz":
                self.output_pos = paddle.to_tensor(new_arr.astype("float32"))
            elif name == "stress":
                self.output_vals = paddle.to_tensor(new_arr.astype("float32"))
            elif name == "param":
                self.design_params = paddle.to_tensor(new_arr.astype("float32"))
            os.makedirs(self.scaler_dir, exist_ok=False)

            scaler_path = os.path.join(self.scaler_dir, f"{name}_scaler.pkl")
            with open(scaler_path, "wb") as f:
                pickle.dump(scaler, f)

    def inverse_transform(self, tensor):
        """
        tensor: [B, N, D] or [N, D]
        scaler: sklearn's MinMaxScaler
        """
        scaler = self.stress_scaler
        original_shape = tensor.shape
        tensor_np = tensor.numpy().reshape(-1, scaler.n_features_in_)
        denorm_np = scaler.inverse_transform(tensor_np)
        denorm_tensor = paddle.to_tensor(denorm_np).reshape(original_shape)
        return denorm_tensor

    def _load_and_transform_scalers(self):
        for name, arr in [
            ("xyz", self.output_pos),
            ("stress", self.output_vals),
            ("param", self.design_params),
        ]:
            scaler_path = os.path.join(self.scaler_dir, f"{name}_scaler.pkl")

            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    scaler = pickle.load(f)
                setattr(self, f"{name}_scaler", scaler)
            else:
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

            flat = (
                arr.reshape(-1, arr.shape[-1])
                if name != "stress"
                else arr.reshape(-1, 1)
            )
            new_arr = scaler.transform(flat).reshape(arr.shape)

            if name == "xyz":
                self.output_pos = paddle.to_tensor(new_arr.astype("float32"))
            elif name == "stress":
                self.output_vals = paddle.to_tensor(new_arr.astype("float32"))
            elif name == "param":
                self.design_params = paddle.to_tensor(new_arr.astype("float32"))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        case_id = self.ids[idx]
        branch = self.design_params[case_id]
        trunk = self.output_pos[case_id]
        label = self.output_vals[case_id]
        mask = paddle.to_tensor(self.mask[case_id])

        if self.transform is not None:
            branch, trunk, label = self.transform(branch, trunk, label)
        return branch, trunk, label, mask, case_id


class BeamDataModule(BaseDataModule):
    """
    7.5 : 0.5 : 2 for train : val : test
    """

    def __init__(
        self,
        input_param_path: str,
        output_npz_path: str,
        clip_max: float = 4.5e8,
        scaler_dir: str = "",
        transform: Optional[Callable] = None,
        gen_scaler: bool = False,
    ):
        super().__init__()
        self.input_param_path = input_param_path
        self.output_npz_path = output_npz_path
        self.clip_max = clip_max
        self.scaler_dir = scaler_dir
        self.transform = transform
        self._make_splits()
        self.train_data = BeamDataset(
            input_param_path,
            output_npz_path,
            self.train_ids,
            clip_max,
            scaler_dir,
            transform,
            gen_scaler,
        )
        self.val_data = BeamDataset(
            input_param_path,
            output_npz_path,
            self.val_ids,
            clip_max,
            scaler_dir,
            transform,
            gen_scaler,
        )
        self.test_data = BeamDataset(
            input_param_path,
            output_npz_path,
            self.test_ids,
            clip_max,
            scaler_dir,
            transform,
            gen_scaler,
        )

    def _make_splits(self):
        n_total_pos = np.load(self.output_npz_path)["a"].shape[0]
        n_total_para = np.load(self.input_param_path).shape[0]
        n_total = min(n_total_pos, n_total_para)
        design_params = np.load(self.input_param_path)[:n_total]
        para_scaler = MinMaxScaler()
        para_scaler.fit(design_params)
        design_params = para_scaler.transform(design_params)
        shifted = design_params - design_params[0]
        dist = np.linalg.norm(shifted[:, :-1], axis=-1)
        order = np.argsort(dist)
        n_train = int(n_total * 0.75)
        n_val = int(n_total * 0.05)
        self.train_ids = order[:n_train]
        self.val_ids = order[n_train : n_train + n_val]
        self.test_ids = order[n_train + n_val :]


def generate_example1_resampled_npz(
    base_dir: str = "../",
    n_case: int = 3000,
    n_repeat: int = 1,
    pts_list=None,
    save_dir: str = "./",
):
    if pts_list is None:
        pts_list = [
            "original",
            250,
            1000,
            2000,
            5000,
            10000,
            25000,
            60000,
        ]

    stress = np.load(os.path.join(base_dir, "stress_targets.npz"))
    xyzd = np.load(os.path.join(base_dir, "xyzs.npz"))
    os.makedirs(save_dir, exist_ok=True)

    for rpt in range(n_repeat):
        for n_pts in pts_list:
            if n_pts == "original":
                max_points = 0

                for k in range(n_case):
                    name = f"Job_{k}"
                    my_xyzd = xyzd[name]
                    max_points = max(max_points, my_xyzd.shape[0])

                output_pos = np.zeros((n_case, max_points, 4), dtype=np.float32)
                output_vals = np.zeros((n_case, max_points, 1), dtype=np.float32)
                mask = np.zeros((n_case, max_points), dtype=np.bool_)

                for k in range(n_case):
                    name = f"Job_{k}"
                    my_xyzd, my_stress = xyzd[name], stress[name]
                    n_actual = my_xyzd.shape[0]
                    output_pos[k, :n_actual] = my_xyzd.copy()
                    output_vals[k, :n_actual] = my_stress.copy()
                    mask[k, :n_actual] = True

                npz_path = os.path.join(save_dir, f"Outputs_rpt{rpt + 3}_original.npz")
                np.savez_compressed(npz_path, a=output_pos, b=output_vals, mask=mask)
                print(f"mask shape: {mask.shape}")
            else:
                output_pos = np.zeros((n_case, n_pts, 4), dtype=np.float32)
                output_vals = np.zeros((n_case, n_pts, 1), dtype=np.float32)
                mask = np.ones((n_case, n_pts), dtype=np.bool_)

                for k in range(n_case):
                    name = f"Job_{k}"
                    my_xyzd, my_stress = xyzd[name], stress[name]
                    n_total = my_xyzd.shape[0]

                    if n_total <= n_pts:
                        if n_total == n_pts:
                            idx = np.arange(n_total)
                        else:
                            all_indices = np.arange(n_total)
                            remaining_count = n_pts - n_total
                            additional_indices = np.random.choice(
                                n_total, remaining_count, replace=True
                            )
                            idx = np.concatenate([all_indices, additional_indices])

                        mask[k, :] = False
                        seen = set()
                        for i, index in enumerate(idx):
                            if index not in seen:
                                mask[k, i] = True
                                seen.add(index)
                    else:
                        idx = np.random.choice(n_total, n_pts, replace=False)
                        mask[k, :] = True

                    output_pos[k] = my_xyzd[idx].copy()
                    output_vals[k] = my_stress[idx].copy()

                npz_path = os.path.join(save_dir, f"Outputs_rpt{rpt + 3}_N{n_pts}.npz")
                np.savez_compressed(npz_path, a=output_pos, b=output_vals, mask=mask)

            print(f"Saved {npz_path}")


if __name__ == "__main__":
    np.random.seed(42)
    std = time.time()
    generate_example1_resampled_npz(
        base_dir="../data/3d_beam/",
        save_dir="./processed_data",
    )
    print("cache time cost: ", time.time() - std, "seconds")

# python -m ppcfd.data.beam_datamodule
