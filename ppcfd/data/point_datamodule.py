import json
import paddle
import logging
import numpy as np
from pathlib import Path
from ppcfd.data.base_datamodule import BaseDataModule

log = logging.getLogger(__name__)


class PointDataset(paddle.io.Dataset):
    def __init__(
        self,
        files: list[str],
        mean_std_file: str,
        num_cases: int = -1,
        downsample_size: int = -1,
    ):
        super().__init__()
        self.files = self._get_files(files)[:num_cases]
        self.mean_std_dict_npy = self._load_mean_std(mean_std_file)
        self.mean_std_dict = {
            k: paddle.to_tensor(self.mean_std_dict_npy[k])
            for k in self.mean_std_dict_npy.keys()
        }
        self.downsample_size = downsample_size
        self.inputs_key = ["centroids"]
        self.targets_key = ["normal", "areas", "idx", "pressure", "wss"]
        self.others_key = ["file_path", "num_points", "air_density", "flow_dir", "flow_speed", "reference_area", "c_p", "c_wss"]

    def _get_files(self, split_file: str) -> list[str]:
        files = []
        if split_file and Path(split_file).exists():
            with open(split_file, "r") as f:
                files = json.load(f)
        return files


    def _load_mean_std(self, mean_std_file: str):
        mean_std_dict = {}
        with open(mean_std_file, "r") as f:
            json_dict = json.load(f)

        for k, v in json_dict.items():
            for kk, vv in v.items():
                mean_std_dict[k + "_" + kk] = np.array(vv, dtype=np.float32)
        mean_std_dict["p_mean"] = mean_std_dict["pressure_mean"]
        mean_std_dict["p_std"] = mean_std_dict["pressure_std"]
        return mean_std_dict

    def __getitem__(self, idx: int):
        inputs: dict[str, np.array] = {}
        targets: dict[str, np.array] = {}
        data_file = self.files[idx]


        if self.downsample_size > 0:
            npy_data = np.load(data_file, mmap_mode="r")
            sampled_indices = np.sort(
                np.random.choice(npy_data["centroid"].shape[0], self.downsample_size, replace=False)
            )
            npy_data = npy_data[sampled_indices]
        else:
            npy_data = np.load(data_file)

        centroid = npy_data["centroid"]
        pressure = npy_data["pressure"]
        wss = npy_data["wss"]

        def clamp(x, min_val=1e-6):
            min_val = np.full_like(x, min_val, dtype=x.dtype)
            return np.where(np.abs(x) > np.abs(min_val), x, min_val)

        centroid = (centroid - self.mean_std_dict_npy["centroid_mean"]) / clamp(self.mean_std_dict_npy["centroid_std"])
        # TODO: maybe only the magnitude should be normalized by the mean and std of the wss

        inputs = {
            "centroids": centroid,  # (N, 3)
        }
        targets = {
            "normal": npy_data["normal"],  # (N, 3)
            "area": npy_data["area"],  # (N, 1)
            "idx": idx,  # (,)
            "pressure": pressure.reshape([-1]),  # (N, 1)
            "wss": wss,  # (N, 3)
        }
        return {
            "inputs": [v for v in inputs.values()],
            "targets": [v for v in targets.values()],
        }

    def __len__(self):
        return len(self.files)

    def data_to_dict(self, data):
        inputs = {k: data["inputs"][i] for i, k in enumerate(self.inputs_key)}
        targets = {k: data["targets"][i] for i, k in enumerate(self.targets_key)}
        file_name_list = [self.files[int(i)] for i in targets["idx"]]
        others = {
            **self.mean_std_dict,
            "num_points": [],
            "air_density": [],
            "flow_dir": [],
            "flow_speed": [],
            "reference_area": [],
            "c_p": [],
            "c_wss": [],
            "Cd": [],
        }

        for f in file_name_list:
            npz_data = np.load(f.replace(".npy", ".npz"))
            others["num_points"].append(npz_data["num_points"])
            others["air_density"].append(npz_data["air_density"])
            others["flow_dir"].append(npz_data["flow_dir"])
            others["flow_speed"].append(npz_data["flow_speed"])
            others["reference_area"].append(npz_data["frontal_area"])
            others["c_p"].append(npz_data["c_p"].reshape([-1,]))
            others["c_wss"].append(npz_data["c_wss"].reshape([-1,]))
            others["Cd"].append(npz_data["c_wss"].reshape([-1,]) + npz_data["c_p"].reshape([-1,]))
        others["file_name"] = [Path(f).name for f in file_name_list]

        for k in others.keys():
            if isinstance(others[k], list):
                if isinstance(others[k][0], str):
                    pass
                else:
                    others[k] = paddle.to_tensor(others[k])
        return inputs, targets, others


class NpyDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        n_train_num: int = 150, # 2772 # 2776-"0978/1034/2860/3641"
        n_test_num: int = 50, # 595
        train_sample_number: int = -1,
    ):
        self.data_dir = Path(data_dir)
        mean_std_file = self.data_dir / "mean_std.json"
        train_split_file = self.data_dir / "split/train_split.json"
        test_split_file = self.data_dir / "split/val_split.json"

        self.mean_std_file = mean_std_file
        super().__init__()
        
        self.train_data = PointDataset(train_split_file, self.mean_std_file, n_train_num, train_sample_number)
        self.test_data = PointDataset(test_split_file, self.mean_std_file, n_test_num)
        print("len(self.train_data)", len(self.train_data), type(len(self.train_data)))
        log.info(f"Train data size: , {len(self.train_data)}")
        log.info(f"Test  data size: , {len(self.test_data)}")
