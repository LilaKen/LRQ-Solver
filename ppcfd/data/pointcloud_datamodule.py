import os
import re
import paddle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from ppcfd.data.base_datamodule import BaseDataModule


def load_mean_std(input_file):
    """
    Load mean and standard deviations from a text file.
    Args:
    input_file (Path): The path to the text file containing the saved
    mean and std values.

    Returns:
    dict: A dictionary with keys as the data categories and values as
    tuples of mean and std.
    """
    mean_std_dict = paddle.load(input_file.as_posix())
    mean_std_dict = {
        "p_mean": mean_std_dict["press_std"],
        "p_std": mean_std_dict["press_std"],
        "wss_mean": mean_std_dict["wss_mean"],
        "wss_std": mean_std_dict["wss_std"],
        "v_mean": mean_std_dict["vel_mean"],
        "v_std": mean_std_dict["vel_std"],
        "centroid_mean": mean_std_dict["centroid_mean"],
        "centroid_std": mean_std_dict["centroid_std"],
    }
    return mean_std_dict


def normlalize_input(mean_std_dict, points, sampled_indices, train=True):
    points_min = np.min(points, axis=0, keepdims=True)
    points_max = np.max(points, axis=0, keepdims=True)
    centroids = points[sampled_indices]
    local_centroid = (centroids - points_min) / (points_max - points_min)
    if train:
        translation_vector = np.random.rand(3) * 0.01 - 0.005
        centroids += translation_vector

    centroids = (centroids - mean_std_dict["centroid_mean"]) / mean_std_dict[
        "centroid_std"
    ]
    return {
        "centroids": paddle.to_tensor(centroids),
        "local_centroid": paddle.to_tensor(local_centroid),
    }


class PointCloudDataset(paddle.io.Dataset):
    """
    A dataset class for handling point cloud data.
    Args:
        root_dir (string): The directory containing all the point cloud files.
        train (bool, optional): A flag indicating whether the dataset is for training.
        Defaults to True.
        translate (bool, optional): A flag indicating whether to apply translation
        to the point cloud data during training. Defaults to True.
        num (int, optional): The number of point cloud files to load from the directory.
        Defaults to 1.
    """

    def __init__(
        self, root_dir, inputs_key, targets_key, train=True, translate=True, num=1, train_sample_number=4000
    ):
        self.root_dir = root_dir
        self.train_sample_number = train_sample_number
        self.df_Cx = pd.read_csv(
            (self.root_dir / "../AeroCoefficients_DrivAerNet_FilteredCorrected.csv").as_posix()
        )
        self.df_FA = pd.read_csv(
            (self.root_dir / "../DrivAerNetPlusPlus_CarDesign_Areas.csv").as_posix()
        )
        # go through folders
        self.file_list = [
            f for f in os.listdir(root_dir) if f.endswith(".npy") and "centroid" in f
        ]
        # filter broken files
        substrings_to_remove = ["0978", "1034", "2860", "3641"]
        self.file_list = [
            item
            for item in self.file_list
            if not any(sub in item for sub in substrings_to_remove)
        ]
        # set train number
        self.file_list = self.file_list[:num]
        if len(self.file_list) == 0:
            raise RuntimeError(f"No files found in provided {root_dir} directory.")
        self.train = train
        self.mean_std_dict_npy = load_mean_std(root_dir / "../mean_std.paddledict")
        self.mean_std_dict = {
            k: paddle.to_tensor(self.mean_std_dict_npy[k])
            for k in self.mean_std_dict_npy.keys()
        }
        self.inputs_key = inputs_key
        self.targets_key = targets_key

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[int(idx)]
        file_path = os.path.join(self.root_dir, file_name)
        points = np.load(file_path).astype(np.float32)
        if self.train:
            sampled_indices = np.random.choice(
                np.arange(points.shape[0]),
                self.train_sample_number,
                replace=False,
            )
        else:
            sampled_indices = np.arange(points.shape[0])

        p = np.load(
            os.path.join(self.root_dir, file_name.replace("centroid", "press"))
        ).astype(np.float32).reshape([-1, 1])
        press_sample = p[sampled_indices]

        wss = np.load(
            os.path.join(
                self.root_dir, file_name.replace("centroid", "wallshearstress")
            )
        ).astype(np.float32)

        normal = np.load(
            os.path.join(self.root_dir, file_name.replace("centroid", "normal"))
        ).astype(np.float32)

        areas = (
            np.load(os.path.join(self.root_dir, file_name.replace("centroid", "area")))
            .astype(np.float32)
            .reshape([-1, 1])
        )
        wss_sample = wss[sampled_indices]

        inputs = {
            **self._normlalize_input(points, sampled_indices, self.train),
        }

        targets = {
            "normal": normal[sampled_indices],
            "areas": areas[sampled_indices],
            "idx": idx,
            "pressure": press_sample,
            "wss": wss_sample,
        }

        return {
            "inputs": paddle.concat(x=[v for v in inputs.values()], axis=-1),
            "targets": [v for v in targets.values()],
        }

    def data_to_dict(self, data):
        def match(df, f, design="Car Design", col_name="Frontal Area (m²)"):
            matched_rows = df[df[design] == f]
            if not matched_rows.empty:
                result = matched_rows[col_name].iloc[0]
            else:
                result = 2.5  # no data return zero
                logging.warning(f"No data found for {design} {col_name} in csv file")
            return result

        inputs = dict(zip(self.inputs_key, data["inputs"]))
        targets = dict(zip(self.targets_key, data["targets"]))
        file_name_list = [self.file_list[i] for i in targets["idx"]]
        Cd_list = [
            "DrivAer_F_D_WM_WW_" + re.findall(r"\d+", f)[0] for f in file_name_list
        ]
        Cd_list = [match(self.df_Cx, f, "Design", "Average Cd") for f in Cd_list]
        Cd_list = [float(f) for f in Cd_list]
        FA_list = ["F_D_WM_WW_" + re.findall(r"\d+", f)[0] for f in file_name_list]
        FA_list = [
            match(self.df_FA, f, "Car Design", "Frontal Area (m²)") for f in FA_list
        ]
        FA_list = [float(f) for f in FA_list]
        others = {
            "file_name": file_name_list,
            "Cd": np.array(Cd_list).reshape([-1, 1, 1]),  # B N C
            "reference_area": np.array(FA_list).reshape([-1, 1, 1]),  # B N C
        }

        return inputs, targets, others

    def _normlalize_input(self, points, sampled_indices, train=True):
        mean_std_dict = self.mean_std_dict_npy
        return normlalize_input(mean_std_dict, points, sampled_indices, train=True)


class PointCloudDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir,
        n_train_num,
        n_val_num,
        n_test_num,
        train_sample_number,
        inputs_key = ["centroids", "local_centroid"],
        targets_key = ["normal", "areas", "idx", "pressure", "wss"]
    ):
        """
        PointCloudDataModule class for handling point cloud data.
        Args:
            train_data_dir (str): Path to the directory containing the training data.
            test_data_dir (str): Path to the directory containing the testing data.
            n_train_num (int): Number of training samples to load.
            n_test_num (int): Number of testing samples to load.
            train_sample_number (int): Number of samples to randomly sample from each point cloud.
            inputs_key (list): List of input keys to use when loading the data.
            targets_key (list): List of target keys to use when loading the data.
        """
        BaseDataModule.__init__(self)
        self.train_data_dir = Path(data_dir) / "train"
        self.test_data_dir = Path(data_dir) / "test"
        self.val_data_dir = Path(data_dir) / "val"

        self.train_data = PointCloudDataset(
            self.train_data_dir,
            inputs_key,
            targets_key,
            train=True,
            num=n_train_num,
            train_sample_number=train_sample_number,
        )

        self.val_data = PointCloudDataset(
            self.val_data_dir,
            inputs_key,
            targets_key,
            train=False, 
            num=n_val_num)

        self.test_data = PointCloudDataset(
            self.test_data_dir,
            inputs_key,
            targets_key,
            train=False, 
            num=n_test_num)
