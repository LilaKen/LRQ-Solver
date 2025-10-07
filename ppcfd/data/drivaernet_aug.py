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

"""
@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper:
"DrivAerNet++: A Large-Scale Multimodal Car Dataset with Computational Fluid Dynamics Simulations and Deep Learning Benchmarks".

The module defines two Paddle Datasets for loading and transforming 3D car models from the DrivAerNet++ dataset:
1. DrivAerNetPlusPlusDataset: Handles point cloud data, allowing loading, transforming, and augmenting 3D car models.
"""


from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from tqdm import tqdm
import numpy as np
import paddle
import pandas as pd

from ppcfd.data.base_datamodule import BaseDataModule


class DataAugmentation:
    """
    Class encapsulating various data augmentation techniques for point clouds.
    """
    @staticmethod
    def translate_pointcloud(
        pointcloud: np.ndarray,
        translation_range: Tuple[float, float] = (2.0 / 3.0, 3.0 / 2.0),
    ) -> np.ndarray:
        """
        Translates the pointcloud by a random factor within a given range.

        Args:
            pointcloud: The input point cloud as a np.ndarray.
            translation_range: A tuple specifying the range for translation factors.

        Returns:
            Translated point cloud as a np.ndarray.
        """
        xyz1 = np.random.uniform(
            low=translation_range[0], high=translation_range[1], size=[3]
        )
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype(
            "float32"
        )
        return paddle.to_tensor(data=translated_pointcloud, dtype="float32")

    @staticmethod
    def jitter_pointcloud(
        pointcloud: np.ndarray, sigma: float = 0.01, clip: float = 0.02
    ) -> np.ndarray:
        """
        Adds Gaussian noise to the pointcloud.

        Args:
            pointcloud: The input point cloud as a np.ndarray.
            sigma: Standard deviation of the Gaussian noise.
            clip: Maximum absolute value for noise.

        Returns:
            Jittered point cloud as a np.ndarray.
        """
        N, C = tuple(pointcloud.shape)
        jittered_pointcloud = pointcloud + paddle.clip(
            x=sigma * paddle.randn(shape=[N, C]), min=-clip, max=clip
        )
        return jittered_pointcloud

    @staticmethod
    def drop_points(pointcloud: np.ndarray, drop_rate: float = 0.1) -> np.ndarray:
        """
        Randomly removes points from the point cloud based on the drop rate.

        Args:
            pointcloud: The input point cloud as a np.ndarray.
            drop_rate: The percentage of points to be randomly dropped.

        Returns:
            The point cloud with points dropped as a np.ndarray.
        """
        num_drop = int(drop_rate * pointcloud.shape[0])
        drop_indices = np.random.choice(pointcloud.shape[0], num_drop, replace=False)
        keep_indices = np.setdiff1d(np.arange(pointcloud.shape[0]), drop_indices)
        dropped_pointcloud = pointcloud[keep_indices, :]
        return dropped_pointcloud


class DrivAerNetPlusPlusDataset(paddle.io.Dataset):
    """
    Paddle Dataset class for the DrivAerNet dataset, handling loading, transforming, and augmenting 3D car models.

    This dataset is designed for tasks involving aerodynamic simulations and deep learning models,
    specifically for predicting aerodynamic coefficients (e.g., Cd values) from 3D car models.

    Args:
        inputs_key (Tuple[str, ...]): Tuple of strings specifying the input keys.
            These keys correspond to the features extracted from the dataset,
            typically the 3D vertices of car models.
            Example: ("vertices",)
        targets_key (Tuple[str, ...]): Tuple of strings specifying the label keys.
            These keys correspond to the ground-truth labels, such as aerodynamic
            coefficients (e.g., Cd values).
            Example: ("cd_value",)
        weight_keys (Tuple[str, ...]): Tuple of strings specifying the weight keys.
            These keys represent optional weighting factors used during model training
            to handle class imbalance or sample importance.
            Example: ("weight_keys",)
        subset_dir (str): Path to the directory containing subsets of the dataset.
            This directory is used to divide the dataset into different subsets
            (e.g., train, validation, test) based on provided IDs.
        ids_file (str): Path to the file containing the list of IDs for the subset.
            The file specifies which models belong to the current subset (e.g., training IDs).
        data_dir (str): Root directory containing the 3D STL files of car models.
            Each 3D model is expected to be stored in a file named according to its ID.
        csv_file (str): Path to the CSV file containing metadata for the car models.
            The CSV file includes information such as aerodynamic coefficients,
            and may also map model IDs to specific attributes.
        num_points (int): Number of points to sample or pad each 3D point cloud to.
            If the model has more points than `num_points`, it will be subsampled.
            If it has fewer points, zero-padding will be applied.
        transform (Optional[Callable]): Optional transformation function applied to each sample.
            This can include augmentations like scaling, rotation, or jittering.
        pointcloud_exist (bool): Whether the point clouds are pre-processed and saved as `.pt` files.
            If `True`, the dataset will directly load the pre-saved point clouds
            instead of generating them from STL files.
    """

    def __init__(
        self,
        inputs_key: Tuple[str, ...],
        targets_key: Tuple[str, ...],
        weight_keys: Tuple[str, ...],
        subset_dir: str,
        ids_file: str,
        data_dir: str,
        csv_file: str,
        area_file: str,
        para_file: str,
        num_points: int,
        transform: Optional[Callable] = None,
        pointcloud_exist: bool = True,
        apply_augmentations: bool = True,
        n_sample: int = 4968,
        z_score: str = "mean_std.npy",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.inputs_key = inputs_key
        self.targets_key = targets_key
        self.targets_key.append("idx")
        self.weight_keys = weight_keys
        self.subset_dir = subset_dir
        self.ids_file = ids_file
        self.augmentation = DataAugmentation()
        self.cache = {}

        try:
            self.data_frame = pd.read_csv(csv_file)
        except Exception as e:
            logging.error(f"Failed to load CSV file: {csv_file}. Error: {e}")
            raise
        self.transform = transform
        self.num_points = num_points
        self.pointcloud_exist = pointcloud_exist

        try:
            with open(os.path.join(self.subset_dir, self.ids_file), "r") as file:
                subset_ids = file.read().split()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Error loading subset file {self.ids_file}: {e}")
        self.data_frame.loc[self.data_frame["Design"].str.startswith("F_D"), "Design"] = "DrivAer_" + self.data_frame["Design"]
        self.subset_indices = self.data_frame[self.data_frame["Design"].isin(subset_ids)].index.tolist()

        design_ids = set(self.data_frame["Design"].unique())
        subset_ids_set = set(subset_ids)
        missing_ids = list(subset_ids_set - design_ids)
        # https://github.com/Mohamedelrefaie/DrivAerNet/issues/21
        logging.info(f"Missing files in [{ids_file}] are : {missing_ids}")
        np.random.shuffle(self.subset_indices)
        self.subset_indices = self.subset_indices[:n_sample]
        self.data_frame = self.data_frame.loc[self.subset_indices].reset_index(drop=True)

        def match(df, design_id, design="Car Design", col_name="Frontal Area (m²)"):
            matched_rows = df[df[design] == design_id]
            if not matched_rows.empty:
                result = matched_rows[col_name].iloc[0]
            else:
                result = 1e-5  # no data return zero
                logging.warning(f"No data found for {design_id} {col_name} in csv file")
            return result
        self.reference_area_df = pd.read_csv(
            Path(area_file).as_posix()
        )
        self.reference_area_df.loc[self.reference_area_df["Car Design"].str.startswith("F_D"), "Car Design"] = "DrivAer_" + self.reference_area_df["Car Design"]
        FA_list = []
        for f in self.data_frame["Design"]:
            FA = match(self.reference_area_df, f)
            FA_list.append(FA)
        self.data_frame["Frontal Area (m²)"] = FA_list

        # 4. Load geometric parameters from para_file (by column index)
        try:
            self.para_df = pd.read_csv(para_file)
        except Exception as e:
            logging.error(f"Failed to load para_file: {para_file}. Error: {e}")
            raise

        if self.para_df.shape[1] < 24:
            raise ValueError(f"para_file must have at least 24 columns, but got {self.para_df.shape[1]}.")

        design_ids = self.para_df.iloc[:, 0]
        param_data = self.para_df.iloc[:, 1:24]

        try:
            param_data = param_data.astype(float)
        except Exception as e:
            logging.error(f"Failed to convert parameters to float: {e}")
            raise

        self.param_dict = {}
        for design_id, row in zip(design_ids, param_data.values):
            clean_id = str(design_id)
            if clean_id.startswith("DrivAer_"):
                clean_id = clean_id[len("DrivAer_"):]

            self.param_dict[clean_id] = row.astype(np.float32)

        if len(self.param_dict) == 0:
            raise ValueError("No valid parameter data loaded from para_file.")

        all_params = np.stack(list(self.param_dict.values()), axis=0)
        self.param_min = all_params.min(axis=0)
        self.param_max = all_params.max(axis=0)
        self.param_max[self.param_max == self.param_min] += 1e-8
        logging.info(f"Loaded 23 geometric parameters from columns 2 to 24 of para_file.")

        if os.path.isfile(z_score):
            logging.info(f"Loading z_score dict from {z_score}")
            self.mean_std_dict = np.load(z_score, allow_pickle=True).item()
        else:
            self.mean_std_dict = {}

        def cache_files(idx):
            row = self.data_frame.iloc[idx]
            design_id = row["Design"]
            # cd_value = row["Average Cd"] / (0.46 * row["Frontal Area (m²)"])
            cd_value = row["Average Cd"]
            if self.pointcloud_exist:
                try:
                    vertices = self._load_point_cloud(design_id)
                    if vertices is None:
                        raise ValueError(
                            f"Point cloud for design {design_id} is not found or corrupted."
                        )
                except Exception as e:
                    raise ValueError(
                        f"Failed to load point cloud for design {design_id}: {e}"
                    )

            if apply_augmentations:
                vertices = self.augmentation.translate_pointcloud(vertices)
                vertices = self.augmentation.jitter_pointcloud(vertices)

            if self.transform:
                vertices = self.transform(vertices)

            vertices = self.min_max_normalize(vertices)
            base_id = design_id

            if base_id.startswith("DrivAer_"):
                base_id = base_id[len("DrivAer_"):]

            if base_id in self.param_dict:
                raw_params = self.param_dict[base_id]
                params = self.normalize_params(raw_params)  # normalize [0,1]
            else:
                params = np.zeros(23, dtype=np.float32)
            inputs_tuple = (vertices, params)
            cd_value = np.array([float(cd_value)], dtype=np.float32).reshape([-1])  # shape: (1,)

            return {
                "inputs": inputs_tuple,  # tuple of numpy arrays
                "targets": [cd_value, idx],  # list of numpy arrays
            }

        device = paddle.device.get_device()
        paddle.device.set_device("cpu")
        for idx in tqdm(range(n_sample), desc="Caching Samples"):
            self.cache[idx] = cache_files(idx)
        paddle.device.set_device(device)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data_frame)

    def data_to_dict(self, data):
        inputs = dict(zip(self.inputs_key, data["inputs"]))
        targets = dict(zip(self.targets_key, data["targets"]))
        idx = targets["idx"].numpy()
        design_id = self.data_frame.iloc[idx]["Design"].tolist()
        reference_area = self.data_frame.iloc[idx]["Frontal Area (m²)"].tolist()
        others = {
            "file_name": design_id,
            "Cd": targets["Cd"],
            "reference_area": reference_area,  # B N C
        }

        return inputs, targets, others

    def min_max_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalizes the data to the range [0, 1] based on min and max values.
        """
        min_vals = data.min(axis=0, keepdim=True)
        max_vals = data.max(axis=0, keepdim=True)
        normalized_data = (data - min_vals) / (max_vals - min_vals)
        return normalized_data

    def normalize_params(self, params: np.ndarray) -> np.ndarray:
        """
        Normalizes the geometric parameters to [0, 1] using global min and max values.

        Args:
            params (np.ndarray): Raw parameters, shape (23,)

        Returns:
            np.ndarray: Normalized parameters, shape (23,), range [0, 1]
        """
        normalized = (params - self.param_min) / (self.param_max - self.param_min)
        return normalized.astype(np.float32)

    def _sample_or_pad_vertices(
        self, vertices: paddle.Tensor, num_points: int
    ) -> paddle.Tensor:
        num_vertices = vertices.shape[0]
        if num_vertices > num_points:
            indices = np.random.choice(num_vertices, num_points, replace=False)
            vertices = vertices[indices]
        elif num_vertices < num_points:
            padding = paddle.zeros(
                shape=(num_points - num_vertices, 3), dtype="float32"
            )
            vertices = paddle.concat(x=(vertices, padding), axis=0)
        return vertices

    def _load_point_cloud(self, design_id: str):
        load_path = os.path.join("/date2/ken/ppcfd_transformer-transformer/data/drivaerpp/paddle_tensor", f"{design_id}.paddle_tensor")
        load_path_npy = os.path.join(self.data_dir, f"{design_id.replace('DrivAer_', '')}.npy")
        if os.path.exists(load_path_npy) and os.path.getsize(load_path_npy) > 0:
            try:
                vertices = paddle.to_tensor(np.load(load_path_npy))
            except (EOFError, RuntimeError, ValueError) as e:
                raise Exception(
                    f"Error loading point cloud from {load_path}: {e}"
                ) from e
        elif os.path.exists(load_path) and os.path.getsize(load_path) > 0:
            logging.info(f"Loading unsampled case {Path(load_path).name}")
            try:
                vertices = paddle.load(path=str(load_path))
            except (EOFError, RuntimeError, ValueError) as e:
                raise Exception(
                    f"Error loading point cloud from {load_path}: {e}"
                ) from e
        else:
            raise ValueError(f"Point cloud for design {design_id} is not found or corrupted.")

        num_vertices = vertices.shape[0]
        if num_vertices > self.num_points:
            indices = np.random.choice(num_vertices, self.num_points, replace=False)
            vertices = vertices.numpy()[indices]
        else:
            vertices = vertices.numpy()
        return vertices

    def __getitem__(
        self, idx: int, apply_augmentations: bool = True
    ) -> Dict[str, List[np.ndarray]]:
        return self.cache[idx]

class DrivAerNet_Aug_DataModule(BaseDataModule):
    def __init__(
        self,
        inputs_key: Tuple[str, ...],
        targets_key: Tuple[str, ...],
        weight_keys: Tuple[str, ...],
        subset_dir: str,
        data_dir: str,
        csv_file: str,
        area_file: str,
        para_file: str,
        num_points: int,
        transform: Optional[Callable] = None,
        pointcloud_exist: bool = True,
        train_ids_file: str = "train_ids.txt",
        test_ids_file: str = "test_ids.txt",
        val_ids_file: str = "val_ids.txt",
        n_train_num: int = 10,
        n_test_num: int = 10,
        n_val_num: int = 1,
        train_sample_number: int = 1000,
        z_score: str = "mean_std.npy",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.inputs_key = inputs_key
        self.targets_key = targets_key
        self.weight_keys = weight_keys
        self.subset_dir = subset_dir
        self.augmentation = DataAugmentation()
        self.cache = {}

        self.train_data = DrivAerNetPlusPlusDataset(
            inputs_key=inputs_key,
            targets_key=targets_key,
            weight_keys=weight_keys,
            subset_dir=subset_dir,
            ids_file=train_ids_file,
            data_dir=data_dir,
            csv_file=csv_file,
            area_file=area_file,
            para_file=para_file,
            num_points=num_points,
            transform=transform,
            pointcloud_exist=pointcloud_exist,
            n_sample=n_train_num,
            z_score=z_score,
        )

        self.val_data = DrivAerNetPlusPlusDataset(
            inputs_key=inputs_key,
            targets_key=targets_key,
            weight_keys=weight_keys,
            subset_dir=subset_dir,
            ids_file=val_ids_file,
            data_dir=data_dir,
            csv_file=csv_file,
            area_file=area_file,
            para_file=para_file,
            num_points=num_points,
            transform=transform,
            pointcloud_exist=pointcloud_exist,
            n_sample=n_val_num
        )
        self.val_data.mean_std_dict = self.train_data.mean_std_dict

        self.test_data = DrivAerNetPlusPlusDataset(
            inputs_key=inputs_key,
            targets_key=targets_key,
            weight_keys=weight_keys,
            subset_dir=subset_dir,
            ids_file=test_ids_file,
            data_dir=data_dir,
            csv_file=csv_file,
            area_file=area_file,
            para_file=para_file,
            num_points=num_points,
            transform=transform,
            pointcloud_exist=pointcloud_exist,
            n_sample=n_test_num,
        )
        self.test_data.mean_std_dict = self.train_data.mean_std_dict
