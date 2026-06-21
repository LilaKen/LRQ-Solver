from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ppcfd.data.base_datamodule import BaseDataModule


class DataAugmentation:
    @staticmethod
    def translate_pointcloud(
        pointcloud: np.ndarray | torch.Tensor,
        translation_range: Tuple[float, float] = (2.0 / 3.0, 3.0 / 2.0),
    ) -> torch.Tensor:
        pointcloud = torch.as_tensor(pointcloud, dtype=torch.float32)
        scale = torch.empty(3).uniform_(translation_range[0], translation_range[1])
        shift = torch.empty(3).uniform_(-0.2, 0.2)
        return pointcloud * scale + shift

    @staticmethod
    def jitter_pointcloud(
        pointcloud: np.ndarray | torch.Tensor, sigma: float = 0.01, clip: float = 0.02
    ) -> torch.Tensor:
        pointcloud = torch.as_tensor(pointcloud, dtype=torch.float32)
        noise = torch.clamp(sigma * torch.randn_like(pointcloud), min=-clip, max=clip)
        return pointcloud + noise

    @staticmethod
    def drop_points(pointcloud: np.ndarray, drop_rate: float = 0.1) -> np.ndarray:
        num_drop = int(drop_rate * pointcloud.shape[0])
        drop_indices = np.random.choice(pointcloud.shape[0], num_drop, replace=False)
        keep_indices = np.setdiff1d(np.arange(pointcloud.shape[0]), drop_indices)
        return pointcloud[keep_indices, :]


class DrivAerNetPlusPlusDataset(torch.utils.data.Dataset):
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
        self.data_dir = Path(data_dir)
        self.inputs_key = list(inputs_key)
        self.targets_key = list(targets_key)
        self.targets_key.append("idx")
        self.weight_keys = list(weight_keys)
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

        f_d_mask = self.data_frame["Design"].str.startswith("F_D")
        self.data_frame.loc[f_d_mask, "Design"] = (
            "DrivAer_" + self.data_frame.loc[f_d_mask, "Design"]
        )
        self.subset_indices = self.data_frame[
            self.data_frame["Design"].isin(subset_ids)
        ].index.tolist()

        design_ids = set(self.data_frame["Design"].unique())
        subset_ids_set = set(subset_ids)
        missing_ids = list(subset_ids_set - design_ids)
        logging.info(f"Missing files in [{ids_file}] are : {missing_ids}")
        if not self.subset_indices:
            raise ValueError(
                f"No samples from {ids_file} matched the Design column in {csv_file}."
            )

        self.subset_indices = [
            idx
            for idx in self.subset_indices
            if self._has_torch_point_cloud_file(self.data_frame.iloc[idx]["Design"])
        ]
        if not self.subset_indices:
            raise ValueError(
                f"No samples from {ids_file} have Torch-readable point clouds "
                f"(.npy/.pt/.pth) in {self.data_dir}."
            )
        np.random.shuffle(self.subset_indices)
        sample_count = min(n_sample, len(self.subset_indices))
        self.subset_indices = self.subset_indices[:sample_count]
        self.data_frame = self.data_frame.loc[self.subset_indices].reset_index(
            drop=True
        )

        self.reference_area_df = pd.read_csv(Path(area_file).as_posix())
        f_d_area_mask = self.reference_area_df["Car Design"].str.startswith("F_D")
        self.reference_area_df.loc[f_d_area_mask, "Car Design"] = (
            "DrivAer_" + self.reference_area_df.loc[f_d_area_mask, "Car Design"]
        )
        self.data_frame["Frontal Area (m²)"] = [
            self._match(
                self.reference_area_df, design_id, "Car Design", "Frontal Area (m²)"
            )
            for design_id in self.data_frame["Design"]
        ]

        try:
            self.para_df = pd.read_csv(para_file)
        except Exception as e:
            logging.error(f"Failed to load para_file: {para_file}. Error: {e}")
            raise

        if self.para_df.shape[1] < 24:
            raise ValueError(
                f"para_file must have at least 24 columns, got {self.para_df.shape[1]}."
            )

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
                clean_id = clean_id[len("DrivAer_") :]
            self.param_dict[clean_id] = row.astype(np.float32)

        if len(self.param_dict) == 0:
            raise ValueError("No valid parameter data loaded from para_file.")

        all_params = np.stack(list(self.param_dict.values()), axis=0)
        self.param_min = all_params.min(axis=0)
        self.param_max = all_params.max(axis=0)
        self.param_max[self.param_max == self.param_min] += 1e-8
        logging.info("Loaded 23 geometric parameters from para_file.")

        if z_score and os.path.isfile(z_score):
            logging.info(f"Loading z_score dict from {z_score}")
            self.mean_std_dict = np.load(z_score, allow_pickle=True).item()
        else:
            self.mean_std_dict = {}

        for idx in tqdm(range(sample_count), desc="Caching Samples"):
            self.cache[idx] = self._cache_file(idx, apply_augmentations)

    def __len__(self) -> int:
        return len(self.data_frame)

    @staticmethod
    def _match(df, design_id, design="Car Design", col_name="Frontal Area (m²)"):
        matched_rows = df[df[design] == design_id]
        if not matched_rows.empty:
            return matched_rows[col_name].iloc[0]
        logging.warning(f"No data found for {design_id} {col_name} in csv file")
        return 1e-5

    def _cache_file(self, idx, apply_augmentations):
        row = self.data_frame.iloc[idx]
        design_id = row["Design"]
        cd_value = row["Average Cd"]

        if self.pointcloud_exist:
            vertices = self._load_point_cloud(design_id)
        else:
            raise NotImplementedError("Generating point clouds from raw geometry is not implemented.")

        if apply_augmentations:
            vertices = self.augmentation.translate_pointcloud(vertices)
            vertices = self.augmentation.jitter_pointcloud(vertices)
        else:
            vertices = torch.as_tensor(vertices, dtype=torch.float32)

        if self.transform:
            vertices = self.transform(vertices)

        vertices = self.min_max_normalize(vertices)
        base_id = design_id
        if base_id.startswith("DrivAer_"):
            base_id = base_id[len("DrivAer_") :]

        if base_id in self.param_dict:
            raw_params = self.param_dict[base_id]
            params = self.normalize_params(raw_params)
        else:
            params = np.zeros(23, dtype=np.float32)

        inputs_tuple = (vertices.float(), torch.from_numpy(params).float())
        cd_value = torch.tensor([float(cd_value)], dtype=torch.float32)

        return {
            "inputs": inputs_tuple,
            "targets": [cd_value, torch.tensor(idx, dtype=torch.long)],
        }

    def data_to_dict(self, data):
        inputs = dict(zip(self.inputs_key, data["inputs"]))
        targets = dict(zip(self.targets_key, data["targets"]))
        idx = targets["idx"].detach().cpu().numpy().reshape(-1).astype(int)
        design_id = self.data_frame.iloc[idx]["Design"].tolist()
        reference_area = self.data_frame.iloc[idx]["Frontal Area (m²)"].tolist()
        others = {
            "file_name": design_id,
            "Cd": targets["Cd"],
            "reference_area": reference_area,
        }

        return inputs, targets, others

    def min_max_normalize(self, data: torch.Tensor) -> torch.Tensor:
        min_vals = data.min(dim=0, keepdim=True).values
        max_vals = data.max(dim=0, keepdim=True).values
        denom = torch.clamp(max_vals - min_vals, min=1e-8)
        return (data - min_vals) / denom

    def normalize_params(self, params: np.ndarray) -> np.ndarray:
        normalized = (params - self.param_min) / (self.param_max - self.param_min)
        return normalized.astype(np.float32)

    def _sample_or_pad_vertices(self, vertices: np.ndarray) -> torch.Tensor:
        vertices = np.asarray(vertices, dtype=np.float32)
        num_vertices = vertices.shape[0]
        if num_vertices > self.num_points:
            indices = np.random.choice(num_vertices, self.num_points, replace=False)
            vertices = vertices[indices]
        elif num_vertices < self.num_points:
            padding = np.zeros((self.num_points - num_vertices, 3), dtype=np.float32)
            vertices = np.concatenate([vertices, padding], axis=0)
        return torch.from_numpy(vertices.astype(np.float32))

    def _load_point_cloud(self, design_id: str) -> torch.Tensor:
        candidates = self._torch_point_cloud_candidates(design_id)

        for path in candidates:
            if not path.exists() or path.stat().st_size == 0:
                continue
            if path.suffix == ".npy":
                return self._sample_or_pad_vertices(np.load(path))
            loaded = torch.load(path, map_location="cpu")
            if isinstance(loaded, torch.Tensor):
                return self._sample_or_pad_vertices(loaded.detach().cpu().numpy())
            return self._sample_or_pad_vertices(np.asarray(loaded, dtype=np.float32))

        paddle_candidates = [self.data_dir / f"{name}.paddle_tensor" for name in names]
        existing_paddle = [path for path in paddle_candidates if path.exists()]
        if existing_paddle:
            raise ValueError(
                "Found Paddle tensor point cloud files, but the Torch version reads "
                f".npy/.pt/.pth files. Convert or point data_module.data_dir to .npy data. "
                f"First Paddle file: {existing_paddle[0]}"
            )

        raise ValueError(f"Point cloud for design {design_id} is not found in {self.data_dir}.")

    def _torch_point_cloud_candidates(self, design_id: str):
        names = [design_id]
        if design_id.startswith("DrivAer_"):
            names.append(design_id[len("DrivAer_") :])

        candidates = []
        for name in names:
            candidates.extend(
                [
                    self.data_dir / f"{name}.npy",
                    self.data_dir / f"{name}.pt",
                    self.data_dir / f"{name}.pth",
                ]
            )
        return candidates

    def _has_torch_point_cloud_file(self, design_id: str):
        return any(
            path.exists() and path.stat().st_size > 0
            for path in self._torch_point_cloud_candidates(design_id)
        )

    def __getitem__(self, idx: int):
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
            n_sample=n_val_num,
            z_score=z_score,
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
            z_score=z_score,
        )
        self.test_data.mean_std_dict = self.train_data.mean_std_dict
