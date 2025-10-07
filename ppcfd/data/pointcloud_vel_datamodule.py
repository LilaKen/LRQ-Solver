import os
import time
from pathlib import Path

import numpy as np
import paddle
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from vtk.util.numpy_support import vtk_to_numpy

from ppcfd.data.base_datamodule import BaseDataModule
from ppcfd.data.pointcloud_datamodule import load_mean_std
from ppcfd.data.starccm_datamodule import get_centroids
from ppcfd.data.starccm_datamodule import get_nodes


def load_velocity(polydata):
    point_data_keys = [
        polydata.GetCellData().GetArrayName(i)
        for i in range(polydata.GetCellData().GetNumberOfArrays())
    ]
    if "UMeanTrim" in point_data_keys:
        vel = vtk_to_numpy(polydata.GetCellData().GetArray("UMeanTrim")).astype(
            np.float32
        )
        return vel
    else:
        print("point_data_keys in polydata", point_data_keys)
        raise NotImplementedError("No velocity found in the point cloud file.")


class PointCloudDataset(paddle.io.Dataset):
    def __init__(self, root_dir, train=True, translate=True, num=1, train_sample_number=4000):
        """
        Args:
            root_dir (string): Directory with all the point cloud files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.file_list = [f for f in os.listdir(root_dir) if (f.endswith(".vtp"))][:num]
        if len(self.file_list) == 0:
            raise RuntimeError(f"No files found in provided {root_dir} directory.")
        self.train = train
        self.translate = translate
        self.train_sample_number = train_sample_number
        self.mean_std_dict = load_mean_std(root_dir / "../mean_std.paddledict")
        self.inputs_key = ["centroids", "local_centroid"]
        self.targets_key = ["idx", "vel"]

    def __len__(self):
        return len(self.file_list)

    def data_to_dict(self, data):
        inputs = {k: data["inputs"][i] for i, k in enumerate(self.inputs_key)}
        targets = {k: data["targets"][i] for i, k in enumerate(self.targets_key)}
        file_name_list = [self.file_list[i] for i in targets["idx"]]
        Cd_list = [float(1.0) for f in file_name_list]
        FA_list = [float(1.0) for f in file_name_list]
        others = {
            "file_name": file_name_list,
            "Cd": np.array(Cd_list).reshape([-1, 1, 1]),             # B N C
            "reference_area": np.array(FA_list).reshape([-1, 1, 1]), # B N C
            **self.mean_std_dict
        }

        return inputs, targets, others

    def normlalize_input(self, points, sampled_indices, train=True):
        mean_std_dict = self.mean_std_dict
        points_min = np.min(points, axis=0, keepdims=True)
        points_max = np.max(points, axis=0, keepdims=True)
        sampled_points = points[sampled_indices]
        local_sampled_points = (sampled_points - points_min) / (points_max - points_min)
        if self.train:
            translation_vector = np.random.rand(3) * 0.01 - 0.005  # 随机平移向量
            sampled_points += translation_vector
        sampled_points = (sampled_points - mean_std_dict["centroid_mean"]) / mean_std_dict["centroid_std"]
        return{
            "centroids": paddle.to_tensor(sampled_points),
            "local_centroid": paddle.to_tensor(local_sampled_points),
        }

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName((self.root_dir / file_name).as_posix())
        reader.Update()
        polydata = reader.GetOutput()
        points = get_centroids(polydata)
        vel = load_velocity(polydata)[:, 0]

        if self.train:
            sampled_indices = np.random.choice(
                np.arange(points.shape[0]),
                self.train_sample_number,
                replace=False,
            )
        else:
            sampled_indices = np.arange(points.shape[0])

        inputs = {
            **self.normlalize_input(
                points,
                sampled_indices,
                self.train
                ),
        }

        targets = {
            "idx": idx,
            "vel": vel[sampled_indices].astype(np.float32),
        }
        
        return {
            "inputs": [v for v in inputs.values()],
            "targets": [v for v in targets.values()],
        }


class PointCloud_Vel_DataModule(BaseDataModule):
    def __init__(self, data_dir, n_train_num, n_test_num, train_sample_number):
        BaseDataModule.__init__(self)
        self.train_data_dir = Path(data_dir) / "train"
        self.test_data_dir = Path(data_dir) / "test"
        if n_train_num != 0:
            self.train_data = PointCloudDataset(
                self.train_data_dir, train=True, train_sample_number=train_sample_number, num=n_train_num
            )
        self.test_data = PointCloudDataset(
            self.test_data_dir, train=False, train_sample_number=train_sample_number, num=n_test_num
        )

    def save_vtk(self, file_path, var, var_name, vtk_name):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(file_path)
        reader.Update()
        polydata = reader.GetOutput()
        np_array = var.numpy()[0]
        vtk_array = numpy_to_vtk(np_array)
        vtk_array.SetName(var_name)  # 设置数据的名称
        polydata.GetCellData().AddArray(vtk_array)
        appendFilter = vtk.vtkAppendFilter()
        appendFilter.AddInputData(polydata)
        appendFilter.Update()
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(vtk_name)  # 设置输出文件的名称
        writer.SetInputData(appendFilter.GetOutput())
        writer.Write()
