try:
    from .cfd_datamodule import AhmedBodyDataModule
    from .cfd_datamodule import CFDDataModule
    from .cfd_datamodule import CFDSDFDataModule
    from .datamodule_lazy import DrivAerDataModule
except ImportError:
    print("open3d 库未安装，因此不会导入相关模块。")

from .beam_datamodule import BeamDataModule
from .csv_datamodule import CSVDataModule
from .drivaernet_aug import DrivAerNet_Aug_DataModule
from .fake_datamodule import FakeDataModule
from .pde_cylinder_datamodule import PDE_Cylinder_DataModule
from .point_datamodule import NpyDataModule
from .pointcloud_datamodule import PointCloudDataModule
from .pointcloud_vel_datamodule import PointCloud_Vel_DataModule
from .shapenetcar_datamodule import GraphDataset
from .starccm_datamodule import StarCCMDataModule

__all__ = [
    "PointCloudDataModule",
    "AhmedBodyDataModule",
    "DrivAerDataModule",
    "NpyDataModule",
    "PDE_Cylinder_DataModule",
    "StarCCMDataModule",
    "DrivAerNet_Aug_DataModule",
    "CSVDataModule",
]
