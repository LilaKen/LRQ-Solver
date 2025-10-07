# trace generated using paraview version 5.11.0
# import paraview
# paraview.compatibility.major = 5
# paraview.compatibility.minor = 11

import os
from pathlib import Path
from time import time

import numpy as np
from paraview import servermanager as sm
from paraview.simple import *
from paraview.vtk.numpy_interface import dataset_adapter as dsa


def calculate_normal_and_area_by_paraview(file_name, debug=False):
    # create a new 'EnSight Reader'
    case_data = EnSightReader(
        registrationName=Path(file_name).name, CaseFileName=file_name
    )

    # create a new 'Extract Region Surface'
    extractRegionSurface1 = ExtractRegionSurface(
        registrationName="ExtractRegionSurface1", Input=case_data
    )

    # create a new 'Generate Surface Normals'
    SurfaceNormals1 = SurfaceNormals(
        registrationName="SurfaceNormals1", Input=extractRegionSurface1
    )

    # Properties modified on SurfaceNormals1
    SurfaceNormals1.ComputeCellNormals = 1

    # create a new 'Cell Size'
    cellSize1 = CellSize(registrationName="CellSize1", Input=SurfaceNormals1)

    # Get the data for the 'CellSize' into numpy array
    vtk_data = sm.Fetch(cellSize1)
    vtk_data = dsa.WrapDataObject(vtk_data)
    normals_list = vtk_data.GetCellData()["Normals"].GetArrays()
    area_list = vtk_data.GetCellData()["Area"].GetArrays()
    normals = np.concatenate(normals_list, axis=0)
    area = np.concatenate(area_list, axis=0)
    return normals, area


if __name__ == "__main__":
    data_dir = "./data_fake_starccm+/case/"

    file_list = [
        data_dir + f
        for f in os.listdir(data_dir)  # 遍历目录下的所有文件(paddledict)
        if f.endswith(".case")  # 检查文件结尾是否为case，是的话，就加入列表
    ]

    for file_name in file_list:
        time_start = time()
        normals, area = calculate_normal_and_area_by_paraview(file_name, debug=True)
        print(f"read [{file_name}] done, time: {(time() - time_start):.2f}")
