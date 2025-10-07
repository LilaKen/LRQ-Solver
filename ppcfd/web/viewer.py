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

# 修改彩标范围，固定在[-1000,0]，可能还需要修改cmap
# 检测并显示载入stl文件的长度单位，并在判断后转换
# 提升stramlit文件上传大小限制
import os
import re
import subprocess
import sys
import time
import json
from pathlib import Path

sys.path.append(".")

import pyvista as pv
import streamlit as st
import vtk
from stpyvista import stpyvista

from utils.utils import detect_and_convert_units, get_bounds_info  # 确保这些函数可用


# 辅助函数：处理数据（单位转换和边界信息获取）
def process_data(data, current_unit, target_unit):
    """
    处理数据，包括单位转换和边界信息获取。
    """
    try:
        min_value, max_value = get_bounds_info(data)
        
        converted_data = [
            detect_and_convert_units(value, current_unit, target_unit)
            for value in data
        ]
        return converted_data, (converted_min, converted_max)
    except ValueError as e:
        raise ValueError(f"Data processing error: {e}")
predict_result_json = Path("./output/predict_result.json")
work_dir = Path("./output")
large_stl = False
input_filename = None
pred_data_key = "void"

st.set_page_config(layout="wide")
st.title("DNNFluid-Car")
st.markdown(
    "[飞桨采用NVIDIA Modulus打造汽车风阻预测模型DNNFluid-Car](https://mp.weixin.qq.com/s/pxmOpfwe0DXCon4uGG93uQ)"
)
st.markdown("<hr />", unsafe_allow_html=True)
os.makedirs(work_dir.as_posix(), exist_ok=True)

col3, col4 = st.columns([1, 5])
with col3:
    # 使用 st.tabs 创建多个选项卡
    tab1, tab2 = st.tabs(["Model Zoo", "Data"])
    # 第一个选项卡：数据展示
    with tab1:
        checkpoint_name = st.selectbox(
            "Network : ", ("AFNO", "UNet", "Geosolver", "FlashBert")
        )
        st.selectbox(
            "Network block : ",
            ("FNO block", "MLP block", "Attention block", "Graph Conv block"),
        )
        st.selectbox(
            "Loss : ",
            ("MSE loss", "L2 loss", "MAE loss", "Pearson correlation coefficient"),
        )
        velocity = st.selectbox("Inlet Velocity", ("120 km/h",))
        velocity_value = re.findall("\d+", velocity)[0]  # 提取数值部分

    # 第二个选项卡：数据加载
    with tab2:
        option = st.radio("Select Data", ["Example", "Upload Geometry"])
        if option == "Example":
            config_path = "../../ppcfd/script/test/"
            data_name = st.selectbox(
                "load demo example : ",
                (
                    "ShapeNetCar(coding)",
                    "Ahmed(coding)",
                    "WindSor(coding)",
                    "DrivAerNet",
                    "DrivAerNet++(coding)",
                    "DrivAerML",
                ),
            )
            geometry = None
            if data_name == "DrivAerML":
                pred_data_key = "pred_vel"
                config_name = "transolver_vel.yaml"
            elif data_name in ["ShapeNetCar", "DrivAerNet"]:
                pred_data_key = "pred_pressure"
                config_name = "transolver.yaml"
        elif option == "Upload Geometry":
            config_path = st.text_input("配置文件路径", "../../configs/")
            config_name = st.text_input("配置文件名称", "transolver.yaml")
            flow_speed = st.number_input("Inlet Flow Speed   [m/s] ", value=33.33, placeholder="33.33")
            mass_density = st.number_input("Air Mass Density [kg/m3] ", value=1.169, placeholder="1.169")
            pred_data_key = st.selectbox("预测物理量 : ", ("pressure", "vel", "wss"))
            pred_data_key = f"pred_{pred_data_key}"
            data_name = "uploaded_geometry"
            geometry = st.file_uploader(
                "Input geometry", type=["ply", "stl", "obj"], label_visibility="hidden"
            )

            if geometry:
                geo = Path(geometry.name)
                input_filename = work_dir / f"{geometry.name}"
                file_size = f"{geometry.size / (1024 * 1024):.2f} MB"
                if geometry.size / (1024 * 1024) > 50:
                    large_stl = True
                if geo.suffix == ".ply":
                    with open(input_filename, "wb") as f:
                        f.write(geometry.getbuffer())
                elif geo.suffix == ".stl":
                    with open(input_filename, "wb") as f:
                        f.write(geometry.getbuffer())
                elif geo.suffix == ".obj":
                    with open(input_filename, "wb") as f:
                        f.write(geometry.getbuffer())
                else:
                    raise NotImplementedError
            file_list = [f for f in os.listdir(work_dir) if f.endswith(".ply")]
            history_file = st.selectbox(
                "load history geometry(coding) : ",
                file_list,
            )
        output_filename = f"{data_name}_{pred_data_key}.vtk"

    if geometry:
        _, file_extension = os.path.splitext(geometry.name)
        if file_extension in ['.stl']:
            # 读取 STL 文件的边界框
            reader = vtk.vtkSTLReader()
            reader.SetFileName(str(input_filename))
            reader.Update()
            polydata = reader.GetOutput()
            bounds = polydata.GetBounds()
            
            # 假设原始单位是 mm，目标单位是 m
            current_unit = 'mm'
            target_unit = 'm'
            
            try:
                # 提取边界框数据
                data = [bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5]]
                
                # 调用 process_data 获取转换后的数据和边界信息
                converted_data, (converted_min, converted_max) = process_data(data, current_unit, target_unit)
                
                # 添加到 file_details
                file_details = {
                    "Input File Name": geometry.name,
                    "Input File Type": file_extension,
                    "Input File Size": file_size,
                    "Original Unit": current_unit,
                    "Target Unit": target_unit,
                    "Original Bounds (min, max)": (min(data), max(data)),
                    "Converted Bounds (min, max)": (converted_min, converted_max),
                }
            except Exception as e:
                st.error(f"Error processing data for file details: {e}")
                file_details = {
                    "Input File Name": geometry.name,
                    "Input File Type": file_extension,
                    "Input File Size": file_size,
                }
            
            st.write(file_details)

    predict_pressure = st.button(f"Predict {pred_data_key}")

with col4:
    if predict_pressure:
        st.toast('代理 AI 模型正在预测物理量...', icon="ℹ️")
        std = time.time()
        completed_process = subprocess.run(
            [
                "python",
                "./ppcfd/web/predict.py",
                "--config-path",
                config_path,
                "--config-name",
                config_name,
                f"checkpoint=[./checkpoints/Transolver_299,./checkpoints/Transolver_starccm+_299_wss]",
                f"output_filename={output_filename}",
                f"input_filename={input_filename}",
                f"large_stl={large_stl}",
                f"mass_density={mass_density}",
                f"flow_speed={flow_speed}",
                
                "mode=inference",
            ],
            capture_output=True,
            text=True,
            encoding='utf-8',  # 明确指定编码
        )
        # 检查命令是否成功执行
        if completed_process.returncode != 0:
            # 打印stderr中的错误信息
            print("\nSubprocess error:", completed_process.stderr)
            # 重新引发异常
            raise subprocess.CalledProcessError(
                completed_process.returncode,
                completed_process.args,
                output=completed_process.stdout,
                stderr=completed_process.stderr,
            )

        # 如果需要，可以处理成功的输出
        print("Subprocess output:", completed_process.stdout)

        # 读取JSON文件
        with open(predict_result_json.as_posix(), "r") as f:
            predict_result = json.load(f)
        vtk_dir = Path(predict_result["vtk_dir"])
        coefficient = predict_result["coefficient"]
        st.success(f'物理场&物理系数预测成功! 前处理+预测耗时:{(time.time() - std):.1f} 秒', icon="✅")
        st.metric(label="风阻系数Cd", value=f"{coefficient['c_d_pred']:.4f}")

        # apt install libgl1-mesa-glx xvfb
        pv.start_xvfb()
        plotter = pv.Plotter(window_size=[400, 400])

        vtk_file_dir = (vtk_dir / output_filename).as_posix()
        mesh = pv.read(vtk_file_dir)
        mesh = mesh.cell_data_to_point_data()
        mesh[pred_data_key] = mesh.point_data[pred_data_key]

        plotter.add_mesh(mesh, scalars=pred_data_key, cmap="bwr", clim=[-1000, 0])
        plotter.view_isometric()
        plotter.background_color = "white"
        stpyvista(plotter, key="pv_cube")
        with open(vtk_file_dir, "rb") as file:
            btn = st.download_button(
                label="Download Pressure .vtk file",
                data=file.read(),
                file_name=output_filename,
                mime="application/octet-stream",
            )
        
        predict_pressure = False

        # with open(work_dir + "output_wss.vtk", "rb") as file:
        #     btn = st.download_button(
        #         label="Download Wall-Shear-Stress .vtk file",
        #         data=file.read(),
        #         file_name=f"{checkpoint_name}_{velocity}_pressure.vtk",
        #         mime="application/octet-stream",
        #     )
        # predict_pressure = False
    else:
        st.info('暂无预测结果,请点击进行风阻预测', icon="ℹ️")
st.markdown("<hr />", unsafe_allow_html=True)

# python -m stramlit run ./ppcfd/web/viewer.py --server.maxUploadSize 2000
