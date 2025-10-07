# 加入mm与m的判断并自动转换
# @zhuhy根据整车数据做出的修改版本，对应了viewer.py
# 新增计时功能，可以看每个文件的处理时间
import json
import os
import sys
import time
from pathlib import Path
import logging

import hydra
import numpy as np
import paddle
import vtk

sys.path.append(".")
from ppcfd.utils.loss import LpLoss

sys.path.append("./ppcfd/script/starccm+/")

from main_v2 import Car_Loss
from main_v2 import load_checkpoint
from ppcfd.data.pointcloud_datamodule import load_mean_std
from ppcfd.data.pointcloud_datamodule import normlalize_input as normlalize_input_points
from ppcfd.data.starccm_datamodule import read_ply
from ppcfd.data.starccm_datamodule import read_stl
from ppcfd.data.starccm_datamodule import read_obj
from ppcfd.data.starccm_datamodule import get_areas
from ppcfd.data.starccm_datamodule import base_data_to_dict as data_to_dict
from ppcfd.data.starccm_datamodule import get_centroids
from ppcfd.data.starccm_datamodule import get_normals
from ppcfd.data.starccm_datamodule import normlalize_input as normlalize_input_star
from ppcfd.data.starccm_datamodule import write
from ppcfd.script.starccm_plus.pyfrontal import calculate_frontal_area

log = logging.getLogger(__name__)


def convert_mm_to_m(polydata):
    """
    将几何体的单位从毫米 (mm) 转换为米 (m)，通过缩放几何体。
    """
    transform = vtk.vtkTransform()
    transform.Scale(0.001, 0.001, 0.001)  # 缩放因子：1 mm = 0.001 m

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(polydata)
    transform_filter.SetTransform(transform)
    transform_filter.Update()

    return transform_filter.GetOutput()


def detect_and_convert_units(polydata):
    """
    检测几何体的单位，并在必要时从 mm 转换为 m。
    """
    if not polydata:
        raise ValueError("Input polydata is empty or invalid.")
    bounds = polydata.GetBounds()
    x_length = abs(bounds[1] - bounds[0])  # x方向的长度
    print(f"Detected x length: {x_length}")

    # 如果 x 长度大于 10，则假设单位为 mm，并转换为 m
    if x_length > 10:
        print("Detected unit as mm. Converting to meters...")
        polydata = convert_mm_to_m(polydata)
        # 重新计算边界框，因为几何体已被缩放
        bounds = polydata.GetBounds()
        print(f"Converted bounds: {bounds}")

    return polydata


def read_geometry(file_path, large_stl=False):
    # if large_stl:
    #     ms = pymeshlab.MeshSet()
    #     ms.load_new_mesh(file_path.as_posix())
    #     ms.apply_filter(
    #         "meshing_decimation_quadric_edge_collapse",
    #         targetfacenum=100000,
    #         preservenormal=True,
    #     )
    #     # ms.apply_filter('meshing_decimation_clustering', threshold = pymeshlab.PercentageValue(0.182621))
    #     file_path = file_path.with_stem(file_path.stem + "_simplified")
    #     ms.save_current_mesh(file_path.as_posix())

    if file_path.suffix == ".ply":
        reader, polydata = read_ply(file_path)
    elif file_path.suffix == ".stl":
        reader, polydata = read_stl(file_path)
    elif file_path.suffix == ".obj":
        reader, polydata = read_obj(file_path)
    else:
        raise ValueError("Unsupported geometry format")
    polydata = detect_and_convert_units(polydata)  # 检测并转换单位
    return None, polydata


def get_bounds_info(polydata):
    bounds = polydata.GetBounds()
    return {
        "x_min": bounds[0],
        "x_max": bounds[1],
        "y_min": bounds[2],
        "y_max": bounds[3],
        "z_min": bounds[4],
        "z_max": bounds[5],
    }


@hydra.main(
    version_base=None,
    config_path="../../configs/",
    config_name="transolver.yaml",
)
def inference(config):
    start_time = time.time()
    output_dir = Path(config.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    def construct_inference_dataset(config):
        
        if config.data_module._target_ == "ppcfd.data.StarCCMDataModule":
            normlalize_input = normlalize_input_star
        elif config.data_module._target_ == "ppcfd.data.PointCloudDataModule":
            normlalize_input = normlalize_input_points
        else:
            raise NotImplementedError
        targets, inputs, others = {}, {}, {}
        polydata = None
        if config.mode == "test":
            inputs["centroids"] = np.load("./data_drivaer/test/centroid_0014.npy").astype(
                np.float32
            )
            targets["areas"] = np.load("./data_drivaer/test/area_0014.npy").reshape(
                [1, -1, 1]
            )
            targets["normal"] = np.load("./data_drivaer/test/normal_0014.npy").reshape(
                [1, -1, 3]
            )
            targets["pressure"] = np.load("./data_drivaer/test/press_0014.npy").reshape(
                [1, -1]
            )
            targets["pressure"] = paddle.to_tensor(targets["pressure"])
            targets["wss"] = np.load(
                "./data_drivaer/test/wallshearstress_0014.npy"
            ).reshape([1, -1, 3])
            targets["wss"] = paddle.to_tensor(targets["wss"])
            targets["reference_area"] = 3.0
        elif config.mode == "inference":
            input_filename = Path(config.input_filename)
            _, polydata = read_geometry(input_filename, config.large_stl)
            inputs["centroids"] = get_centroids(polydata)
            targets["areas"] = get_areas(polydata).reshape([1, -1, 1])
            targets["normal"] = get_normals(polydata).reshape([1, -1, 3])
            targets["pressure"] = None
            targets["wss"] = None
            targets["reference_area"] = calculate_frontal_area(
                    file_name=input_filename.as_posix(),
                    vtk_data=polydata,
                    proj_axis="X",
                    debug=False,
                )[0]
        else:
            raise ValueError("mode must be test or inference")

        mean_std_dict_dir = Path(config.data_module.data_dir) / "mean_std.paddledict"
        mean_std_dict = load_mean_std(mean_std_dict_dir)
        for k, v in mean_std_dict.items():
            if not isinstance(v, paddle.Tensor):
                mean_std_dict[k] = paddle.to_tensor(v)
        others = {**others, **mean_std_dict}
        others = {k: paddle.to_tensor(others[k]) for k in others.keys()}
        inputs = normlalize_input(
            others,
            inputs["centroids"],
            sampled_indices=np.arange(inputs["centroids"].shape[0]),
        )
        # dict_keys(['areas', 'normal', 'pressure'])
        # targets_key ['areas', 'normal', 'pressure', 'wss', 'reference_area']
        data = {
            "targets": [v for v in targets.values()],
            "inputs": [v for v in inputs.values()],
        }
        return data, polydata, mean_std_dict
    
    data, polydata, mean_std_dict = construct_inference_dataset(config)
    car_loss = Car_Loss(config)
    loss_fn = LpLoss(size_average=True)

    def model_fn(config):
        model = hydra.utils.instantiate(config.model)
        load_checkpoint(config, model)
        with paddle.no_grad():
            outputs = model(data["inputs"])
            inputs, targets, others = data_to_dict(
                data,
                ["centroids", "local_centroid"],
                ["areas", "normal", "pressure", "wss", "reference_area"],
                file_list=[config.input_filename],
                mean_std_dict=mean_std_dict,
                )
        if config.mode == "inference":
            cx, pred = car_loss(inputs, outputs, targets, others, loss_fn, None)
            return cx, targets, pred
        elif config.mode == "test": # for debug
            loss_list = car_loss(inputs, outputs, targets, others, loss_fn, None)
            log.info(f"l2 loss {loss_list[0].item():.2f}, cp {loss_list[1]:.2f}")
            log.info(f"Processing file {config.input_filename} took {time.time()-start_time:.2f} seconds.")
            exit()
        else:
            raise ValueError("mode must be test or inference")

    # infer pressure
    checkpoint_list = config.checkpoint
    if checkpoint_list is None or len(checkpoint_list) < 2:
        raise ValueError("Please provide two checkpoint paths in config for pressure and wss prediction")
    config.checkpoint = checkpoint_list[0]
    config.model.out_dim = 1
    config.out_channels = [1]
    config.out_keys = ["pressure"]
    cx, true, pred = model_fn(config)
    cp = cx.c_p_pred.item()
    write(
        polydata,
        pred.p[0],
        f"pred_{config.out_keys[0]}",
        (output_dir / config.output_filename).as_posix(),
    )

    # infer wss
    config.checkpoint = checkpoint_list[1]
    config.model.out_dim = 3
    config.out_channels = [3]
    config.out_keys = ["wss"]
    cx, true, pred= model_fn(config)
    cf = cx.c_f_pred.item()
    write(
        polydata,
        pred.wss[0],
        f"pred_{config.out_keys[0]}",
        (output_dir / config.output_filename).as_posix(),
    )

    log.info(f"Processing file {config.input_filename} took {time.time()-start_time:.2f} seconds.")

    data = {
        "vtk_dir": output_dir.as_posix(),
        "coefficient": {
            "c_p_pred": cp,
            "c_f_pred": cf,
            "c_d_pred": cp + cf,
        },
    }
    json_data = json.dumps(data, indent=4)
    txt_dir = Path("./output/predict_result.json")
    with open(txt_dir.as_posix(), "w") as f:
        f.write(json_data)


inference()

# python ppcfd/web/predict.py --config-path=../../ppcfd/script/inference/ --config-name=transolver.yaml input_filename=/mnt/cfd/leiyao/workspace/DNNFluid-Car_web_viewer_fix/output/DrivAer_F_D_WM_WW_0001.stl output_filename=test_pred.vtk
# python ppcfd/web/predict.py --config-path=../../ppcfd/script/inference/ --config-name=transolver.yaml input_filename=./data_drivaer/test/mesh_rec_0001.ply output_filename=test_pred.vtk
# python ppcfd/web/predict.py --config-path=../../ppcfd/script/inference/ --config-name=transolver.yaml input_filename=/workspace/DNNFluid_Car/DNNFluid-Car_0519/data_fake_starccm+/test.case.ply output_filename=test_pred.vtk flow_speed=33.33 mass_density=1.169 large_stl=True
