import pickle

import hydra
import numpy as np
import paddle
import vtk
from omegaconf import DictConfig
from vtk.util.numpy_support import numpy_to_vtk
from vtk.util.numpy_support import vtk_to_numpy


@hydra.main(version_base=None, config_path="./configs", config_name="geomdeeponet.yaml")
def main(cfg: DictConfig):
    model = hydra.utils.instantiate(cfg.model)
    checkpoint = "./output/pqformer_beam/model_ep5000.pdparams"
    model_state_dict = paddle.load(checkpoint)
    model.set_state_dict(model_state_dict)
    stress_scaler_path = "./data/3d_beam/scalers/stress_scaler.pkl"
    with open(stress_scaler_path, "rb") as f:
        stress_scaler = pickle.load(f)
    text_param_path = "./data/3d_beam/input_params.npy"
    text_param = np.load(text_param_path)
    case_id = 17
    vtk_path = f"./data/3d_beam/vtk/vtks/Geom{case_id}.vtk"
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(vtk_path)
    reader.Update()
    data = reader.GetOutput()
    point = vtk_to_numpy(data.GetPoints().GetData())
    branch = text_param[case_id]
    xyzs = np.load("./data/3d_beam/xyzs.npz")
    xyzs_i = xyzs[f"Job_{case_id}"]
    sdf_i = xyzs_i[:, 3:4]
    trunk = np.concatenate([point, sdf_i], axis=-1)
    trunk = paddle.to_tensor(trunk, dtype="float32").unsqueeze(0)
    branch = paddle.to_tensor(branch, dtype="float32").unsqueeze(0)
    output = model((branch, trunk))
    p = output.squeeze().cpu().numpy()
    p_label = np.load("./data/3d_beam/stress_targets.npz")["Job_17"].reshape(-1)
    p = numpy_to_vtk(p)
    p_label = numpy_to_vtk(p_label)
    p.SetName("Predicted Von_Mises Equivalent Stresses")
    p_label.SetName("Label Von_Mises Equivalent Stresses")
    data.GetPointData().AddArray(p)
    data.GetPointData().AddArray(p_label)
    appendFilter = vtk.vtkAppendFilter()
    appendFilter.AddInputData(data)
    appendFilter.Update()
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(f"./3D_Beam_{case_id}.vtk")
    writer.SetInputData(appendFilter.GetOutput())
    writer.Write()


def visual_beam(p_label, p_predict, case_id):
    vtk_path = f"./data/3d_beam/vtk/vtks/Geom{case_id}.vtk"
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(vtk_path)
    reader.Update()
    data = reader.GetOutput()
    point = vtk_to_numpy(data.GetPoints().GetData())
    p_label = numpy_to_vtk(p_label)
    p_predict = numpy_to_vtk(p_predict)
    p_err = numpy_to_vtk(np.abs(p_label - p_predict))
    p_label.SetName("Label")
    p_predict.SetName("Pred")
    p_err.SetName("Abs Err")
    data.GetPointData().AddArray(p_label)
    data.GetPointData().AddArray(p_predict)
    data.GetPointData().AddArray(p_err)
    appendFilter = vtk.vtkAppendFilter()
    appendFilter.AddInputData(data)
    appendFilter.Update()
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(f"./3D_Beam_{case_id}.vtk")
    writer.SetInputData(appendFilter.GetOutput())
    writer.Write()


if __name__ == "__main__":
    main()
