import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# import meshio
import numpy as np
import open3d as o3d
import paddle

from ppcfd.data.base_datamodule import BaseDataModule


class LoadMesh:
    def __init__(self, path, query_points=None, closest_points_to_query=False):
        self.path = path
        self.query_points = query_points
        self.closest_points_to_query = closest_points_to_query

    def index_to_mesh_path(self, index, extension: str = ".ply") -> Path:
        return self.path / ("mesh_" + index + extension)

    def load_mesh(self, mesh_path: Path) -> o3d.geometry.TriangleMesh:
        assert mesh_path.exists(), "Mesh path does not exist"
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        return mesh

    def load_mesh_tri(self, mesh_path: Path) -> o3d.t.geometry.TriangleMesh:
        mesh = self.load_mesh(mesh_path)
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        return mesh

    def vertices_from_mesh(self, mesh: o3d.geometry.TriangleMesh) -> paddle.Tensor:
        return paddle.to_tensor(data=np.asarray(mesh.vertices).astype(np.float32))

    def triangles_from_mesh(self, mesh: o3d.geometry.TriangleMesh) -> paddle.Tensor:
        return paddle.to_tensor(data=np.asarray(mesh.triangles).astype(np.int64))

    def get_triangle_centroids(
        self, vertices: paddle.Tensor, triangles: paddle.Tensor
    ) -> paddle.Tensor:
        A, B, C = (
            vertices[triangles[:, 0]],
            vertices[triangles[:, 1]],
            vertices[triangles[:, 2]],
        )
        centroids = (A + B + C) / 3
        areas = (
            paddle.sqrt(x=paddle.sum(x=paddle.cross(x=B - A, y=C - A) ** 2, axis=1)) / 2
        )
        return centroids, areas

    def compute_df(self, mesh: o3d.t.geometry.TriangleMesh) -> np.ndarray:
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        distance = scene.compute_distance(self.query_points).numpy()
        if self.closest_points_to_query:
            closest_points = scene.compute_closest_points(self.query_points)[
                "points"
            ].numpy()
        else:
            closest_points = None
        return distance, closest_points

    def save_df_closest(self, df_closest_dict, index: str, file_type="npy"):
        if file_type == "pt":
            for k, v in df_closest_dict.items():
                df_closest_dict[k] = paddle.to_tensor(data=v)
            paddle.save(
                obj=df_closest_dict,
                path=os.path.join(self.path, f"df_closest_{index}.pt"),
            )
        elif file_type == "npy":
            np.save(
                os.path.join("./dataset/sdf", f"df_{index}.npy"),
                df_closest_dict["df"],
            )
            np.save(
                os.path.join(
                    "./dataset/sdf", f"closest_{index}.npy"
                ),
                df_closest_dict["closest"],
            )

    def get_df_closest(
        self, mesh: Union[Path, o3d.t.geometry.TriangleMesh], index: str = None
    ):
        assert self.query_points is not None, "query_points does not be None"
        if isinstance(mesh, Path):
            mesh = self.load_mesh_tri(mesh)
        df, closest = self.compute_df(mesh)
        df_closest_dict = {"df": df, "closest": closest}
        if index is not None:
            self.save_df_closest(df_closest_dict, index)
        return paddle.to_tensor(data=df), paddle.to_tensor(data=closest)

    def compute_sdf(self, mesh: Union[Path, o3d.t.geometry.TriangleMesh]) -> np.ndarray:
        if isinstance(mesh, Path):
            mesh = self.load_mesh(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        signed_distance = scene.compute_signed_distance(self.query_points).numpy()
        if self.closest_points_to_query:
            closest_points = scene.compute_closest_points(query_points)
        else:
            closest_points = None
        return signed_distance, closest_points

    def sdf_vertices_closest_from_mesh(
        self, mesh: Union[Path, o3d.t.geometry.TriangleMesh]
    ) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        assert self.query_points is not None, "query_points does not be None"
        if isinstance(mesh, Path):
            mesh = self.load_mesh_tri(mesh)
        sdf, closest_points = self.compute_sdf(mesh)
        vertices = mesh.vertex.positions.numpy()
        return (
            paddle.to_tensor(data=sdf),
            vertices,
            paddle.to_tensor(data=closest_points),
        )


class LoadFile:
    def __init__(self, path):
        self.path = path

    def index_to_file(self, filename: str, extension: str = ".npy") -> Path:
        return self.path / (filename + extension)

    def load_file(
        self, file_path: Union[Path, str], extension: str = ".npy"
    ) -> paddle.Tensor:
        if isinstance(file_path, str):
            file_path = self.index_to_file(file_path, extension)
        assert file_path.exists(), f"File path {file_path} does not exist"
        if extension == ".npy":
            data = paddle.to_tensor(data=np.load(str(file_path)).astype(np.float32))
        elif extension == ".json":
            with open(str(file_path), "r") as json_file:
                data = json.load(json_file)
        return data


class PathDictDataset(paddle.io.Dataset, LoadMesh, LoadFile):
    def __init__(
        self,
        path: str = None,
        query_points=None,
        closest_points_to_query=False,
        indices: Optional[List[str]] = None,
        norms_dict: Optional[Dict[str, Callable]] = {},
        data_keys: Optional[List[str]] = ["info", "pressure", "wss"],
        lazy_loading=True,
        only_save_sdf=False,
        mode="train",
        subsample=0,
        max_in_points=70000,
    ):
        LoadMesh.__init__(self, path, query_points, closest_points_to_query)
        LoadFile.__init__(self, path)
        assert path is not None, "path is None"
        self.path = Path(path)
        self.indices = indices
        self.norms_dict = norms_dict
        self.data_keys = data_keys
        self.lazy_loading = lazy_loading
        self.mode = mode
        self.subsample = subsample
        self.subsample_keys = ["pressure", "wss", "normal", "areas"]
        self.max_in_points = max_in_points
        if only_save_sdf:
            for i in indices:
                self.get_df_closest(self.index_to_mesh_path(i), i)
        if not self.lazy_loading:
            self.all_return_dict = [self.get_item(i) for i in range(len(self.indices))]
        self.inputs_key = ["vertices", 'centroids',"df", "closest_points", "sdf_query_points", 'areas']
        if self.mode == "train":
            self.inputs_key += ['centroids_sampled']
            self.inputs_key += ['areas_sampled']

        self.targets_key = ["pressure", "wss", 'areas', 'normal', "idx"]
        self.others_key = ["info", 'p_mean', 'p_std', 'wss_mean', 'wss_std', "mesh", 'Cd', 'file_name', 'reference_area']
        mean_std_dict_path = self.path.parent / "mean_std_1.paddledict"
        self.mean_std_dict = paddle.load(mean_std_dict_path.as_posix())
        self.mean_std_dict = {k:paddle.to_tensor(self.mean_std_dict[k]) for k in self.mean_std_dict.keys()}

    def get_item(self, index):
        # 根据索引获取文件索引，如果没有指定索引，则使用索引值的四位填充字符串
        file_index = self.indices[int(index)] if self.indices else str(int(index)).zfill(4)
        # 初始化返回字典
        return_dict = {}
        # 定义文件键的映射字典
        file_key_dict = {"pressure": "press", "wss": "wallshearstress"}
        for key in self.data_keys:
            # 根据文件类型设置文件扩展名
            extension = ".json" if key == "info" else ".npy"
            # 根据文件键映射字典获取文件键，如果不存在则直接使用原键
            file_key = file_key_dict[key] if key in file_key_dict else key
            # 加载文件并保存到返回字典中
            return_dict[key] = self.load_file(f"{file_key}_{file_index}", extension)
        try:
            # 尝试加载df文件
            return_dict["df"] = self.load_file(f"df_{file_index}")
            # 如果需要获取最近的点，则加载相应的文件
            if self.closest_points_to_query:
                return_dict["closest_points"] = self.load_file(f"closest_{file_index}")
        except Exception:
            # 捕获异常并打印警告信息
            print(
                "Warning: No 'df' files now, please generate them at first or set 'num_workers=0' for this dataloader."
            )
            # 如果df文件不存在，则生成df文件和最近的点
            return_dict["df"], closest_points = self.get_df_closest(
                self.index_to_mesh_path(file_index)
            )
            # 如果需要获取最近的点，则保存到返回字典中
            if self.closest_points_to_query:
                return_dict["closest_points"] = closest_points
        # 将查询点转换为张量并保存到返回字典中
        return_dict["sdf_query_points"] = paddle.to_tensor(data=self.query_points)
        # 判断是否需要计算法向量
        if "compute_normal" in return_dict["info"]:
            # 获取参考面积
            reference_area = return_dict["info"]["reference_area"]
            areas = self.load_file(f"area_{file_index}")
            centroids = self.load_file(f"centroid_{file_index}")
            triangle_normals = self.load_file(f"normal_{file_index}")
            vertices = centroids
            # 设置顶点为质心
            return_dict["vertices"] = vertices
            # 保存顶点信息到返回字典中
            mesh_test_path = self.path / f"mesh_rec_{file_index}.ply"
            if self.path.name == "test" and os.path.exists(mesh_test_path):
                # mesh = meshio.read(mesh_test_path)
                mesh = None
            # 如果路径为测试路径且网格测试文件存在，则读取网格并保存到返回字典中
                return_dict["mesh"] = mesh
        else:
            mesh = self.load_mesh(self.index_to_mesh_path(file_index))
            vertices = self.vertices_from_mesh(mesh)
            # 加载网格文件并获取顶点和三角形信息
            triangles = self.triangles_from_mesh(mesh)
            centroids, areas = self.get_triangle_centroids(vertices, triangles)
            return_dict["vertices"] = vertices
            mesh.compute_triangle_normals()
            triangle_normals = paddle.to_tensor(
                data=mesh.triangle_normals, dtype="float64"
            ).reshape((-1, 3))
            reference_area = (
                return_dict["info"]["width"] * return_dict["info"]["height"] / 2 * 1e-06
            )
        areas = areas.reshape([-1,1])
        return_dict["Cd"] = np.array([0.0])
        return_dict["areas"] = areas
        return_dict["centroids"] = centroids
        return_dict["file_name"] = f"centroid_{file_index}.npy"
        return_dict["normal"] = triangle_normals
        return_dict["reference_area"] = reference_area
        return_dict["idx"] = int(index)

        # subsample for saving GPU memory
        if self.mode == "train":
            n = centroids.shape[0]
            r = min(self.max_in_points, n)
            idx = np.random.permutation(n)[:r]
            return_dict["centroids_sampled"] = centroids[:: self.subsample, ...][idx, ...]
            return_dict["areas_sampled"] = areas[:: self.subsample, ...][idx]
            for key in self.subsample_keys:
                return_dict[key] = return_dict[key][:: self.subsample, ...][idx]

        for key in self.norms_dict:
            if key in return_dict:
                return_dict[key] = self.norms_dict[key](return_dict[key])
        if "location" in self.norms_dict:
            if return_dict["vertices"] is not None:
                return_dict["vertices"] = self.norms_dict["location"](vertices)
            return_dict["centroids"] = self.norms_dict["location"](
                return_dict["centroids"]
            )
            return_dict["sdf_query_points"] = self.norms_dict["location"](
                return_dict["sdf_query_points"]
            ).transpose(perm=[3, 0, 1, 2])
            if self.closest_points_to_query:
                return_dict["closest_points"] = self.norms_dict["location"](
                    return_dict["closest_points"]
                ).transpose(perm=[3, 0, 1, 2])
        return {
            "inputs": [return_dict[k] for k in self.inputs_key],
            "targets": [return_dict[k] for k in self.targets_key]
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.lazy_loading:
            return self.get_item(index)
        else:
            return self.all_return_dict[index]

    def data_to_dict(self, data):
        inputs = {k: data["inputs"][i] for i, k in enumerate(self.inputs_key)}
        targets = {k: data["targets"][i] for i, k in enumerate(self.targets_key)}
        file_name_list = [self.indices[i] for i in targets["idx"]]
        Cd_list = [float(1.0) for f in file_name_list]
        FA_list = [float(1.0) for f in file_name_list]
        others = {
            "file_name": file_name_list,
            "Cd": np.array(Cd_list).reshape([-1, 1, 1]),             # B N C
            "reference_area": np.array(FA_list).reshape([-1, 1, 1]), # B N C
            **self.mean_std_dict
        }
        return inputs, targets, others


class BaseCFDDataModule(BaseDataModule):
    def __init__(self):
        super().__init__()

    def encode(self, norm_fn, data: paddle.Tensor) -> paddle.Tensor:
        # norm_fn.to(data.place)
        return norm_fn.encode(data)

    def decode(self, norm_fn, data: paddle.Tensor) -> paddle.Tensor:
        return norm_fn.decode(data)

    def load_bound(
        self, data_path, filename="watertight_global_bounds.txt", eps=1e-06
    ) -> Tuple[List[float], List[float]]:
        with open(data_path / filename, "r") as fp:
            min_bounds = fp.readline().split(" ")
            max_bounds = fp.readline().split(" ")
            min_bounds = [(float(a) - eps) for a in min_bounds]
            max_bounds = [(float(a) + eps) for a in max_bounds]
        return min_bounds, max_bounds

    def location_normalization(
        self,
        locations: paddle.Tensor,
        min_bounds: Union[paddle.Tensor, List[float]],
        max_bounds: Union[paddle.Tensor, List[float]],
    ) -> paddle.Tensor:
        """
        Normalize locations to [-1, 1].
        """
        if not isinstance(min_bounds, paddle.Tensor):
            min_bounds = paddle.to_tensor(data=min_bounds)
        if not isinstance(max_bounds, paddle.Tensor):
            max_bounds = paddle.to_tensor(data=max_bounds)
        # print("\nlocations", locations)
        locations = (locations - min_bounds) / (max_bounds - min_bounds)
        locations = 2 * locations - 1
        return locations

    def info_normalization(
        self, info: dict, min_bounds: List[float], max_bounds: List[float]
    ) -> dict:
        """
        Normalize info to [0, 1].
        """
        for i, (k, v) in enumerate(info.items()):
            info[k] = (v - min_bounds[i]) / (max_bounds[i] - min_bounds[i])
        return info

    def area_normalization(
        self, area: paddle.Tensor, min_bounds: float, max_bounds: float
    ) -> paddle.Tensor:
        """
        Normalize info to [0, 1].
        """
        return (area - min_bounds) / (max_bounds - min_bounds)


class DrivAerDataModule(BaseCFDDataModule):
    def __init__(
        self,
        data_path,
        out_keys: List[str] = ["pressure"],
        out_channels: List[int] = [1],
        n_train_num: int = 1,
        n_val_num: int = 1,
        n_test_num: int = 1,
        spatial_resolution: Tuple[int, int, int] = None,
        query_points=None,
        closest_points_to_query=True,
        eps=0.01,
        only_save_sdf=False,
        lazy_loading=True,
        subsample_train=1,
        max_in_points=4000,
    ):
        super().__init__()
        if isinstance(data_path, str):
            data_path = Path(data_path)
        data_path = data_path.expanduser()
        assert data_path.exists(), "Path does not exist"
        assert data_path.is_dir(), "Path is not a directory"
        self.data_path = data_path
        self.out_keys = out_keys
        self.out_channels = out_channels
        self.query_points = query_points
        self.closest_points_to_query = closest_points_to_query
        self.spatial_resolution = spatial_resolution
        self.eps = eps
        self.only_save_sdf = only_save_sdf
        self.lazy_loading = lazy_loading
        self.subsample_train = subsample_train
        self.max_in_points = max_in_points
        self.get_indices(n_train_num, n_val_num, n_test_num)
        self.get_norms(data_path)
        self.get_data()
        assert (
            not only_save_sdf
        ), "Onle save sdf and exit when only_save_sdf is True, no training."

    def load_ids(self, idx_path: str):
        idx_str_lst = []
        with open(idx_path, "r") as file:
            line = file.readline()
            while line:
                line = line.strip()
                idx_str_lst.append(line.split("_")[-1])
                line = file.readline()
        return idx_str_lst

    def init_idx(self, n_data, filename):
        idx_path = self.data_path / filename
        if idx_path.exists():
            indices = self.load_ids(idx_path)
            indices.sort()
            assert n_data <= len(
                indices
            ), f"only {len(indices)} meshes are available, but {n_data} are requested."
            indices = indices[:n_data]
        else:
            indices = [str(i).zfill(4) for i in range(1, n_data + 1)]
        return indices

    def init_data(self, indices, mode="train"):
        data_keys = ["info"]
        data_keys.extend(self.out_keys)
        data_keys.extend(["pressure", "wss"])
        if mode == "train":
            subsample = self.subsample_train
        else:
            subsample = 0
        data_dict = PathDictDataset(
            path=self.data_path / mode,
            query_points=self.query_points,
            closest_points_to_query=self.closest_points_to_query,
            indices=indices,
            norms_dict=self.norms_dict,
            data_keys=data_keys,
            only_save_sdf=self.only_save_sdf,
            lazy_loading=self.lazy_loading,
            mode=mode,
            subsample=subsample,
            max_in_points=self.max_in_points,
        )
        return data_dict

    def get_indices(self, n_train_num, n_val_num, n_test_num):
        self.train_indices = self.init_idx(n_train_num, "train_design_ids.txt")
        self.val_indices = self.init_idx(n_val_num, "val_design_ids.txt")
        self.test_indices = self.init_idx(n_test_num, "test_design_ids.txt")
        print("5 self.train_indices", self.train_indices[:5])
        print("5 self.test_indices", self.test_indices[:5])

    def get_norms(self, data_path):
        min_bounds, max_bounds = self.load_bound(
            data_path, filename="global_bounds.txt", eps=self.eps
        )
        min_info_bounds, max_info_bounds = self.load_bound(
            data_path, filename="info_bounds.txt", eps=0.0
        )
        min_area_bound, max_area_bound = self.load_bound(
            data_path, filename="area_bounds.txt", eps=0.0
        )
        if self.query_points is None:
            assert (
                self.spatial_resolution is not None
            ), "spatial_resolution must be given"
            tx = np.linspace(min_bounds[0], max_bounds[0], self.spatial_resolution[0])
            ty = np.linspace(min_bounds[1], max_bounds[1], self.spatial_resolution[1])
            tz = np.linspace(min_bounds[2], max_bounds[2], self.spatial_resolution[2])
            self.query_points = np.stack(
                np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1
            ).astype(np.float32)
        location_norm_fn = lambda x: self.location_normalization(
            x, min_bounds, max_bounds
        )
        info_norm_fn = lambda x: self.info_normalization(
            x, min_info_bounds, max_info_bounds
        )
        area_norm_fn = lambda x: self.area_normalization(
            x, min_area_bound[0], max_area_bound[0]
        )
        self.norms_dict = {"location": location_norm_fn, "area": area_norm_fn}

    def get_data(self):
        self.train_data = self.init_data(self.train_indices, "train")
        self.val_data = self.init_data(self.val_indices, "val")
        self.test_data = self.init_data(self.test_indices, "test")
        self._aggregatable = [
            "df",
            "sdf_query_points",
            "pressure",
            "normal",
            "closest_points",
            "vertices",
            "areas",
            "wss",
            "centroids",
        ]

    def load_file(self, file_path: Path) -> np.ndarray:
        assert file_path.exists(), f"File {file_path} does not exist"
        data = np.load(file_path).astype(np.float32)
        return data

    def decode(self, data, idx: int) -> paddle.Tensor:
        return super(DrivAerDataModule, self).decode(
            self.output_normalization[idx], data
        )

    # def collate_fn(self, batch):
    #     aggr_dict = {}
    #     for key in self._aggregatable:
    #         # Skip stacking for keys that have inconsistent shapes
    #         if key in ['pressure', 'wss', 'normal', 'areas', 'centroids']:
    #             aggr_dict.update({key: [data_dict[key] for data_dict in batch]})
    #         else:
    #             try:
    #                 aggr_dict.update(
    #                     {key: paddle.stack(x=[data_dict[key] for data_dict in batch])}
    #                 )
    #             except ValueError:
    #                 # If shapes are inconsistent, keep as list
    #                 aggr_dict.update({key: [data_dict[key] for data_dict in batch]})
    #     remaining = list(set(batch[0].keys()) - set(self._aggregatable))
    #     for key in remaining:
    #         aggr_dict.update({key: [data_dict[key] for data_dict in batch]})
    #     return aggr_dict
