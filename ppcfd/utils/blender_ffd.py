from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

working_dir = r"/Users/wangguan12/Desktop/CAR/"
input_file = working_dir + r"/0618_13cars/Arteon_2021_body_500_50000.stl"


print(input_file)


# 删除所有物体
for obj in bpy.data.objects:
    if (
        obj.type == "MESH"
        or obj.type == "CURVE"
        or obj.type == "SURFACE"
        or obj.type == "FONT"
        or obj.type == "ARMATURE"
        or obj.type == "LATTICE"
        or obj.type == "EMPTY"
        or obj.type == "CAMERA"
        or obj.type == "LAMP"
        or obj.type == "SPEAKER"
    ):
        bpy.data.objects.remove(obj)
# 导入obj
# bpy.ops.import_mesh.ply(filepath=input_file)
bpy.ops.import_mesh.stl(filepath=input_file)
# bpy.ops.import_scene.obj(filepath=input_file, axis_forward='Y', axis_up='Z')
cube = bpy.data.objects.get(Path(input_file).stem)
cube.name = "car"

# print('导入obj的长宽高, ', cube.dimensions)
lattice_data = bpy.data.lattices.new("LatticeData")

# 设置lattice的分辨率（例如：U, V, W方向上的分段数）
lattice_data.points_u = 32
lattice_data.points_v = 4
lattice_data.points_w = 8

bbox_corners = [Vector(corner) for corner in cube.bound_box]
# 计算局部坐标系中的最小和最大坐标
min_box = min(bbox_corners, key=lambda c: (c.x, c.y, c.z))
max_box = max(bbox_corners, key=lambda c: (c.x, c.y, c.z))

# 转换到世界坐标系
min_box = cube.matrix_world @ min_box
max_box = cube.matrix_world @ max_box

points = []
for point in lattice_data.points:
    points.append([point.co_deform.x, point.co_deform.y, point.co_deform.z])

points = np.array(points)
x = points[:, 0]
y = points[:, 2]

# index
x_min = x.min()
y_min = y.min()
x_max = x.max()
y_max = y.max()

# 1. ramp_angle_index
ramp_z_std = 1.5
ramp_x_std = 1.5
index_1 = np.where((x <= x_min + ramp_x_std + 0) & (y <= y_min + ramp_z_std + 0))[0]
index_2 = np.where(
    (x > x_min + ramp_x_std + 0)
    & (x <= x_min + ramp_x_std + 1)
    & (y <= y_min + ramp_z_std)
)[0]
index_3 = np.where(
    (x > x_min + ramp_x_std + 1)
    & (x <= x_min + ramp_x_std + 2)
    & (y <= y_min + ramp_z_std)
)[0]
ramp_angle_index = [index_1, index_2, index_3]

# 2. front_bumper_index
index_1 = np.where((x <= x_min + 3))[0]
index_2 = np.where((x <= x_min + 2))[0]
index_3 = np.where((x <= x_min + 1))[0]
front_bumper_index = [index_1, index_2, index_3]

# 3. diffusor_angle_index
ramp_z_std = 1.5
ramp_x_std = 1.5
index_1 = np.where((x >= x_max - ramp_x_std - 0) & (y <= y_min + ramp_z_std + 0))[0]
index_2 = np.where(
    (x < x_max - ramp_x_std - 0)
    & (x >= x_max - ramp_x_std - 1)
    & (y <= y_min + ramp_z_std)
)[0]
index_3 = np.where(
    (x < x_max - ramp_x_std - 1)
    & (x >= x_max - ramp_x_std - 2)
    & (y <= y_min + ramp_z_std)
)[0]
diffusor_angle_index = [index_1, index_2, index_3]

# 4. trunklid_length_index
index_1 = np.where((x >= x_max - 3))[0]
index_2 = np.where((x >= x_max - 2))[0]
index_3 = np.where((x >= x_max - 1))[0]
trunklid_length_index = [index_1, index_2, index_3]

# 5. trunklid_angle_index
ramp_z_std = 3
ramp_x_std = 1.5
index_1 = np.where((x >= x_max - ramp_x_std - 0) & (y >= y_max - ramp_z_std))[0]
index_2 = np.where(
    (x < x_max - ramp_x_std - 0)
    & (x >= x_max - ramp_x_std - 1)
    & (y >= y_max - ramp_z_std)
)[0]
index_3 = np.where(
    (x < x_max - ramp_x_std - 1)
    & (x >= x_max - ramp_x_std - 2)
    & (y >= y_max - ramp_z_std)
)[0]
trunklid_angle_index = [index_1, index_2, index_3]


def ramp_angle(points, move_z):
    for j in ramp_angle_index[0]:
        point = lattice_data.points[j]
        point.co_deform.z += move_z

    for j in ramp_angle_index[1]:
        point = lattice_data.points[j]
        point.co_deform.z += move_z * 2 / 3

    for j in ramp_angle_index[2]:
        point = lattice_data.points[j]
        point.co_deform.z += move_z * 1 / 3


def front_bumper_length(points, move_x):
    for j in front_bumper_index[0]:
        point = lattice_data.points[j]
        point.co_deform.x += move_x * 7 / 10

    for j in front_bumper_index[1]:
        point = lattice_data.points[j]
        point.co_deform.x += move_x * 9 / 10

    for j in front_bumper_index[2]:
        point = lattice_data.points[j]
        point.co_deform.x += move_x


def diffusor_angle(points, move_z):
    for j in diffusor_angle_index[0]:
        point = lattice_data.points[j]
        point.co_deform.z += move_z

    for j in diffusor_angle_index[1]:
        point = lattice_data.points[j]
        point.co_deform.z += move_z * 2 / 3

    for j in diffusor_angle_index[2]:
        point = lattice_data.points[j]
        point.co_deform.z += move_z * 1 / 3


def trunklid_length(points, move_x):
    for j in trunklid_length_index[0]:
        point = lattice_data.points[j]
        point.co_deform.x += move_x * 7 / 10

    for j in trunklid_length_index[1]:
        point = lattice_data.points[j]
        point.co_deform.x += move_x * 9 / 10

    for j in trunklid_length_index[2]:
        point = lattice_data.points[j]
        point.co_deform.x += move_x


def trunklid_angle(points, move_z):
    for j in trunklid_angle_index[0]:
        point = lattice_data.points[j]
        point.co_deform.z += move_z

    for j in trunklid_angle_index[1]:
        point = lattice_data.points[j]
        point.co_deform.z += move_z * 2 / 3

    for j in trunklid_angle_index[2]:
        point = lattice_data.points[j]
        point.co_deform.z += move_z * 1 / 3


# 创建一个新的对象，使用上面创建的 Lattice 数据
lattice_object = bpy.data.objects.new("MyLatticeObject", lattice_data)

# 移动格子包围住几何
lattice_object.location.x = (min_box[0] + max_box[0]) / 2
lattice_object.location.y = (min_box[1] + max_box[1]) / 2
lattice_object.location.z = (min_box[2] + max_box[2]) / 2
lattice_object.scale.x = cube.dimensions.x / (lattice_data.points_u - 1) * 1.1
lattice_object.scale.y = cube.dimensions.y / (lattice_data.points_v - 1) * 1.1
lattice_object.scale.z = cube.dimensions.z / (lattice_data.points_w - 1) * 1.1

# 获取当前场景
scene = bpy.context.scene

# 将新创建的对象添加到当前场景
scene.collection.objects.link(lattice_object)

# 创建 Lattice 修改器并应用到立方体
lattice_modifier = cube.modifiers.new(name="LatticeModifier", type="LATTICE")
lattice_modifier.object = lattice_object
lattice_modifier.strength = 1.0


input_csv = working_dir + r"/lhs_parameters.csv"
input_params = np.loadtxt(input_csv, delimiter=",")[:10]


def FFD(params):
    trunklid_angle(points, params[0])
    ramp_angle(points, params[1])
    diffusor_angle(points, params[2])
    front_bumper_length(points, params[3])
    trunklid_length(points, params[4])


def write_FFD(i, params):
    FFD(params)
    # 更新场景，确保一切正常
    out_file = rf"/Users/wangguan12/Desktop/CAR/test/DOE/mesh_{str(i).zfill(4)}.stl"
    bpy.ops.export_mesh.stl(filepath=out_file)
    bpy.context.view_layer.update()
    FFD(-params)


for i, params in enumerate(input_params):
    write_FFD(i, params)
