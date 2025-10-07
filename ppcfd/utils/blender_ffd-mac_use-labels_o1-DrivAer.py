import bpy
import bmesh
import numpy as np
from pathlib import Path
import os
import csv
from mathutils import Vector

# 全局变量
WORKING_DIR = "/Users/lkuang/MyOneDrive/OneDrive - NVIDIA Corporation/1-TAD/2-Modulus/3-Collabs/C-SAIC/DOE-ex/"
INPUT_FILE = WORKING_DIR + "drivaer_1.stl"
LABEL_FILE = WORKING_DIR + "Drivaer_Labeling.csv"
OUTPUT_FILE = WORKING_DIR + "output_drivaer_1.log"
INPUT_CSV = WORKING_DIR + "lhs_parameters.csv"

def delete_all_objects():
    """删除所有物体"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def import_stl_file(file_path):
    """导入 STL 文件"""
    bpy.ops.import_mesh.stl(filepath=str(file_path))
    car_obj = bpy.context.active_object
    car_obj.name = "car"
    return car_obj

def setup_log_file():
    """设置日志文件"""
    return open(OUTPUT_FILE, 'a', encoding='utf-8')

def setup_lattice_data():
    """设置晶格数据"""
    lattice_data = bpy.data.lattices.new("LatticeData")
    lattice_data.points_u = 32
    lattice_data.points_v = 8
    lattice_data.points_w = 8
    return lattice_data

def read_labels(label_file_path):
    """读取标签文件"""
    labels = {}
    try:
        with open(label_file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                vertex_index = int(row[0])
                label = row[3]
                labels[vertex_index] = label
    except FileNotFoundError:
        print(f"标签文件 {label_file_path} 未找到。")
    return labels

def mark_faces_with_label(obj, face_labels, target_label='40', colori=1):
    """优化后的标记函数"""
    mesh = obj.data
    faces_to_mark = [
        face.index for face in mesh.polygons 
        if face.index in face_labels and face_labels[face.index] == str(target_label)
    ]
    marked_count = len(faces_to_mark)
    return marked_count, faces_to_mark

def extract_face_center_coordinates(obj, face_indices):
    """提取部分面的面心坐标"""
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    face_centers = []
    for index in face_indices:
        if index < len(bm.faces):
            face = bm.faces[index]
            center = sum((v.co for v in face.verts), Vector()) / len(face.verts)
            face_centers.append(center)
    bm.free()
    return face_centers

def display_face_centers(face_centers):
    """通过黄色小圆点展示面心坐标"""
    yellow_material = bpy.data.materials.new(name="YellowMaterial")
    yellow_material.diffuse_color = (1, 1, 0, 1)
    for center in face_centers:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.025, location=center)
        sphere = bpy.context.active_object
        if len(sphere.data.materials) == 0:
            sphere.data.materials.append(yellow_material)
        else:
            sphere.data.materials[0] = yellow_material

def lattice_display(lattice_data, cube):
    """显示晶格"""
    lattice_object = bpy.data.objects.new("MyLatticeObject", lattice_data)
    bbox_corners = [Vector(corner) for corner in cube.bound_box]
    min_box = min(bbox_corners, key=lambda c: (c.x, c.y, c.z))
    max_box = max(bbox_corners, key=lambda c: (c.x, c.y, c.z))
    min_box = cube.matrix_world @ min_box
    max_box = cube.matrix_world @ max_box
    lattice_object.location.x = (min_box[0] + max_box[0]) / 2
    lattice_object.location.y = (min_box[1] + max_box[1]) / 2
    lattice_object.location.z = (min_box[2] + max_box[2]) / 2
    lattice_object.scale.x = cube.dimensions.x / (lattice_data.points_u - 1) * 1.1
    lattice_object.scale.y = cube.dimensions.y / (lattice_data.points_v - 1) * 1.1
    lattice_object.scale.z = cube.dimensions.z / (lattice_data.points_w - 1) * 1.1
    scene = bpy.context.scene
    scene.collection.objects.link(lattice_object)
    lattice_modifier = cube.modifiers.new(name="LatticeModifier", type="LATTICE")
    lattice_modifier.object = lattice_object
    lattice_modifier.strength = 1.0
    return lattice_object

def get_lattice_points(lattice_obj):
    """获取晶格点坐标和索引"""
    lattice = lattice_obj.data
    points = []
    for index, point in enumerate(lattice.points):
        point_co = lattice_obj.matrix_world @ point.co
        points.append((index, point_co))
    return points

def find_lattice_points_in_bounding_box(face_centers, lattice_points, lattice_object):
    """找出能包围某个 label 的面元的最小立方格中所有晶格点"""
    if not face_centers:
        return []
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')
    min_z = float('inf')
    max_z = float('-inf')
    for center in face_centers:
        x, y, z = center.x, center.y, center.z
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        min_z = min(min_z, z)
        max_z = max(max_z, z)
    min_x -= lattice_object.scale.x
    max_x += lattice_object.scale.x
    min_y -= lattice_object.scale.y
    max_y += lattice_object.scale.y
    min_z -= lattice_object.scale.z
    max_z += lattice_object.scale.z
    inside_points_indices = []
    for index, point_co in lattice_points:
        x = point_co.x * lattice_object.scale.x + lattice_object.location.x
        y = point_co.y * lattice_object.scale.y + lattice_object.location.y
        z = point_co.z * lattice_object.scale.z + lattice_object.location.z
        if min_x < x < max_x and min_y < y < max_y and min_z < z < max_z:
            inside_points_indices.append(index)
    return inside_points_indices

def auto_bumper_move(lattice_data, associated_indices, move_z):
    """#1 自动移动保险杠对应的晶格点"""
    for j in associated_indices:
        point = lattice_data.points[j]
        point.co_deform.x += move_z * 0.25

def auto_top_move(lattice_data, associated_indices, move_z):
    """#2 自动移动顶部对应的晶格点"""
    for j in associated_indices:
        point = lattice_data.points[j]
        point.co_deform.z += move_z * 0.25

def auto_spoiler_move(lattice_data, associated_indices, move_z):
    """#3 自动移动Spoiler对应的晶格点"""
    for j in associated_indices:
        point = lattice_data.points[j]
        point.co_deform.x += move_z * 0.25

def show_lattices_points(lattice_object, lattice_data, lattices):
    """显示晶格点"""
    world_point = lattice_object.matrix_world @ lattice_data.points[0].co_deform
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=world_point)
    for j in lattices:
        point = lattice_data.points[j]
        world_point = lattice_object.matrix_world @ point.co_deform
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02, location=world_point)

def write_FFD(i, working_dir):
    """导出 STL 文件"""
    out_file = working_dir + f"DrivAer_00_mph_{str(i).zfill(4)}.stl"
    bpy.ops.export_mesh.stl(filepath=str(out_file))
    bpy.context.view_layer.update()

def main():
    # 删除所有物体
    delete_all_objects()

    # 导入 STL 文件
    car_obj = import_stl_file(INPUT_FILE)

    # 设置日志文件
    file = setup_log_file()
    file.write(f'导入obj的长宽高, {car_obj.dimensions} \n')

    # 设置晶格数据
    lattice_data = setup_lattice_data()

    # 读取标签文件
    face_labels = read_labels(LABEL_FILE)


    def find_associated(label):
        # 标记面片
        count, face_indices = mark_faces_with_label(car_obj, face_labels, label, 1)
        file.write(f"成功标记 {count} 个面片\n")
        # 提取部分面的面心坐标
        face_centers = extract_face_center_coordinates(car_obj, face_indices)
        # 显示面心坐标 For Debugging
        scale = count // 100
        display_face_centers(face_centers[::scale])
        # 显示晶格
        lattice_object = lattice_display(lattice_data, car_obj)
        # 获取晶格点坐标和索引
        lattice_points = get_lattice_points(lattice_object)
        # 找出能包围某个 label 的面元的最小立方格中所有晶格点
        associated_indices = find_lattice_points_in_bounding_box(face_centers, lattice_points, lattice_object)
        # 显示晶格点 For Debugging
        show_lattices_points(lattice_object, lattice_data, associated_indices)        
        return associated_indices

        # 自动移动晶格点
    # '33' Spoiler; '32' Roof; "2" front Bumper; "44","45" tire
    label_list = ['2', '32', '33']
    associated_indices_list = []
    for label in label_list:
        associated_indices_list.append(find_associated(label))

    # 读取输入参数
    try:
        input_params = np.loadtxt(INPUT_CSV, delimiter=",")[:1]
    except FileNotFoundError:
        print(f"输入文件 {INPUT_CSV} 未找到。")
        return

    for i, params in enumerate(input_params):
        file.write(f"Exec Param #{i}: {params} \n")
        auto_bumper_move(lattice_data, associated_indices_list[0], params[0])        
        auto_top_move(lattice_data, associated_indices_list[1], params[1])
        auto_spoiler_move(lattice_data, associated_indices_list[2], params[2])
        # 导出 STL 文件
        write_FFD(i, WORKING_DIR)
        auto_bumper_move(lattice_data, associated_indices_list[0], -params[0])        
        auto_top_move(lattice_data, associated_indices_list[1], -params[1])
        auto_spoiler_move(lattice_data, associated_indices_list[2], -params[2])

    # 关闭日志文件
    file.close()

if __name__ == "__main__":
    main()