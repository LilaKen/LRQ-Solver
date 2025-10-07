# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2023 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# This file is machine generated. Do not modify.
from collections import namedtuple as _namedtuple

three_nn = _namedtuple("three_nn", "dist idx")
voxel_pooling = _namedtuple("voxel_pooling", "pooled_positions pooled_features")
voxelize = _namedtuple(
    "voxelize",
    "voxel_coords voxel_point_indices voxel_point_row_splits voxel_batch_splits",
)
roi_pool = _namedtuple("roi_pool", "features flags")
radius_search = _namedtuple(
    "radius_search", "neighbors_index neighbors_row_splits neighbors_distance"
)
knn_search = _namedtuple(
    "knn_search", "neighbors_index neighbors_row_splits neighbors_distance"
)
invert_neighbors_list = _namedtuple(
    "invert_neighbors_list", "neighbors_index neighbors_row_splits neighbors_attributes"
)
fixed_radius_search = _namedtuple(
    "fixed_radius_search", "neighbors_index neighbors_row_splits neighbors_distance"
)
build_spatial_hash_table = _namedtuple(
    "build_spatial_hash_table",
    "hash_table_index hash_table_cell_splits hash_table_splits",
)
