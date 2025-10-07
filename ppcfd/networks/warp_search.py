# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

# ruff: noqa: S101,F722,F821
from typing import Tuple
from typing import Union

import paddle
import warp as wp
from jaxtyping import Float


@dataclass
class NeighborSearchReturnType:
    neighbors_index: paddle.Tensor
    neighbors_distance: paddle.Tensor
    neighbors_row_splits: paddle.Tensor


@wp.kernel
def _radius_search_count(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),  # type: ignore
    queries: wp.array(dtype=wp.vec3),  # type: ignore
    result_count: wp.array(dtype=wp.int32),  # type: ignore
    radius: wp.float32,
):
    tid = wp.tid()

    # create grid query around point
    qp = queries[tid]
    query = wp.hash_grid_query(hashgrid, qp, radius)
    index = int(0)
    result_count_tid = int(0)

    while wp.hash_grid_query_next(query, index):
        neighbor = points[index]

        # compute distance to neighbor point
        dist = wp.length(qp - neighbor)
        if dist <= radius:
            result_count_tid += 1

    result_count[tid] = result_count_tid


@wp.kernel
def _radius_search_query(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    result_offset: wp.array(dtype=wp.int32),
    result_point_idx: wp.array(dtype=wp.int32),
    result_point_dist: wp.array(dtype=wp.float32),
    radius: wp.float32,
):
    tid = wp.tid()

    # create grid query around point
    qp = queries[tid]
    query = wp.hash_grid_query(hashgrid, qp, radius)
    index = int(0)
    result_count = int(0)
    offset_tid = result_offset[tid]

    while wp.hash_grid_query_next(query, index):
        neighbor = points[index]

        # compute distance to neighbor point
        dist = wp.length(qp - neighbor)
        if dist <= radius:
            result_point_idx[offset_tid + result_count] = index
            result_point_dist[offset_tid + result_count] = dist
            result_count += 1


def _radius_search_warp(
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    radius: float,
    grid_dim: Union[int, Tuple[int, int, int]] = (128, 128, 128),
    device: str = "cuda",
):
    # convert grid_dim to Tuple if it is int
    if isinstance(grid_dim, int):
        grid_dim = (grid_dim, grid_dim, grid_dim)
    result_count = wp.zeros(shape=len(queries), dtype=wp.int32)
    grid = wp.HashGrid(
        dim_x=grid_dim[0],
        dim_y=grid_dim[1],
        dim_z=grid_dim[2],
        device=device,
    )
    grid.build(points=points, radius=2 * radius)

    # For 10M radius search, the result can overflow and fail
    wp.launch(
        kernel=_radius_search_count,
        dim=len(queries),
        inputs=[grid.id, points, queries, result_count, radius],
        device=device,
    )

    torch_offset = paddle.zeros(len(result_count) + 1, dtype=paddle.int32)
    result_count_torch = wp.to_paddle(result_count)
    torch_offset[1:] = paddle.cumsum(result_count_torch, axis=0)
    total_count = torch_offset[-1].item()
    assert (
        total_count < 2**31 - 1
    ), f"Total result count is too large: {total_count} > 2**31 - 1"

    result_point_idx = wp.zeros(shape=(total_count,), dtype=wp.int32)
    result_point_dist = wp.zeros(shape=(total_count,), dtype=wp.float32)

    wp.launch(
        kernel=_radius_search_query,
        dim=len(queries),
        inputs=[
            grid.id,
            points,
            queries,
            wp.from_paddle(torch_offset),
            result_point_idx,
            result_point_dist,
            radius,
        ],
        device=device,
    )

    return (result_point_idx, result_point_dist, torch_offset)


def radius_search_warp(
    points, queries, radius, grid_dim=(128, 128, 128), device="cuda"
):
    """
    Args:
        points: [N, 3]
        queries: [M, 3]
        radius: float
        grid_dim: Union[int, Tuple[int, int, int]]
        device: str

    Returns:
        neighbor_index: [Q]
        neighbor_distance: [Q]
        neighbor_split: [M + 1]
    """
    # Convert from warp to torch
    points = points.clone()
    queries = queries.clone()
    assert points.is_contiguous(), "points must be contiguous"
    assert queries.is_contiguous(), "queries must be contiguous"
    points_wp = wp.from_paddle(points, dtype=wp.vec3)
    queries_wp = wp.from_paddle(queries, dtype=wp.vec3)

    result_point_idx, result_point_dist, torch_offset = _radius_search_warp(
        points=points_wp,
        queries=queries_wp,
        radius=radius,
        grid_dim=grid_dim,
        device=device,
    )

    # Convert from warp to torch
    result_point_idx = wp.to_paddle(result_point_idx)
    result_point_dist = wp.to_paddle(result_point_dist)

    # Neighbor index, Neighbor Distance, Neighbor Split
    return_val = NeighborSearchReturnType(
        result_point_idx, result_point_dist, torch_offset
    )

    return return_val


def batched_radius_search_warp(
    points, queries, radius, grid_dim=(128, 128, 128), device="cuda"
):
    """
    Args:
        points: [B, N, 3]
        queries: [B, M, 3]
        radius: float
        grid_dim: Union[int, Tuple[int, int, int]]
        device: str

    Returns:
        neighbor_index: [Q]
        neighbor_distance: [Q]
        neighbor_split: [B*M + 1]
    """
    B, N, _ = points.shape
    neighbor_index_list = []
    neighbor_distance_list = []
    neighbor_split_list = []
    index_offset = 0
    split_offset = 0
    for b in range(B):
        neighbor_index, neighbor_distance, neighbor_split = radius_search_warp(
            points=points[b],
            queries=queries[b],
            radius=radius,
            grid_dim=grid_dim,
            device=device,
        )
        neighbor_index_list.append(neighbor_index + index_offset)
        neighbor_distance_list.append(neighbor_distance)
        exit()
        # if b is last, append all neighbor_split since the last element is the total count
        if b == B - 1:
            neighbor_split_list.append(neighbor_split + split_offset)
        else:
            neighbor_split_list.append(neighbor_split[:-1] + split_offset)

        index_offset += N
        split_offset += len(neighbor_index)

    # Neighbor index, Neighbor Distance, Neighbor Split
    return (
        paddle.concat(neighbor_index_list),
        paddle.concat(neighbor_distance_list),
        paddle.concat(neighbor_split_list),
    )


_WARP_NEIGHBOR_SEARCH_INIT = False
if not _WARP_NEIGHBOR_SEARCH_INIT:
    wp.init()
    _WARP_NEIGHBOR_SEARCH_INIT = True


if __name__ == "__main__":
    paddle.seed(42)
    # Test search
    B = 5
    N = 100_000
    M = 200_000
    points = paddle.rand([B, N, 3])
    queries = paddle.rand([B, M, 3])

    radii = [0.05, 0.01, 0.005]
    for radius in radii:
        print(f"Testing radius: {radius}")
        result_point_idx, result_point_dist, torch_offset = batched_radius_search_warp(
            points=points, queries=queries, radius=radius
        )
        print(result_point_idx.shape)
        print(result_point_dist.shape)
        print(torch_offset.shape)
