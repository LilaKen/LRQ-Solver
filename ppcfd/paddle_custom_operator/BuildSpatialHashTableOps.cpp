// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <vector>
#include "PaddleHelper.h"

template <class T>
void BuildSpatialHashTableCPU(const paddle::Tensor& points,
                              double radius,
                              const paddle::Tensor& points_row_splits,
                              const std::vector<uint32_t>& hash_table_splits,
                              paddle::Tensor& hash_table_index,
                              paddle::Tensor& hash_table_cell_splits);
#ifdef BUILD_CUDA_MODULE
template <class T>
void BuildSpatialHashTableCUDA(const paddle::Tensor& points,
                               double radius,
                               const paddle::Tensor& points_row_splits,
                               const std::vector<uint32_t>& hash_table_splits,
                               paddle::Tensor& hash_table_index,
                               paddle::Tensor& hash_table_cell_splits);
#endif

std::vector<paddle::Tensor> BuildSpatialHashTableForward(
    paddle::Tensor& points,
    paddle::Tensor& points_row_splits,
    float radius_input,
    float hash_table_size_factor_input,
    int max_hash_table_size_input
    ){
    auto radius = (double) radius_input;
    auto hash_table_size_factor = (double) hash_table_size_factor_input;
    auto max_hash_table_size = (int64_t) max_hash_table_size_input;
    points_row_splits = points_row_splits.copy_to(paddle::CPUPlace(), false);

    CHECK_TYPE(points_row_splits, paddle::DataType::INT64);
    // check input shapes
    using namespace open3d::ml::op_util;
    Dim num_points("num_points");
    Dim batch_size("batch_size");
    CHECK_SHAPE(points, num_points, 3);
    CHECK_SHAPE(points_row_splits, batch_size + 1);
    const auto& point_type = points.dtype();
    std::vector<uint32_t> hash_table_splits(batch_size.value() + 1, 0);
    for (int i = 0; i < batch_size.value(); ++i) {
        int64_t num_points_i = points_row_splits.data<int64_t>()[i + 1] -
                               points_row_splits.data<int64_t>()[i];
        int64_t hash_table_size = std::min<int64_t>(
                std::max<int64_t>(hash_table_size_factor * num_points_i, 1),
                max_hash_table_size);
        hash_table_splits[i + 1] = hash_table_splits[i] + hash_table_size;
    }

    auto device = points.place();
    auto device_idx = points.place().GetDeviceId();
    paddle::Tensor hash_table_index = paddle::empty(
            {points.shape()[0]},
            paddle::DataType(ToPaddleDtype<int64_t>()), device);
    paddle::Tensor hash_table_cell_splits = paddle::empty(
            {hash_table_splits.back() + 1},
            paddle::DataType(ToPaddleDtype<int64_t>()), device);
    paddle::Tensor out_hash_table_splits = paddle::empty(
            {batch_size.value() + 1}, paddle::DataType(ToPaddleDtype<int32_t>()));
    for (size_t i = 0; i < hash_table_splits.size(); ++i) {
        out_hash_table_splits.data<int32_t>()[i] = hash_table_splits[i];
    }

#define FN_PARAMETERS                                                       \
    points, radius, points_row_splits, hash_table_splits, hash_table_index, \
            hash_table_cell_splits
#define CALL(type, fn)                                                   \
    if (ComparePaddleDtype<type>(point_type)) {                          \
        fn<type>(FN_PARAMETERS);                                         \
        std::vector<paddle::Tensor> out;                                 \
        out.push_back(hash_table_index);                                 \
        out.push_back(hash_table_cell_splits);                           \
        out.push_back(out_hash_table_splits);                            \
        return {out};                                                    \
    }
    if (points.is_gpu()) {
#ifdef BUILD_CUDA_MODULE
        // pass to cuda function
        CALL(float, BuildSpatialHashTableCUDA)
#else
        PD_CHECK(false,
                    "BuildSpatialHashTable was not compiled with CUDA support");
#endif
    } else {
        CALL(float, BuildSpatialHashTableCPU)
        CALL(double, BuildSpatialHashTableCPU)
    }
    PD_CHECK(false, "BuildSpatialHashTable does not support this tensor as input for points");
    auto out = std::vector<paddle::Tensor>();
    return {out};
}


PD_BUILD_OP(open_3d_build_spatial_hash_table)
    .Inputs({"points", "points_row_splits"})
    .Outputs({"Out[0]", "Out[1]", "Out[2]"})
    .Attrs({
        "radius: float",
        "hash_table_size_factor: float",
        "max_hash_table_size: int"
    })
    .SetKernelFn(PD_KERNEL(BuildSpatialHashTableForward));
