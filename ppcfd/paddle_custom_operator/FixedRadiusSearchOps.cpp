// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include <vector>

#include "../../Open3D/cpp/open3d/core/Dtype.h"
#include "../../Open3D/cpp/open3d/core/nns/NeighborSearchCommon.h"
#include "../../Open3D/cpp/open3d/utility/Helper.h"
#include "PaddleHelper.h"
// #include "paddle/script.h"

using namespace open3d::core::nns;

template <class T, class TIndex>
void FixedRadiusSearchCPU(const paddle::Tensor& points,
                          const paddle::Tensor& queries,
                          double radius,
                          const paddle::Tensor& points_row_splits,
                          const paddle::Tensor& queries_row_splits,
                          const paddle::Tensor& hash_table_splits,
                          const paddle::Tensor& hash_table_index,
                          const paddle::Tensor& hash_table_cell_splits,
                          const Metric metric,
                          const bool ignore_query_point,
                          const bool return_distances,
                          paddle::Tensor& neighbors_index,
                          paddle::Tensor& neighbors_row_splits,
                          paddle::Tensor& neighbors_distance);
#ifdef BUILD_CUDA_MODULE
template <class T, class TIndex>
void FixedRadiusSearchCUDA(const paddle::Tensor& points,
                           const paddle::Tensor& queries,
                           double radius,
                           const paddle::Tensor& points_row_splits,
                           const paddle::Tensor& queries_row_splits,
                           const paddle::Tensor& hash_table_splits,
                           const paddle::Tensor& hash_table_index,
                           const paddle::Tensor& hash_table_cell_splits,
                           const Metric metric,
                           const bool ignore_query_point,
                           const bool return_distances,
                           paddle::Tensor& neighbors_index,
                           paddle::Tensor& neighbors_row_splits,
                           paddle::Tensor& neighbors_distance);
#endif

std::vector<paddle::Tensor> FixedRadiusSearch(
    const paddle::Tensor& points,
    const paddle::Tensor& queries,
    paddle::Tensor& points_row_splits,
    const paddle::Tensor& queries_row_splits_gpu,
    const paddle::Tensor& hash_table_splits_gpu,
    const paddle::Tensor& hash_table_index,
    const paddle::Tensor& hash_table_cell_splits,
    int index_dtype_id,
    float radius,
    const std::string& metric_str,
    const bool ignore_query_point,
    const bool return_distances
    ) {
        if (index_dtype_id == 7) {
            auto index_dtype = paddle::DataType::INT32;
        } else if (index_dtype_id == 9) {
            auto index_dtype = paddle::DataType::INT64;
        } else {
            std::cout << "FixedRadiusSearchOps.cpp : Index type not supported : " << index_dtype_id << std::endl;
            exit(-1);
        }
    auto index_dtype = ToPaddleDtype<int32_t>();
    Metric metric = L2;
    if (metric_str == "L1") {
        metric = L1;
    } else if (metric_str == "L2") {
        metric = L2;
    } else if (metric_str == "Linf") {
        metric = Linf;
    } else {
        PD_CHECK(false, "metric must be one of (L1, L2, Linf) but got " + metric_str);
    }
    CHECK_TYPE(points_row_splits, paddle::DataType::INT64);
    CHECK_TYPE(queries_row_splits_gpu, paddle::DataType::INT64);
    CHECK_TYPE(hash_table_splits_gpu, paddle::DataType::INT32);
    CHECK_TYPE(hash_table_index, paddle::DataType::INT64);
    CHECK_TYPE(hash_table_cell_splits, paddle::DataType::INT64);
    CHECK_SAME_DTYPE(points, queries);
    CHECK_SAME_DEVICE_TYPE(points, queries);

    PD_CHECK(index_dtype == paddle::DataType::INT32 || index_dtype == paddle::DataType::INT64,
                "index_dtype must be int32 or int64");
    // ensure that these are on the cpu
    points_row_splits = points_row_splits.copy_to(paddle::CPUPlace(), false);
    paddle::Tensor queries_row_splits = queries_row_splits_gpu.copy_to(paddle::CPUPlace(), false);
    paddle::Tensor hash_table_splits = hash_table_splits_gpu.copy_to(paddle::CPUPlace(), false);

    // check input shapes
    using namespace open3d::ml::op_util;
    Dim num_points("num_points");
    Dim num_queries("num_queries");
    Dim batch_size("batch_size");
    Dim num_cells("num_cells");
    CHECK_SHAPE(points, num_points, 3);
    CHECK_SHAPE(hash_table_index, num_points);
    CHECK_SHAPE(queries, num_queries, 3);
    CHECK_SHAPE(points_row_splits, batch_size + 1);
    CHECK_SHAPE(queries_row_splits, batch_size + 1);
    CHECK_SHAPE(hash_table_splits, batch_size + 1);
    CHECK_SHAPE(hash_table_cell_splits, num_cells + 1);
    const auto& point_type = points.dtype();
    auto device = points.place();
    paddle::Tensor neighbors_index;
    paddle::Tensor neighbors_row_splits = paddle::empty(
            {queries.shape()[0] + 1},
            ToPaddleDtype<int64_t>(), device);
    paddle::Tensor neighbors_distance;

#define FN_PARAMETERS                                                      \
    points, queries, radius, points_row_splits, queries_row_splits,        \
            hash_table_splits, hash_table_index, hash_table_cell_splits,   \
            metric, ignore_query_point, return_distances, neighbors_index, \
            neighbors_row_splits, neighbors_distance
    if (points.is_gpu()) {
#ifdef BUILD_CUDA_MODULE
        // pass to cuda function
        if (ComparePaddleDtype<float>(point_type)) {
            if (index_dtype == paddle::DataType::INT32) {

                FixedRadiusSearchCUDA<float, int32_t>(FN_PARAMETERS);
            } else {

                FixedRadiusSearchCUDA<float, int64_t>(FN_PARAMETERS);
            }

            auto out = std::vector<paddle::Tensor>();
            if (neighbors_index.shape()[0] == 0)
            {
                neighbors_index = paddle::full({1,}, 0, paddle::DataType::INT32);
                neighbors_distance = paddle::full({1,}, 0, paddle::DataType::INT32);
            }

            out.push_back(neighbors_index);
            out.push_back(neighbors_row_splits);
            out.push_back(neighbors_distance);

            return {out};
        }
#else
        PD_CHECK(false,
                    "FixedRadiusSearch was not compiled with CUDA support");
#endif
    } else {
        if (ComparePaddleDtype<float>(point_type)) {
            if (index_dtype == paddle::DataType::INT32) {
                FixedRadiusSearchCPU<float, int32_t>(FN_PARAMETERS);
            } else {
                FixedRadiusSearchCPU<float, int64_t>(FN_PARAMETERS);
            }
        } else {
            if (index_dtype == paddle::DataType::INT32) {
                FixedRadiusSearchCPU<double, int32_t>(FN_PARAMETERS);
            } else {
                FixedRadiusSearchCPU<double, int64_t>(FN_PARAMETERS);
            }
        }
        auto out = std::vector<paddle::Tensor>();
        out.push_back(neighbors_index);
        out.push_back(neighbors_row_splits);
        out.push_back(neighbors_distance);
        std::cout << "FIxedRadiusSearchCPU" << std::endl;
        return {out};
    }
    PD_CHECK(false, "FixedRadiusSearch does not support this Tensor as input for points");
    paddle::Tensor test = paddle::empty({1});
    auto out = std::vector<paddle::Tensor>();
    out.push_back(test);
    out.push_back(test);
    out.push_back(test);
    return {out};
}

// 定义一个名为open_3d_fixed_radius_search的函数，用于在给定半径内搜索三维空间中的点
// 输入参数：
//   points：存储所有点的数组
//   queries：存储查询点的数组
//   points_row_splits：存储points数组的行分割信息的数组
//   queries_row_splits_gpu：存储queries数组的行分割信息的数组
//   hash_table_splits_gpu：存储哈希表分割信息的数组
//   hash_table_index：哈希表索引
//   hash_table_cell_splits：存储哈希表每个单元的分割信息的数组
// 输出参数：
//   Out[0]：存储搜索结果的数组
//   Out[1]：存储距离的数组（可选）
//   Out[2]：存储索引的数组（可选）
// 属性参数：
//   index_dtype：数据类型，默认为int类型
//   radius：搜索半径，默认为float类型
//   metric_str：距离度量标准，默认为std::string类型
//   ignore_query_point：是否忽略查询点，默认为bool类型
//   return_distances：是否返回距离，默认为bool类型
// 设置Kernel函数为FixedRadiusSearch
PD_BUILD_OP(open_3d_fixed_radius_search)
    .Inputs({"points", "queries", "points_row_splits", "queries_row_splits_gpu", "hash_table_splits_gpu", "hash_table_index", "hash_table_cell_splits"})
    .Outputs({"Out[0]", "Out[1]", "Out[2]"})
    .Attrs({
        "index_dtype: int",
        "radius: float",
        "metric_str: std::string",
        "ignore_query_point: bool",
        "return_distances: bool"
    })
    .SetKernelFn(PD_KERNEL(FixedRadiusSearch));
