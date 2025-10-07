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
#include "PaddleHelper.h"
#include "../../Open3D/cpp/open3d/utility/Helper.h"
// #include "paddle/script.h"

using namespace open3d::core::nns;

template <class T, class TIndex>
void KnnSearchCPU(const paddle::Tensor& points,
                  const paddle::Tensor& queries,
                  const int64_t k,
                  const paddle::Tensor& points_row_splits,
                  const paddle::Tensor& queries_row_splits,
                  const Metric metric,
                  const bool ignore_query_point,
                  const bool return_distances,
                  paddle::Tensor& neighbors_index,
                  paddle::Tensor& neighbors_row_splits,
                  paddle::Tensor& neighbors_distance);

std::vector<paddle::Tensor> KnnSearch(
        paddle::Tensor points,
        paddle::Tensor queries,
        const int64_t k,
        paddle::Tensor points_row_splits,
        paddle::Tensor queries_row_splits,
        paddle::ScalarType index_dtype,
        const std::string& metric_str,
        const bool ignore_query_point,
        const bool return_distances) {
    Metric metric = L2;
    if (metric_str == "L1") {
        metric = L1;
    } else if (metric_str == "L2") {
        metric = L2;
    } else {
        PD_CHECK(false,
                    "metric must be one of (L1, L2) but got " + metric_str);
    }
    PD_CHECK(k > 0, "k must be greater than zero");
    CHECK_TYPE(points_row_splits, kInt64);
    CHECK_TYPE(queries_row_splits, kInt64);
    CHECK_SAME_DTYPE(points, queries);
    CHECK_SAME_DEVICE_TYPE(points, queries);
    PD_CHECK(index_dtype == paddle::kInt32 || index_dtype == paddle::kInt64,
                "index_dtype must be int32 or int64");
    // ensure that these are on the cpu
    points_row_splits = points_row_splits.to(paddle::kCPU);
    queries_row_splits = queries_row_splits.to(paddle::kCPU);
    points = points.contiguous();
    queries = queries.contiguous();
    points_row_splits = points_row_splits.contiguous();
    queries_row_splits = queries_row_splits.contiguous();

    // check input shapes
    using namespace open3d::ml::op_util;
    Dim num_points("num_points");
    Dim num_queries("num_queries");
    Dim batch_size("batch_size");
    Dim num_cells("num_cells");
    CHECK_SHAPE(points, num_points, 3);
    CHECK_SHAPE(queries, num_queries, 3);
    CHECK_SHAPE(points_row_splits, batch_size + 1);
    CHECK_SHAPE(queries_row_splits, batch_size + 1);

    const auto& point_type = points.dtype();

    auto device = points.device().type();
    auto device_idx = points.device().index();

    paddle::Tensor neighbors_index;
    paddle::Tensor neighbors_row_splits = paddle::empty(
            {queries.shape()[0] + 1},
            ToPaddleDtype<int64_t>(), device);
    paddle::Tensor neighbors_distance;

#define FN_PARAMETERS                                                  \
    points, queries, k, points_row_splits, queries_row_splits, metric, \
            ignore_query_point, return_distances, neighbors_index,     \
            neighbors_row_splits, neighbors_distance

    if (points.is_gpu()) {
        PD_CHECK(false, "KnnSearch does not support CUDA")
    } else {
        if (ComparePaddleDtype<float>(point_type)) {
            if (index_dtype == paddle::kInt32) {
                KnnSearchCPU<float, int32_t>(FN_PARAMETERS);
            } else {
                KnnSearchCPU<float, int64_t>(FN_PARAMETERS);
            }
        } else {
            if (index_dtype == paddle::kInt32) {
                KnnSearchCPU<double, int32_t>(FN_PARAMETERS);
            } else {
                KnnSearchCPU<double, int64_t>(FN_PARAMETERS);
            }
        }
        return std::make_tuple(neighbors_index, neighbors_row_splits,
                               neighbors_distance);
    }
    PD_CHECK(false, "KnnSearch does not support " + points.toString() +
                               " as input for points")
    return std::vector<paddle::Tensor>();
}

const char* knn_fn_format =
        "open3d::knn_search(Tensor points, Tensor queries, int "
        "k, Tensor points_row_splits, Tensor queries_row_splits, ScalarType "
        "index_dtype=%d,"
        "str metric=\"L2\", bool ignore_query_point=False, bool "
        "return_distances=False) -> "
        "(Tensor neighbors_index, Tensor "
        "neighbors_row_splits, Tensor neighbors_distance)";

static auto registry = paddle::RegisterOperators(
        open3d::utility::FormatString(knn_fn_format, int(c10::ScalarType::Int)),
        &KnnSearch);
