// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

// #include "ATen/cuda/CUDAContext.h"
#include "../../Open3D/cpp/open3d/core/nns/FixedRadiusSearchImpl.cuh"
#include "../../Open3D/cpp/open3d/core/nns/NeighborSearchCommon.h"
#include "PaddleHelper.h"
#include "NeighborSearchAllocator.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
// #include "paddle/script.h"

using namespace open3d::core::nns;

__global__ void int64_to_uint32_kernel(const int64_t* paddle_array, uint32_t* array, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        // 将uint32_t转换为int64_t
        array[index] = static_cast<int64_t>(paddle_array[index]);
    }
}


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
                           paddle::Tensor& neighbors_distance) {
    auto stream = points.stream();
    
    auto device = points.place();
    auto device_idx = points.place().GetDeviceId();

    const auto& cuda_device_props = phi::backends::gpu::GetDeviceProperties(device_idx);
    const int texture_alignment = cuda_device_props.textureAlignment;


    NeighborSearchAllocator<T, TIndex> output_allocator(device, device_idx);
    void* temp_ptr = nullptr;
    size_t temp_size = 0;

    uint32_t *index;
    uint32_t *cell;

    const int size_1 = hash_table_cell_splits.shape()[0];
    const int size_2 = hash_table_index.shape()[0];

    cudaError_t err1 = cudaMalloc((void **)&index, hash_table_index.numel()       * sizeof(uint32_t));
    cudaError_t err2 = cudaMalloc((void **)&cell,  hash_table_cell_splits.numel() * sizeof(uint32_t));

        // 定义线程块和网格尺寸
    int blockSize = 256;
    int gridSize_1 = (size_1 + blockSize - 1) / blockSize;

    // 执行类型转换内核
    int64_to_uint32_kernel<<<gridSize_1, blockSize>>>(hash_table_cell_splits.data<int64_t>(), cell, size_1);

    // 等待内核完成
    cudaDeviceSynchronize();

    // 定义线程块和网格尺寸
    int gridSize_2 = (size_2 + blockSize - 1) / blockSize;

    // 执行类型转换内核
    int64_to_uint32_kernel<<<gridSize_2, blockSize>>>(hash_table_index.data<int64_t>(), index, size_2);

    // 等待内核完成
    cudaDeviceSynchronize();

    // determine temp_size
    impl::FixedRadiusSearchCUDA<T, TIndex>(
            stream, temp_ptr, temp_size, texture_alignment,
            neighbors_row_splits.data<int64_t>(), points.shape()[0],
            points.data<T>(), queries.shape()[0], queries.data<T>(),
            T(radius), points_row_splits.shape()[0],
            points_row_splits.data<int64_t>(), queries_row_splits.shape()[0],
            queries_row_splits.data<int64_t>(),
            (uint32_t*)hash_table_splits.data<int32_t>(),
            hash_table_cell_splits.shape()[0],
            cell,
            index,
        //     (uint32_t*)hash_table_cell_splits.data<int32_t>(),
        //     (uint32_t*)hash_table_index.data<int32_t>(),

            metric,
            ignore_query_point, return_distances, output_allocator);

    auto temp_tensor = CreateTempTensor(temp_size, points.place(), &temp_ptr);

    // actually run the search
    impl::FixedRadiusSearchCUDA<T, TIndex>(
            stream, temp_ptr, temp_size, texture_alignment,
            neighbors_row_splits.data<int64_t>(), points.shape()[0],
            points.data<T>(), queries.shape()[0], queries.data<T>(),
            T(radius), points_row_splits.shape()[0],
            points_row_splits.data<int64_t>(), queries_row_splits.shape()[0],
            queries_row_splits.data<int64_t>(),
            (uint32_t*)hash_table_splits.data<int32_t>(),
            hash_table_cell_splits.shape()[0],
            cell,
            index,
            // (uint32_t*)hash_table_cell_splits.data<int32_t>(),
            // (uint32_t*)hash_table_index.data<int32_t>(),
            metric,
            ignore_query_point, return_distances, output_allocator);


    cudaFree(cell);
    cudaFree(index);

    neighbors_index = output_allocator.NeighborsIndex();
    neighbors_distance = output_allocator.NeighborsDistance();
}

#define INSTANTIATE(T, TIndex)                                                \
    template void FixedRadiusSearchCUDA<T, TIndex>(                           \
            const paddle::Tensor& points, const paddle::Tensor& queries,        \
            double radius, const paddle::Tensor& points_row_splits,            \
            const paddle::Tensor& queries_row_splits,                          \
            const paddle::Tensor& hash_table_splits,                           \
            const paddle::Tensor& hash_table_index,                            \
            const paddle::Tensor& hash_table_cell_splits, const Metric metric, \
            const bool ignore_query_point, const bool return_distances,       \
            paddle::Tensor& neighbors_index,                                   \
            paddle::Tensor& neighbors_row_splits,                              \
            paddle::Tensor& neighbors_distance);

INSTANTIATE(float, int32_t)
INSTANTIATE(float, int64_t)