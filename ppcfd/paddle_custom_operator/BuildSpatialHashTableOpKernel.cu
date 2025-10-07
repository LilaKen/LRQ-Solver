// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "../../Open3D/cpp/open3d/core/nns/FixedRadiusSearchImpl.cuh"
#include "PaddleHelper.h"
#include "paddle/phi/backends/all_context.h"
using namespace open3d::core::nns;

__global__ void uint32ToInt64Kernel(const uint32_t* cell, int64_t* paddle_cell, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        // 将uint32_t转换为int64_t
        paddle_cell[index] = static_cast<int64_t>(cell[index]);
    }
}

template <class T>
void BuildSpatialHashTableCUDA(const paddle::Tensor& points,
                               double radius,
                               const paddle::Tensor& points_row_splits,
                               const std::vector<uint32_t>& hash_table_splits,
                               paddle::Tensor& hash_table_index,
                               paddle::Tensor& hash_table_cell_splits) {
    auto stream = points.stream();
    auto cuda_device_props = phi::backends::gpu::GetDeviceProperties(-1);
    const int texture_alignment = cuda_device_props.textureAlignment;

    void* temp_ptr = nullptr;
    size_t temp_size = 0;

    uint32_t *index;
    uint32_t *cell;

    const int size_1 = hash_table_cell_splits.shape()[0];
    const int size_2 = hash_table_index.shape()[0];

    cudaMalloc((void **)&cell,  size_1 * sizeof(uint32_t));
    cudaMalloc((void **)&index, size_2 * sizeof(uint32_t));

    // determine temp_size
    impl::BuildSpatialHashTableCUDA(
            stream, temp_ptr, temp_size, texture_alignment, points.shape()[0],
            points.data<T>(), T(radius), points_row_splits.shape()[0],
            points_row_splits.data<int64_t>(), hash_table_splits.data(),
            hash_table_cell_splits.shape()[0],
            cell,
            index);
    auto device = points.place();
    auto temp_tensor = CreateTempTensor(temp_size, device, &temp_ptr);

    // actually build the table
    impl::BuildSpatialHashTableCUDA(
            stream, temp_ptr, temp_size, texture_alignment, points.shape()[0],
            points.data<T>(), T(radius), points_row_splits.shape()[0],
            points_row_splits.data<int64_t>(), hash_table_splits.data(),
            hash_table_cell_splits.shape()[0],
            cell,
            index);

    // 定义线程块和网格尺寸
    int blockSize = 256;
    int gridSize_1 = (size_1 + blockSize - 1) / blockSize;

    // 执行类型转换内核
    uint32ToInt64Kernel<<<gridSize_1, blockSize>>>(cell, hash_table_cell_splits.data<int64_t>(), size_1);

    // 等待内核完成
    cudaDeviceSynchronize();

    // 定义线程块和网格尺寸
    int gridSize_2 = (size_2 + blockSize - 1) / blockSize;

    // 执行类型转换内核
    uint32ToInt64Kernel<<<gridSize_2, blockSize>>>(index, hash_table_index.data<int64_t>(), size_2);

    // 等待内核完成
    cudaDeviceSynchronize();

    cudaFree(cell);
    cudaFree(index);
}

#define INSTANTIATE(T)                                          \
    template void BuildSpatialHashTableCUDA<T>(                 \
            const paddle::Tensor&, double, const paddle::Tensor&, \
            const std::vector<uint32_t>&, paddle::Tensor&, paddle::Tensor&);

INSTANTIATE(float)
