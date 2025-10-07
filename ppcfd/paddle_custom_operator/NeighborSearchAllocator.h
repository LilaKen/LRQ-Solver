// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//

#include "PaddleHelper.h"

// These classes implement functors that can be passed to the neighbor search
// functions.

template <class T, class TIndex>
class NeighborSearchAllocator {
public:
    NeighborSearchAllocator(paddle::Place device, int device_idx)
        : device(device), device_idx(device_idx) {}

    void AllocIndices(TIndex** ptr, size_t num) {
        neighbors_index = paddle::empty(
                {int64_t(num)}, paddle::DataType(ToPaddleDtype<TIndex>()), device);
        *ptr = neighbors_index.data<TIndex>();
    }

    void AllocDistances(T** ptr, size_t num) {
        neighbors_distance = paddle::empty(
                {int64_t(num)}, paddle::DataType(ToPaddleDtype<T>()), device);
        *ptr = neighbors_distance.data<T>();
    }

    const TIndex* IndicesPtr() const {
        return neighbors_index.data<TIndex>();
    }

    const T* DistancesPtr() const { return neighbors_distance.data<T>(); }

    const paddle::Tensor& NeighborsIndex() const { return neighbors_index; }
    const paddle::Tensor& NeighborsDistance() const {
        return neighbors_distance;
    }

private:
    paddle::Tensor neighbors_index;
    paddle::Tensor neighbors_distance;
    paddle::Place device;
    int device_idx;
};
