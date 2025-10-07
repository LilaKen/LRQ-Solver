// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2023 www.open3d.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <sstream>
#include <type_traits>
#include "../../Open3D/cpp/open3d/ml/ShapeChecking.h"
#include "paddle/extension.h"
// Macros for checking tensor properties
#define CHECK_CUDA(x)                                         \
    do {                                                      \
        PD_CHECK(x.is_gpu(), #x " must be a CUDA tensor"); \
    } while (0)

#define CHECK_CONTIGUOUS(x)                                      \
    do {                                                         \
        PD_CHECK(x.is_contiguous(), #x " must be contiguous"); \
    } while (0)

#define CHECK_TYPE(x, type)                                                \
    do {                                                                   \
        PD_CHECK(x.dtype() == type, #x " must have type " #type); \
    } while (0)

#define CHECK_SAME_DEVICE_TYPE(...)                                          \
    do {                                                                     \
        if (!SameDeviceType({__VA_ARGS__})) {                                \
            PD_CHECK(                                                     \
                    false,                                                   \
                    #__VA_ARGS__                                             \
                            " must all have the same device type but got " + \
                            TensorInfoStr({__VA_ARGS__}));                    \
        }                                                                    \
    } while (0)

#define CHECK_SAME_DTYPE(...)                                              \
    do {                                                                   \
        if (!SameDtype({__VA_ARGS__})) {                                   \
            PD_CHECK(false,                                             \
                        #__VA_ARGS__                                       \
                                " must all have the same dtype but got " + \
                                TensorInfoStr({__VA_ARGS__}));              \
        }                                                                  \
    } while (0)
// Conversion from standard types to paddle types
typedef std::remove_const<decltype(paddle::DataType::INT32)>::type PaddleDtype_t;
template <class T>
inline PaddleDtype_t ToPaddleDtype() {
    PD_CHECK(false, "Unsupported type");
}
template <>
inline PaddleDtype_t ToPaddleDtype<uint8_t>() {
    return paddle::DataType::UINT8;
}
template <>
inline PaddleDtype_t ToPaddleDtype<int8_t>() {
    return paddle::DataType::INT8;
}
template <>
inline PaddleDtype_t ToPaddleDtype<int16_t>() {
    return paddle::DataType::INT16;
}
template <>
inline PaddleDtype_t ToPaddleDtype<int32_t>() {
    return paddle::DataType::INT32;
}
template <>
inline PaddleDtype_t ToPaddleDtype<int64_t>() {
    return paddle::DataType::INT64;
}
template <>
inline PaddleDtype_t ToPaddleDtype<float>() {
    return paddle::DataType::FLOAT32;
}
template <>
inline PaddleDtype_t ToPaddleDtype<double>() {
    return paddle::DataType::FLOAT64;
}

// convenience function for comparing standard types with paddle types
template <class T, class TDtype>
inline bool ComparePaddleDtype(const TDtype& t) {
    return ToPaddleDtype<T>() == t;
}

// convenience function to check if all tensors have the same device type
inline bool SameDeviceType(std::initializer_list<paddle::Tensor> tensors) {
    if (tensors.size()) {
        auto device_type = tensors.begin()->place();
        for (auto t : tensors) {
            if (device_type != t.place()) {
                return false;
            }
        }
    }
    return true;
}

// convenience function to check if all tensors have the same dtype
inline bool SameDtype(std::initializer_list<paddle::Tensor> tensors) {
    if (tensors.size()) {
        auto dtype = tensors.begin()->dtype();
        for (auto t : tensors) {
            if (dtype != t.dtype()) {
                return false;
            }
        }
    }
    return true;
}

inline std::string TensorInfoStr(std::initializer_list<paddle::Tensor> tensors) {
    std::stringstream sstr;
    size_t count = 0;
    for (const auto t : tensors) {
        sstr << t.size() << " " << "t.toString() missing in paddle" << " " << t.place(); //t.toString()
        ++count;
        if (count < tensors.size()) sstr << ", ";
    }
    return sstr.str();
}

// convenience function for creating a tensor for temp memory
inline paddle::Tensor CreateTempTensor(const int64_t size,
                                      const paddle::Place& device,
                                      void** ptr = nullptr) {
    paddle::Tensor tensor = paddle::empty(
            {size}, ToPaddleDtype<uint8_t>(), device);
    if (ptr) {
        *ptr = tensor.data<uint8_t>();
    }
    return tensor;
}

inline std::vector<open3d::ml::op_util::DimValue> GetShapeVector(
        paddle::Tensor tensor) {
    using namespace open3d::ml::op_util;
    const auto old_shape = tensor.shape();
    std::vector<DimValue> shape;
    for (auto i = 0; i < old_shape.size(); ++i) {
        shape.push_back(old_shape[i]);
    }
    return shape;
}

template <open3d::ml::op_util::CSOpt Opt = open3d::ml::op_util::CSOpt::NONE,
          class TDimX,
          class... TArgs>
std::tuple<bool, std::string> CheckShape(paddle::Tensor tensor,
                                         TDimX&& dimex,
                                         TArgs&&... args) {
    return open3d::ml::op_util::CheckShape<Opt>(GetShapeVector(tensor),
                                                std::forward<TDimX>(dimex),
                                                std::forward<TArgs>(args)...);
}

//
// Macros for checking the shape of Tensors.
// Usage:
//   {
//     using namespace open3d::ml::op_util;
//     Dim w("w");
//     Dim h("h");
//     CHECK_SHAPE(tensor1, 10, w, h); // checks if the first dim is 10
//                                     // and assigns w and h based on
//                                     // the shape of tensor1
//
//     CHECK_SHAPE(tensor2, 10, 20, h); // this checks if the the last dim
//                                      // of tensor2 matches the last dim
//                                      // of tensor1. The first two dims
//                                      // must match 10, 20.
//   }
//
//
// See "../ShapeChecking.h" for more info and limitations.
//
#define CHECK_SHAPE(tensor, ...)                                             \
    do {                                                                     \
        bool cs_success_;                                                    \
        std::string cs_errstr_;                                              \
        std::tie(cs_success_, cs_errstr_) = CheckShape(tensor, __VA_ARGS__); \
        PD_CHECK(cs_success_,                                             \
                    "invalid shape for '" #tensor "', " + cs_errstr_);        \
    } while (0)

#define CHECK_SHAPE_COMBINE_FIRST_DIMS(tensor, ...)                         \
    do {                                                                    \
        bool cs_success_;                                                   \
        std::string cs_errstr_;                                             \
        std::tie(cs_success_, cs_errstr_) =                                 \
                CheckShape<CSOpt::COMBINE_FIRST_DIMS>(tensor, __VA_ARGS__); \
        PD_CHECK(cs_success_,                                            \
                    "invalid shape for '" #tensor "', " + cs_errstr_);       \
    } while (0)

#define CHECK_SHAPE_IGNORE_FIRST_DIMS(tensor, ...)                         \
    do {                                                                   \
        bool cs_success_;                                                  \
        std::string cs_errstr_;                                            \
        std::tie(cs_success_, cs_errstr_) =                                \
                CheckShape<CSOpt::IGNORE_FIRST_DIMS>(tensor, __VA_ARGS__); \
        PD_CHECK(cs_success_,                                           \
                    "invalid shape for '" #tensor "', " + cs_errstr_);      \
    } while (0)

#define CHECK_SHAPE_COMBINE_LAST_DIMS(tensor, ...)                         \
    do {                                                                   \
        bool cs_success_;                                                  \
        std::string cs_errstr_;                                            \
        std::tie(cs_success_, cs_errstr_) =                                \
                CheckShape<CSOpt::COMBINE_LAST_DIMS>(tensor, __VA_ARGS__); \
        PD_CHECK(cs_success_,                                           \
                    "invalid shape for '" #tensor "', " + cs_errstr_);      \
    } while (0)

#define CHECK_SHAPE_IGNORE_LAST_DIMS(tensor, ...)                         \
    do {                                                                  \
        bool cs_success_;                                                 \
        std::string cs_errstr_;                                           \
        std::tie(cs_success_, cs_errstr_) =                               \
                CheckShape<CSOpt::IGNORE_LAST_DIMS>(tensor, __VA_ARGS__); \
        PD_CHECK(cs_success_,                                          \
                    "invalid shape for '" #tensor "', " + cs_errstr_);     \
    } while (0)
