#ifndef TENSOR_PRIVATE_H
#define TENSOR_PRIVATE_H
#pragma once

#include <tensorflow/core/framework/tensor.h>

#include "txeo/Tensor.h"
#include "txeo/TensorShape.h"

template <typename T>
struct txeo::Tensor<T>::Impl {
    std::unique_ptr<tensorflow::Tensor> tf_tensor{nullptr};
    txeo::TensorShape txeo_shape{};
    // const tensorflow::Tensor *ext_tf_tensor{nullptr};
    // bool owns{false};
};

#endif