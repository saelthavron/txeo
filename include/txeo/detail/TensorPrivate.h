#ifndef TENSORPRIVATE_H
#define TENSORPRIVATE_H
#pragma once

#include <tensorflow/core/framework/tensor.h>

#include "txeo/Tensor.h"
#include "txeo/TensorShape.h"

template <typename T>
struct txeo::Tensor<T>::Impl {
    std::unique_ptr<tensorflow::Tensor> tf_tensor{nullptr};
    std::unique_ptr<txeo::TensorShape> txeo_shape{nullptr};
};

#endif