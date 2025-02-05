#ifndef TENSORPRIVATE_H
#define TENSORPRIVATE_H
#pragma once

#include "txeo/Tensor.h"
#include <tensorflow/core/framework/tensor.h>

template <typename T>
struct txeo::Tensor<T>::Impl {
    std::unique_ptr<tensorflow::Tensor> tf_tensor{nullptr};
};

#endif