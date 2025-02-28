#ifndef TENSOR_PRIVATE_H
#define TENSOR_PRIVATE_H
#pragma once

#include <tensorflow/core/framework/tensor.h>

#include "txeo/Tensor.h"
#include "txeo/TensorShape.h"
#include "txeo/detail/utils.h"

template <typename T>
struct txeo::Tensor<T>::Impl {
    std::unique_ptr<tensorflow::Tensor> tf_tensor{nullptr};
    txeo::TensorShape txeo_shape{};
};

// namespace txeo::detail {

// template <typename T, typename U>
// txeo::Tensor<T> to_txeo_tensor(U &&tf_tensor) {
//   txeo::Tensor<T> resp;
//   resp._impl->tf_tensor = std::make_unique<tensorflow::Tensor>(std::forward<U>(tf_tensor));
//   resp._impl->txeo_shape._impl->ext_tf_shape = &resp._impl->tf_tensor->shape();
//   resp._impl->txeo_shape._impl->stride =
//       txeo::detail::calc_stride(*resp._impl->txeo_shape._impl->ext_tf_shape);

//   return resp;
// }
// } // namespace txeo::detail

#endif