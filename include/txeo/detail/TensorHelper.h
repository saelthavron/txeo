#ifndef TENSORHELPER_H
#define TENSORHELPER_H
#pragma once

#include "txeo/Matrix.h"
#include "txeo/detail/TensorPrivate.h"
#include "txeo/detail/TensorShapePrivate.h"

#include "txeo/detail/utils.h"

#include <initializer_list>
#include <stdexcept>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/framework/types.pb.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

namespace tf = tensorflow;

namespace txeo::detail {

class TensorHelper {
  public:
    TensorHelper();
    TensorHelper(const TensorHelper &) = delete;
    TensorHelper(TensorHelper &&) = delete;
    TensorHelper &operator=(const TensorHelper &) = delete;
    TensorHelper &operator=(TensorHelper &&) = delete;
    ~TensorHelper();

    using ReduFunc31 =
        std::function<tf::Output(const tf::Scope &scope, tf::Input input, int64_t axis)>;

    using ReduFunc3 =
        std::function<tf::Output(const tf::Scope &scope, tf::Input input, tf::Input axis)>;

    using ReduFunc2 = std::function<tf::Output(const tf::Scope &scope, tf::Input input)>;

    using OpeFunc =
        std::function<tf::Output(const tf::Scope &scope, tf::Input left, tf::Input right)>;

    template <typename T, typename U>
    static txeo::Tensor<T> to_txeo_tensor(U &&tf_tensor);

    template <typename T, typename U>
    static txeo::Matrix<T> to_txeo_matrix(U &&tf_tensor);

    template <typename T>
    static txeo::Tensor<T> reduce_tensor(const tf::Tensor &M, const std::vector<size_t> &axes,
                                         ReduFunc3 func);

    template <typename T>
    static txeo::Tensor<T> reduce_tensor(const tf::Tensor &M, ReduFunc2 func);

    template <typename T>
    static txeo::Tensor<size_t> reduce_tensor_to_indexes(const tf::Tensor &M, int64_t index,
                                                         ReduFunc31 func);

    template <typename T>
    static txeo::Tensor<T> reduce_tensor(const tf::Tensor &M, const int64_t &axis, ReduFunc31 func);

    template <typename T>
    static txeo::Tensor<T> ope_tensors(const tf::Tensor &M, const tf::Tensor &N, OpeFunc func);
};

template <typename T, typename U>
txeo::Tensor<T> TensorHelper::to_txeo_tensor(U &&tf_tensor) {
  txeo::Tensor<T> resp;
  resp._impl->tf_tensor = std::make_unique<tensorflow::Tensor>(std::forward<U>(tf_tensor));
  resp._impl->txeo_shape._impl->tf_shape = nullptr;
  resp._impl->txeo_shape._impl->ext_tf_shape = &resp._impl->tf_tensor->shape();
  resp._impl->txeo_shape._impl->stride =
      txeo::detail::calc_stride(*resp._impl->txeo_shape._impl->ext_tf_shape);

  return resp;
}

template <typename T, typename U>
txeo::Matrix<T> TensorHelper::to_txeo_matrix(U &&tf_tensor) {
  txeo::Matrix<T> resp;
  resp._impl->tf_tensor = std::make_unique<tensorflow::Tensor>(std::forward<U>(tf_tensor));
  resp._impl->txeo_shape._impl->ext_tf_shape = &resp._impl->tf_tensor->shape();
  resp._impl->txeo_shape._impl->stride =
      txeo::detail::calc_stride(*resp._impl->txeo_shape._impl->ext_tf_shape);

  return resp;
}

template <typename T>
txeo::Tensor<T> TensorHelper::reduce_tensor(const tf::Tensor &M, const std::vector<size_t> &axes,
                                            ReduFunc3 func) {
  tf::Scope root = tf::Scope::NewRootScope();

  auto vec = txeo::detail::to_int64(axes);
  tf::Tensor axes_tensor(tf::DT_INT64, tf::TensorShape({txeo::detail::to_int64(vec.size())}));
  auto axes_tensor_flat = axes_tensor.flat<int64_t>();
  for (size_t i = 0; i < vec.size(); ++i)
    axes_tensor_flat(i) = vec[i];

  auto ope = func(root, M, axes_tensor);

  tf::ClientSession session(root);
  std::vector<tf::Tensor> outputs;
  auto status = session.Run({ope}, &outputs);
  if (!status.ok())
    throw std::runtime_error(status.ToString());
  return txeo::detail::TensorHelper::to_txeo_tensor<T>(std::move(outputs[0]));
}

template <typename T>
txeo::Tensor<T> TensorHelper::reduce_tensor(const tf::Tensor &M, ReduFunc2 func) {
  tf::Scope root = tf::Scope::NewRootScope();

  auto ope = func(root, M);

  tf::ClientSession session(root);
  std::vector<tf::Tensor> outputs;
  auto status = session.Run({ope}, &outputs);
  if (!status.ok())
    throw std::runtime_error(status.ToString());
  return txeo::detail::TensorHelper::to_txeo_tensor<T>(std::move(outputs[0]));
}

template <typename T>
txeo::Tensor<size_t> TensorHelper::reduce_tensor_to_indexes(const tf::Tensor &M, int64_t index,
                                                            ReduFunc31 func) {
  tf::Scope root = tf::Scope::NewRootScope();

  auto ope = func(root, M, index);

  tf::ClientSession session(root);
  std::vector<tf::Tensor> outputs;
  auto status = session.Run({ope}, &outputs);
  if (!status.ok())
    throw std::runtime_error(status.ToString());

  auto resp = txeo::detail::TensorHelper::to_txeo_tensor<size_t>(std::move(outputs[0]));

  return resp;
}

template <typename T>
inline txeo::Tensor<T> TensorHelper::reduce_tensor(const tf::Tensor &M, const int64_t &axis,
                                                   ReduFunc31 func) {
  tf::Scope root = tf::Scope::NewRootScope();

  auto ope = func(root, M, axis);

  tf::ClientSession session(root);
  std::vector<tf::Tensor> outputs;
  auto status = session.Run({ope}, &outputs);
  if (!status.ok())
    throw std::runtime_error(status.ToString());

  auto resp = txeo::detail::TensorHelper::to_txeo_tensor<T>(std::move(outputs[0]));

  return resp;
}

template <typename T>
txeo::Tensor<T> TensorHelper::ope_tensors(const tf::Tensor &M, const tf::Tensor &N, OpeFunc func) {
  tf::Scope root = tf::Scope::NewRootScope();

  auto ope = func(root, M, N);

  tf::ClientSession session(root);
  std::vector<tf::Tensor> outputs;
  auto status = session.Run({ope}, &outputs);
  if (!status.ok())
    throw std::runtime_error(status.ToString());

  auto resp = txeo::detail::TensorHelper::to_txeo_tensor<T>(std::move(outputs[0]));

  return resp;
}

} // namespace txeo::detail
#endif