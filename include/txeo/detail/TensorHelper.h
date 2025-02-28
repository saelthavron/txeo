#ifndef TENSORHELPER_H
#define TENSORHELPER_H
#pragma once

#include "txeo/detail/TensorPrivate.h"
#include "txeo/detail/TensorShapePrivate.h"
#include "txeo/detail/utils.h"

#include <cstddef>
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

    using ReduFunc3 =
        std::function<tf::Output(const tf::Scope &scope, tf::Input input, tf::Input axis)>;

    using ReduFunc2 = std::function<tf::Output(const tf::Scope &scope, tf::Input input)>;

    template <typename T, typename U>
    static txeo::Tensor<T> to_txeo_tensor(U &&tf_tensor) {
      txeo::Tensor<T> resp;
      resp._impl->tf_tensor = std::make_unique<tensorflow::Tensor>(std::forward<U>(tf_tensor));
      resp._impl->txeo_shape._impl->ext_tf_shape = &resp._impl->tf_tensor->shape();
      resp._impl->txeo_shape._impl->stride =
          txeo::detail::calc_stride(*resp._impl->txeo_shape._impl->ext_tf_shape);

      return resp;
    }

    template <typename T>
    static txeo::Tensor<T> reduce_tensor(const tf::Tensor &M, const std::vector<size_t> &axes,
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
    static txeo::Tensor<T> reduce_tensor(const tf::Tensor &M, ReduFunc2 func) {
      tf::Scope root = tf::Scope::NewRootScope();

      auto ope = func(root, M);

      tf::ClientSession session(root);
      std::vector<tf::Tensor> outputs;
      auto status = session.Run({ope}, &outputs);
      if (!status.ok())
        throw std::runtime_error(status.ToString());
      return txeo::detail::TensorHelper::to_txeo_tensor<T>(std::move(outputs[0]));
    }
};
} // namespace txeo::detail
#endif