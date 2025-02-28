#include "txeo/TensorAgg.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "txeo/Tensor.h"
#include "txeo/detail/TensorPrivate.h"
#include "txeo/detail/TensorShapePrivate.h"
#include "txeo/detail/utils.h"

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

namespace txeo {

namespace tf = tensorflow;

template <typename T>
txeo::Tensor<T> TensorAgg<T>::reduce_sum(const txeo::Tensor<T> &tensor,
                                         std::initializer_list<size_t> axes) {

  tf::Scope root = tf::Scope::NewRootScope();

  auto sum_op = tf::ops::Sum(root, *tensor._impl->tf_tensor, tf::ops::Const(root, axes));

  tf::ClientSession session(root);
  std::vector<tf::Tensor> outputs;
  auto status = session.Run({sum_op}, &outputs);
  if (!status.ok())
    txeo::TensorAggError("Error reducing tensor: " + status.ToString());

  Tensor<T> resp;

  //  auto resp = txeo::detail::to_txeo_tensor<T>(std::move(outputs[0]));

  return resp;
}

template class TensorAgg<short>;
template class TensorAgg<int>;
template class TensorAgg<bool>;
template class TensorAgg<long>;
template class TensorAgg<long long>;
template class TensorAgg<float>;
template class TensorAgg<double>;

} // namespace txeo