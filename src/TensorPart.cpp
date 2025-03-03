#include "txeo/TensorPart.h"
#include "txeo/detail/TensorHelper.h"
#include "txeo/detail/utils.h"

#include <cstddef>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/framework/ops.h>
#include <tensorflow/cc/framework/scope.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

namespace tf = tensorflow;

namespace txeo {

template <typename T>
std::vector<txeo::Tensor<T>> TensorPart<T>::unstack(const txeo::Tensor<T> &tensor, size_t axis) {
  if (axis >= txeo::detail::to_size_t(tensor.order()))
    throw txeo::TensorError("Axis inconsistent with the orderof this tensor!");

  auto shp = tensor.shape();

  auto root = tf::Scope::NewRootScope();

  auto aux = tf::ops::Unstack(root, *tensor._impl->tf_tensor, shp.axis_dim(axis),
                              tf::ops::Unstack::Attrs().Axis(txeo::detail::to_int64(axis)));

  tf::ClientSession session(root);
  std::vector<tf::Tensor> outputs;
  auto status = session.Run({aux.output}, &outputs);
  if (!status.ok())
    throw txeo::TensorPartError("This tensor could not be unstacked: " + status.ToString());

  std::vector<txeo::Tensor<T>> resp;
  for (auto &item : outputs)
    resp.emplace_back(txeo::detail::TensorHelper::to_txeo_tensor<T>(std::move(item)));

  return resp;
}

template <typename T>
inline txeo::Tensor<T> TensorPart<T>::slice(const txeo::Tensor<T> &tensor, size_t first_axis_begin,
                                            size_t first_axis_end) {
  if (first_axis_end < first_axis_begin)
    throw txeo::TensorError("The end index can not be less than the initial index!");
  if (txeo::detail::to_int64(first_axis_end) > tensor._impl->txeo_shape.axis_dim(0))
    throw txeo::TensorPartError(
        "The end index can not be greater than or equal to the dimension of first axis!");

  auto t_slice = tensor._impl->tf_tensor->Slice(first_axis_begin, first_axis_end);
  Tensor<T> resp{txeo::detail::to_txeo_tensor_shape(t_slice.shape())};
  if (!resp._impl->tf_tensor->CopyFrom(t_slice, t_slice.shape()))
    throw txeo::TensorPartError("This tensor could not be sliced!");

  return resp;
}

template <typename T>
inline std::vector<txeo::Tensor<T>> TensorPart<T>::unstack(const txeo::Tensor<T> &tensor,
                                                           std::vector<size_t> axes) {

  auto unstacked = TensorPart<T>::unstack(tensor, axes[0]);
  for (size_t j{0}; j < unstacked.size(); ++j) {
  }
}

template class TensorPart<short>;
template class TensorPart<int>;
template class TensorPart<bool>;
template class TensorPart<long>;
template class TensorPart<long long>;
template class TensorPart<float>;
template class TensorPart<double>;
template class TensorPart<size_t>;

} // namespace txeo