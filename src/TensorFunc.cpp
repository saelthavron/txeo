#include "txeo/TensorFunc.h"
#include "txeo/Tensor.h"
#include "txeo/TensorOp.h"
#include "txeo/detail/TensorHelper.h"
#include "txeo/detail/TensorPrivate.h"
#include "txeo/detail/utils.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tensorflow/cc/ops/array_ops.h>
#include <utility>

namespace txeo {

template <typename T>
txeo::Tensor<T> TensorFunc<T>::power_elem(const txeo::Tensor<T> &tensor, const T &exponent) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");

  txeo::Tensor<T> resp(tensor.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = static_cast<T>(std::pow(tensor.data()[i], exponent));

  return resp;
}

template <typename T>
txeo::Tensor<T> TensorFunc<T>::power_elem_by(txeo::Tensor<T> &tensor, const T &exponent) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");

  for (size_t i{0}; i < tensor.dim(); ++i)
    tensor.data()[i] = static_cast<T>(std::pow(tensor.data()[i], exponent));

  return tensor;
}

template <typename T>
txeo::Tensor<T> TensorFunc<T>::square(const txeo::Tensor<T> &tensor) {
  auto resp = TensorOp<T>::hadamard_prod(tensor, tensor);
  return resp;
}

template <typename T>
txeo::Tensor<T> &TensorFunc<T>::square_by(txeo::Tensor<T> &tensor) {
  TensorOp<T>::hadamard_prod_by(tensor, tensor);
  return tensor;
}

template <typename T>
txeo::Tensor<T> TensorFunc<T>::sqrt(const txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");

  txeo::Tensor<T> resp(tensor.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = static_cast<T>(std::sqrt(tensor.data()[i]));

  return resp;
}

template <typename T>
inline txeo::Tensor<T> &TensorFunc<T>::sqrt_by(txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");

  for (size_t i{0}; i < tensor.dim(); ++i)
    tensor.data()[i] = static_cast<T>(std::sqrt(tensor.data()[i]));

  return tensor;
}

template <typename T>
inline txeo::Tensor<T> TensorFunc<T>::abs(const txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");

  try {
    auto resp = txeo::detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, [](const tf::Scope &scope, tf::Input input) -> tf::Output {
          return tf::ops::Abs(scope, input);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw txeo::TensorFuncError("Reduction error: " + std::string{e.what()});
  }
}

template <typename T>
inline txeo::Tensor<T> &TensorFunc<T>::abs_by(txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");

  for (size_t i{0}; i < tensor.dim(); ++i)
    tensor.data()[i] = static_cast<T>(std::abs(tensor.data()[i]));

  return tensor;
}

template <typename T>
txeo::Tensor<T> TensorFunc<T>::permute(const txeo::Tensor<T> &tensor,
                                       const std::vector<size_t> &axes) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");

  if (tensor.order() != txeo::detail::to_int(axes.size()))
    throw txeo::TensorFuncError("Tensor order and number of axes are different.");

  for (auto &item : axes)
    if (item >= txeo::detail::to_size_t(tensor.order()))
      throw txeo::TensorFuncError("Inconsistent axes.");

  tf::Tensor perm(tf::DT_INT64, tf::TensorShape({txeo::detail::to_int64(axes.size())}));
  auto perm_flat = perm.flat<int64_t>();
  for (int64_t i{0}; i < txeo::detail::to_int64(axes.size()); ++i)
    perm_flat(i) = txeo::detail::to_int64((axes[i]));

  return txeo::detail::TensorHelper::ope_tensors<T>(
      *tensor._impl->tf_tensor, perm, [](const tf::Scope &scope, tf::Input left, tf::Input right) {
        return tf::ops::Transpose(scope, left, right);
      });
}

template <typename T>
txeo::Tensor<T> &TensorFunc<T>::permute_by(txeo::Tensor<T> &tensor,
                                           const std::vector<size_t> &axes) {
  tensor = std::move(TensorFunc<T>::permute(tensor, axes));
  return tensor;
}

template <typename T>
txeo::Matrix<T> TensorFunc<T>::transpose(const txeo::Matrix<T> &matrix) {
  return txeo::Matrix(TensorFunc<T>::permute(matrix, {1, 0}));
}

template <typename T>
inline txeo::Matrix<T> &TensorFunc<T>::transpose_by(txeo::Matrix<T> &matrix) {
  matrix = std::move(TensorFunc<T>::transpose(matrix));
  return matrix;
}

// Type specialization to avoid calling abs for unsigned types
template <>
inline txeo::Tensor<bool> &TensorFunc<bool>::abs_by(txeo::Tensor<bool> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");
  return tensor;
}

// Type specialization to avoid calling abs for unsigned types
template <>
inline txeo::Tensor<size_t> &TensorFunc<size_t>::abs_by(txeo::Tensor<size_t> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorFuncError("Tensor has dimension zero.");
  return tensor;
}

template class TensorFunc<size_t>;
template class TensorFunc<short>;
template class TensorFunc<int>;
template class TensorFunc<bool>;
template class TensorFunc<long>;
template class TensorFunc<long long>;
template class TensorFunc<float>;
template class TensorFunc<double>;

} // namespace txeo