#include "txeo/TensorFunc.h"
#include "txeo/TensorOp.h"
#include "txeo/detail/TensorHelper.h"

#include <cmath>

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

template class TensorFunc<short>;
template class TensorFunc<int>;
template class TensorFunc<bool>;
template class TensorFunc<long>;
template class TensorFunc<long long>;
template class TensorFunc<float>;
template class TensorFunc<double>;
template class TensorFunc<size_t>;

} // namespace txeo