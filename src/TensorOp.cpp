#include "txeo/TensorOp.h"
#include "txeo/detail/TensorHelper.h"
#include "txeo/detail/utils.h"

#include <cmath>

namespace txeo {

template <typename T>
txeo::Tensor<T> TensorOp<T>::sum(const txeo::Tensor<T> &left, const txeo::Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw txeo::TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw txeo::TensorOpError("Operands have different shapes.");

  txeo::Tensor<T> resp(left.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = left.data()[i] + right.data()[i];

  return resp;
}

template <typename T>
void TensorOp<T>::sum_by(txeo::Tensor<T> &left, const txeo::Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw txeo::TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw txeo::TensorOpError("Operands have different shapes.");

  for (size_t i{0}; i < left.dim(); ++i)
    left.data()[i] += right.data()[i];
}

template <typename T>
txeo::Tensor<T> TensorOp<T>::sum(const txeo::Tensor<T> &left, const T &right) {
  if (left.dim() == 0)
    throw txeo::TensorOpError("Left operand has dimension zero.");

  txeo::Tensor<T> resp(left.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = left.data()[i] + right;

  return resp;
}

template <typename T>
void TensorOp<T>::sum_by(txeo::Tensor<T> &left, const T &right) {
  if (left.dim() == 0)
    throw txeo::TensorOpError("Left operand has dimension zero.");

  for (size_t i{0}; i < left.dim(); ++i)
    left.data()[i] += right;
}

template <typename T>
txeo::Tensor<T> TensorOp<T>::subtract(const txeo::Tensor<T> &left, const txeo::Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw txeo::TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw txeo::TensorOpError("Operands have different shapes.");

  txeo::Tensor<T> resp(left.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = left.data()[i] - right.data()[i];

  return resp;
}

template <typename T>
void TensorOp<T>::subtract_by(txeo::Tensor<T> &left, const txeo::Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw txeo::TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw txeo::TensorOpError("Operands have different shapes.");

  for (size_t i{0}; i < left.dim(); ++i)
    left.data()[i] -= right.data()[i];
}

template <typename T>
txeo::Tensor<T> TensorOp<T>::subtract(const txeo::Tensor<T> &left, const T &right) {
  if (left.dim() == 0)
    throw txeo::TensorOpError("Left operand has dimension zero.");

  txeo::Tensor<T> resp(left.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = left.data()[i] - right;

  return resp;
}

template <typename T>
void TensorOp<T>::subtract_by(txeo::Tensor<T> &left, const T &right) {
  if (left.dim() == 0)
    throw txeo::TensorOpError("Left operand has dimension zero.");

  for (size_t i{0}; i < left.dim(); ++i)
    left.data()[i] -= right;
}

template <typename T>
txeo::Tensor<T> TensorOp<T>::subtract(const T &left, const txeo::Tensor<T> &right) {
  if (right.dim() == 0)
    throw txeo::TensorOpError("Right operand has dimension zero.");

  txeo::Tensor<T> resp(right.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = left - right.data()[i];

  return resp;
}

template <typename T>
void TensorOp<T>::subtract_by(const T &left, txeo::Tensor<T> &right) {
  if (right.dim() == 0)
    throw txeo::TensorOpError("Right operand has dimension zero.");

  for (size_t i{0}; i < right.dim(); ++i)
    right.data()[i] = left - right.data()[i];
}

template <typename T>
txeo::Tensor<T> TensorOp<T>::multiply(const txeo::Tensor<T> &tensor, const T &scalar) {
  if (tensor.dim() == 0)
    throw txeo::TensorOpError("Tensor has dimension zero.");

  txeo::Tensor<T> resp(tensor.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = tensor.data()[i] * scalar;

  return resp;
}

template <typename T>
void TensorOp<T>::multiply_by(txeo::Tensor<T> &tensor, const T &scalar) {
  if (tensor.dim() == 0)
    throw txeo::TensorOpError("Tensor has dimension zero.");

  for (size_t i{0}; i < tensor.dim(); ++i)
    tensor.data()[i] *= scalar;
}

template <typename T>
txeo::Tensor<T> TensorOp<T>::divide(const txeo::Tensor<T> &tensor, const T &scalar) {
  if (txeo::detail::is_zero(scalar))
    throw txeo::TensorOpError("Denominator is zero.");
  if (tensor.dim() == 0)
    throw txeo::TensorOpError("Tensor has dimension zero.");

  txeo::Tensor<T> resp(tensor.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = tensor.data()[i] / scalar;

  return resp;
}

template <typename T>
void TensorOp<T>::divide_by(txeo::Tensor<T> &tensor, const T &scalar) {
  if (txeo::detail::is_zero(scalar))
    throw txeo::TensorOpError("Denominator is zero.");
  if (tensor.dim() == 0)
    throw txeo::TensorOpError("Tensor has dimension zero.");

  for (size_t i{0}; i < tensor.dim(); ++i)
    tensor.data()[i] /= scalar;
}

template <typename T>
inline txeo::Tensor<T> TensorOp<T>::divide(const T &scalar, const txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorOpError("Right operand has dimension zero.");

  txeo::Tensor<T> resp(tensor.shape());
  for (size_t i{0}; i < resp.dim(); ++i) {
    if (txeo::detail::is_zero(tensor.data()[i]))
      throw txeo::TensorOpError("Zero element in right operand.");
    resp.data()[i] = scalar / tensor.data()[i];
  }

  return resp;
}

template <typename T>
inline void TensorOp<T>::divide_by(const T &scalar, txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorOpError("Tensor has dimension zero.");

  for (size_t i{0}; i < tensor.dim(); ++i) {
    if (txeo::detail::is_zero(tensor.data()[i]))
      throw txeo::TensorOpError("Zero element in right operand.");
    tensor.data()[i] = scalar / tensor.data()[i];
  }
}

template <typename T>
txeo::Tensor<T> TensorOp<T>::hadamard_prod(const txeo::Tensor<T> &left,
                                           const txeo::Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw txeo::TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw txeo::TensorOpError("Operands have different shapes.");

  txeo::Tensor<T> resp(left.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = left.data()[i] * right.data()[i];

  return resp;
}

template <typename T>
void TensorOp<T>::hadamard_prod_by(txeo::Tensor<T> &left, const txeo::Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw txeo::TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw txeo::TensorOpError("Operands have different shapes.");

  for (size_t i{0}; i < left.dim(); ++i)
    left.data()[i] *= right.data()[i];
}

template <typename T>
inline txeo::Tensor<T> TensorOp<T>::hadamard_div(const txeo::Tensor<T> &left,
                                                 const txeo::Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw txeo::TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw txeo::TensorOpError("Operands have different shapes.");

  txeo::Tensor<T> resp(left.shape());
  for (size_t i{0}; i < resp.dim(); ++i) {
    if (txeo::detail::is_zero(right.data()[i]))
      throw txeo::TensorOpError("Zero element in right operand.");
    resp.data()[i] = left.data()[i] / right.data()[i];
  }

  return resp;
}

template <typename T>
inline void TensorOp<T>::hadamard_div_by(txeo::Tensor<T> &left, const txeo::Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw txeo::TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw txeo::TensorOpError("Operands have different shapes.");

  for (size_t i{0}; i < left.dim(); ++i) {
    if (txeo::detail::is_zero(right.data()[i]))
      throw txeo::TensorOpError("Zero element in right operand.");
    left.data()[i] /= right.data()[i];
  }
}

template <typename T>
txeo::Tensor<T> TensorOp<T>::power_elem(const txeo::Tensor<T> &tensor, const T &exponent) {
  if (tensor.dim() == 0)
    throw txeo::TensorOpError("Tensor has dimension zero.");

  txeo::Tensor<T> resp(tensor.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = static_cast<T>(std::pow(tensor.data()[i], exponent));

  return resp;
}

template <typename T>
void TensorOp<T>::power_elem_by(txeo::Tensor<T> &tensor, const T &exponent) {
  if (tensor.dim() == 0)
    throw txeo::TensorOpError("Tensor has dimension zero.");

  for (size_t i{0}; i < tensor.dim(); ++i)
    tensor.data()[i] = static_cast<T>(std::pow(tensor.data()[i], exponent));
}

template <typename T>
txeo::Tensor<T> TensorOp<T>::square(const txeo::Tensor<T> &tensor) {
  auto resp = TensorOp<T>::hadamard_prod(tensor, tensor);
  return resp;
}

template <typename T>
void TensorOp<T>::square_by(txeo::Tensor<T> &tensor) {
  TensorOp<T>::hadamard_prod_by(tensor, tensor);
}

template <typename T>
txeo::Tensor<T> TensorOp<T>::sqrt(const txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorOpError("Tensor has dimension zero.");

  txeo::Tensor<T> resp(tensor.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = static_cast<T>(std::sqrt(tensor.data()[i]));

  return resp;
}

template <typename T>
inline void TensorOp<T>::sqrt_by(txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorOpError("Tensor has dimension zero.");

  for (size_t i{0}; i < tensor.dim(); ++i)
    tensor.data()[i] = static_cast<T>(std::sqrt(tensor.data()[i]));
}

template <typename T>
inline txeo::Tensor<T> TensorOp<T>::abs(const txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorOpError("Tensor has dimension zero.");

  try {
    auto resp = txeo::detail::TensorHelper::reduce_tensor<T>(
        *tensor._impl->tf_tensor, [](const tf::Scope &scope, tf::Input input) -> tf::Output {
          return tf::ops::Abs(scope, input);
        });
    return resp;
  } catch (std::runtime_error e) {
    throw txeo::TensorOpError("Reduction error: " + std::string{e.what()});
  }
}

// Required for templated elements in cpp files

template class TensorOp<short>;
template class TensorOp<int>;
template class TensorOp<bool>;
template class TensorOp<long>;
template class TensorOp<long long>;
template class TensorOp<float>;
template class TensorOp<double>;
template class TensorOp<size_t>;

} // namespace txeo
