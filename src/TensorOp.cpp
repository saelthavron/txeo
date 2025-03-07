#include "txeo/TensorOp.h"
#include "txeo/Matrix.h"
#include "txeo/Tensor.h"
#include "txeo/detail/TensorHelper.h"
#include "txeo/detail/utils.h"

#include <cmath>
#include <tensorflow/cc/ops/math_ops.h>

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
txeo::Tensor<T> &TensorOp<T>::sum_by(txeo::Tensor<T> &left, const txeo::Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw txeo::TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw txeo::TensorOpError("Operands have different shapes.");

  for (size_t i{0}; i < left.dim(); ++i)
    left.data()[i] += right.data()[i];

  return left;
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
txeo::Tensor<T> &TensorOp<T>::sum_by(txeo::Tensor<T> &left, const T &right) {
  if (left.dim() == 0)
    throw txeo::TensorOpError("Left operand has dimension zero.");

  for (size_t i{0}; i < left.dim(); ++i)
    left.data()[i] += right;

  return left;
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
txeo::Tensor<T> &TensorOp<T>::subtract_by(txeo::Tensor<T> &left, const txeo::Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw txeo::TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw txeo::TensorOpError("Operands have different shapes.");

  for (size_t i{0}; i < left.dim(); ++i)
    left.data()[i] -= right.data()[i];

  return left;
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
txeo::Tensor<T> &TensorOp<T>::subtract_by(txeo::Tensor<T> &left, const T &right) {
  if (left.dim() == 0)
    throw txeo::TensorOpError("Left operand has dimension zero.");

  for (size_t i{0}; i < left.dim(); ++i)
    left.data()[i] -= right;

  return left;
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
const T &TensorOp<T>::subtract_by(const T &left, txeo::Tensor<T> &right) {
  if (right.dim() == 0)
    throw txeo::TensorOpError("Right operand has dimension zero.");

  for (size_t i{0}; i < right.dim(); ++i)
    right.data()[i] = left - right.data()[i];

  return left;
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
txeo::Tensor<T> &TensorOp<T>::multiply_by(txeo::Tensor<T> &tensor, const T &scalar) {
  if (tensor.dim() == 0)
    throw txeo::TensorOpError("Tensor has dimension zero.");

  for (size_t i{0}; i < tensor.dim(); ++i)
    tensor.data()[i] *= scalar;

  return tensor;
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
txeo::Tensor<T> &TensorOp<T>::divide_by(txeo::Tensor<T> &tensor, const T &scalar) {
  if (txeo::detail::is_zero(scalar))
    throw txeo::TensorOpError("Denominator is zero.");
  if (tensor.dim() == 0)
    throw txeo::TensorOpError("Tensor has dimension zero.");

  for (size_t i{0}; i < tensor.dim(); ++i)
    tensor.data()[i] /= scalar;

  return tensor;
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
inline txeo::Tensor<T> &TensorOp<T>::divide_by(const T &scalar, txeo::Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw txeo::TensorOpError("Tensor has dimension zero.");

  for (size_t i{0}; i < tensor.dim(); ++i) {
    if (txeo::detail::is_zero(tensor.data()[i]))
      throw txeo::TensorOpError("Zero element in right operand.");
    tensor.data()[i] = scalar / tensor.data()[i];
  }

  return tensor;
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
txeo::Tensor<T> &TensorOp<T>::hadamard_prod_by(txeo::Tensor<T> &left,
                                               const txeo::Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw txeo::TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw txeo::TensorOpError("Operands have different shapes.");

  for (size_t i{0}; i < left.dim(); ++i)
    left.data()[i] *= right.data()[i];

  return left;
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
inline txeo::Tensor<T> &TensorOp<T>::hadamard_div_by(txeo::Tensor<T> &left,
                                                     const txeo::Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw txeo::TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw txeo::TensorOpError("Operands have different shapes.");

  for (size_t i{0}; i < left.dim(); ++i) {
    if (txeo::detail::is_zero(right.data()[i]))
      throw txeo::TensorOpError("Zero element in right operand.");
    left.data()[i] /= right.data()[i];
  }

  return left;
}

template <typename T>
inline txeo::Matrix<T> TensorOp<T>::product(const txeo::Matrix<T> &left,
                                            const txeo::Matrix<T> &right) {

  if (left.dim() == 0 || right.dim() == 0)
    throw txeo::TensorOpError("One of the operands has dimension zero.");

  if (left.shape().axis_dim(1) != right.shape().axis_dim(0))
    throw txeo::TensorOpError("Operands are incompatible.");

  auto aux = txeo::detail::TensorHelper::ope_tensors<T>(
      *left._impl->tf_tensor, *right._impl->tf_tensor,
      [](const tf::Scope &scope, tf::Input left, tf::Input right) {
        return tf::ops::MatMul(scope, left, right);
      });

  return txeo::Matrix<T>(std::move(aux));
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
