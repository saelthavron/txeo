#include "txeo/TensorOp.h"
#include "txeo/Matrix.h"
#include "txeo/Tensor.h"
#include "txeo/Vector.h"
#include "txeo/detail/TensorHelper.h"
#include "txeo/detail/utils.h"

#include <cmath>

#include <tensorflow/cc/ops/math_ops.h>

namespace txeo {

template <typename T>
Tensor<T> TensorOp<T>::sum(const Tensor<T> &left, const Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw TensorOpError("Operands have different shapes.");

  Tensor<T> resp(left.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = left.data()[i] + right.data()[i];

  return resp;
}

template <typename T>
Tensor<T> &TensorOp<T>::sum_by(Tensor<T> &left, const Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw TensorOpError("Operands have different shapes.");

  for (size_t i{0}; i < left.dim(); ++i)
    left.data()[i] += right.data()[i];

  return left;
}

template <typename T>
Tensor<T> TensorOp<T>::sum(const Tensor<T> &left, const T &right) {
  if (left.dim() == 0)
    throw TensorOpError("Left operand has dimension zero.");

  Tensor<T> resp(left.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = left.data()[i] + right;

  return resp;
}

template <typename T>
Tensor<T> &TensorOp<T>::sum_by(Tensor<T> &left, const T &right) {
  if (left.dim() == 0)
    throw TensorOpError("Left operand has dimension zero.");

  for (size_t i{0}; i < left.dim(); ++i)
    left.data()[i] += right;

  return left;
}

template <typename T>
Tensor<T> TensorOp<T>::subtract(const Tensor<T> &left, const Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw TensorOpError("Operands have different shapes.");

  Tensor<T> resp(left.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = left.data()[i] - right.data()[i];

  return resp;
}

template <typename T>
Tensor<T> &TensorOp<T>::subtract_by(Tensor<T> &left, const Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw TensorOpError("Operands have different shapes.");

  for (size_t i{0}; i < left.dim(); ++i)
    left.data()[i] -= right.data()[i];

  return left;
}

template <typename T>
Tensor<T> TensorOp<T>::subtract(const Tensor<T> &left, const T &right) {
  if (left.dim() == 0)
    throw TensorOpError("Left operand has dimension zero.");

  Tensor<T> resp(left.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = left.data()[i] - right;

  return resp;
}

template <typename T>
Tensor<T> &TensorOp<T>::subtract_by(Tensor<T> &left, const T &right) {
  if (left.dim() == 0)
    throw TensorOpError("Left operand has dimension zero.");

  for (size_t i{0}; i < left.dim(); ++i)
    left.data()[i] -= right;

  return left;
}

template <typename T>
Tensor<T> TensorOp<T>::subtract(const T &left, const Tensor<T> &right) {
  if (right.dim() == 0)
    throw TensorOpError("Right operand has dimension zero.");

  Tensor<T> resp(right.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = left - right.data()[i];

  return resp;
}

template <typename T>
const T &TensorOp<T>::subtract_by(const T &left, Tensor<T> &right) {
  if (right.dim() == 0)
    throw TensorOpError("Right operand has dimension zero.");

  for (size_t i{0}; i < right.dim(); ++i)
    right.data()[i] = left - right.data()[i];

  return left;
}

template <typename T>
Tensor<T> TensorOp<T>::multiply(const Tensor<T> &tensor, const T &scalar) {
  if (tensor.dim() == 0)
    throw TensorOpError("Tensor has dimension zero.");

  Tensor<T> resp(tensor.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = tensor.data()[i] * scalar;

  return resp;
}

template <typename T>
Tensor<T> &TensorOp<T>::multiply_by(Tensor<T> &tensor, const T &scalar) {
  if (tensor.dim() == 0)
    throw TensorOpError("Tensor has dimension zero.");

  for (size_t i{0}; i < tensor.dim(); ++i)
    tensor.data()[i] *= scalar;

  return tensor;
}

template <typename T>
Tensor<T> TensorOp<T>::divide(const Tensor<T> &tensor, const T &scalar) {
  if (detail::is_zero(scalar))
    throw TensorOpError("Denominator is zero.");
  if (tensor.dim() == 0)
    throw TensorOpError("Tensor has dimension zero.");

  Tensor<T> resp(tensor.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = tensor.data()[i] / scalar;

  return resp;
}

template <typename T>
Tensor<T> &TensorOp<T>::divide_by(Tensor<T> &tensor, const T &scalar) {
  if (detail::is_zero(scalar))
    throw TensorOpError("Denominator is zero.");
  if (tensor.dim() == 0)
    throw TensorOpError("Tensor has dimension zero.");

  for (size_t i{0}; i < tensor.dim(); ++i)
    tensor.data()[i] /= scalar;

  return tensor;
}

template <typename T>
Tensor<T> TensorOp<T>::divide(const T &scalar, const Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw TensorOpError("Right operand has dimension zero.");

  Tensor<T> resp(tensor.shape());
  for (size_t i{0}; i < resp.dim(); ++i) {
    if (detail::is_zero(tensor.data()[i]))
      throw TensorOpError("Zero element in right operand.");
    resp.data()[i] = scalar / tensor.data()[i];
  }

  return resp;
}

template <typename T>
Tensor<T> &TensorOp<T>::divide_by(const T &scalar, Tensor<T> &tensor) {
  if (tensor.dim() == 0)
    throw TensorOpError("Tensor has dimension zero.");

  for (size_t i{0}; i < tensor.dim(); ++i) {
    if (detail::is_zero(tensor.data()[i]))
      throw TensorOpError("Zero element in right operand.");
    tensor.data()[i] = scalar / tensor.data()[i];
  }

  return tensor;
}

template <typename T>
Tensor<T> TensorOp<T>::hadamard_prod(const Tensor<T> &left, const Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw TensorOpError("Operands have different shapes.");

  Tensor<T> resp(left.shape());
  for (size_t i{0}; i < resp.dim(); ++i)
    resp.data()[i] = left.data()[i] * right.data()[i];

  return resp;
}

template <typename T>
Tensor<T> &TensorOp<T>::hadamard_prod_by(Tensor<T> &left, const Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw TensorOpError("Operands have different shapes.");

  for (size_t i{0}; i < left.dim(); ++i)
    left.data()[i] *= right.data()[i];

  return left;
}

template <typename T>
Tensor<T> TensorOp<T>::hadamard_div(const Tensor<T> &left, const Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw TensorOpError("Operands have different shapes.");

  Tensor<T> resp(left.shape());
  for (size_t i{0}; i < resp.dim(); ++i) {
    if (detail::is_zero(right.data()[i]))
      throw TensorOpError("Zero element in right operand.");
    resp.data()[i] = left.data()[i] / right.data()[i];
  }

  return resp;
}

template <typename T>
Tensor<T> &TensorOp<T>::hadamard_div_by(Tensor<T> &left, const Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw TensorOpError("One of the operands has dimension zero.");
  if (left.shape() != right.shape())
    throw TensorOpError("Operands have different shapes.");

  for (size_t i{0}; i < left.dim(); ++i) {
    if (detail::is_zero(right.data()[i]))
      throw TensorOpError("Zero element in right operand.");
    left.data()[i] /= right.data()[i];
  }

  return left;
}

template <typename T>
T TensorOp<T>::inner(const Tensor<T> &left, const Tensor<T> &right) {
  if (left.dim() == 0 || right.dim() == 0)
    throw TensorOpError("One of the operands has dimension zero.");

  if (left.dim() != right.dim())
    throw TensorOpError("Operands are incompatible.");

  auto l_data = left.data();
  auto r_data = right.data();

  T resp = 0.0;
  for (size_t i{0}; i < left.dim(); ++i)
    resp += l_data[i] * r_data[i];

  return resp;
}

template <typename T>
Tensor<T> TensorOp<T>::product_tensors(const Tensor<T> &left, const Tensor<T> &right) {

  if (left.dim() == 0 || right.dim() == 0)
    throw TensorOpError("One of the operands has dimension zero.");

  if (left.order() != 2 || right.order() != 2)
    throw TensorOpError("One of the operands is not a matrix.");

  if (left.shape().axis_dim(1) != right.shape().axis_dim(0))
    throw TensorOpError("Operands are incompatible.");

  auto aux = detail::TensorHelper::ope_tensors<T>(
      *left._impl->tf_tensor, *right._impl->tf_tensor,
      [](const tf::Scope &scope, tf::Input left, tf::Input right) {
        return tf::ops::MatMul(scope, left, right);
      });

  return aux;
}

template <typename T>
Matrix<T> TensorOp<T>::dot(const Matrix<T> &left, const Matrix<T> &right) {
  auto resp = Matrix<T>::to_matrix(TensorOp<T>::product_tensors(left, right));
  return resp;
}

template <typename T>
Tensor<T> TensorOp<T>::dot(const Matrix<T> &left, const Vector<T> &right) {

  if (left.dim() == 0 || right.dim() == 0)
    throw TensorOpError("One of the operands has dimension zero.");

  if (left.col_size() != right.size())
    throw TensorOpError("Operands are incompatible.");

  auto left_flat = left.data();
  auto right_flat = right.data();

  T aux{0};
  size_t step{0};
  Tensor<T> resp({left.row_size(), 1});
  auto resp_flat = resp.data();
  for (size_t i{0}; i < left.row_size(); ++i) {
    aux = 0;
    for (size_t j{0}; j < left.col_size(); ++j)
      aux += left_flat[step + j] * right_flat[j];
    resp_flat[i] = aux;
    step += left.col_size();
  }

  return resp;
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
