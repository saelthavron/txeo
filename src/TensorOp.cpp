#include "txeo/TensorOp.h"
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

// Required for templated elements in cpp files

template class TensorOp<short>;
template class TensorOp<int>;
template class TensorOp<bool>;
template class TensorOp<long>;
template class TensorOp<long long>;
template class TensorOp<float>;
template class TensorOp<double>;

} // namespace txeo
