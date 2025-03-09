#include "txeo/Matrix.h"
#include "txeo/Tensor.h"
#include "txeo/detail/utils.h"
#include <cstddef>

namespace txeo {

template <typename T>
Matrix<T>::Matrix(txeo::Tensor<T> &&tensor) : txeo::Tensor<T>(std::move(tensor)) {
  if (tensor.order() != 2)
    throw txeo::MatrixError("Tensor does not have order two.");
}

template <typename T>
void Matrix<T>::reshape(const txeo::TensorShape &shape) {
  if (shape.number_of_axes() != 2)
    throw txeo::MatrixError("Shape does not have two axes.");
  txeo::Tensor<T>::reshape(shape);
}

template <typename T>
Matrix<T> Matrix<T>::to_matrix(txeo::Tensor<T> &&tensor) {
  if (tensor.order() != 2)
    throw txeo::MatrixError("Tensor does not have order two.");

  Matrix<T> resp{std::move(tensor)};

  return resp;
}

template <typename T>
Matrix<T> Matrix<T>::to_matrix(const txeo::Tensor<T> &tensor) {
  if (tensor.order() != 2)
    throw txeo::MatrixError("Tensor does not have order two.");

  auto row_dim = txeo::detail::to_size_t(tensor.shape().axis_dim(0));
  auto col_dim = txeo::detail::to_size_t(tensor.shape().axis_dim(1));

  Matrix<T> resp{row_dim, col_dim};
  for (size_t i{0}; i < tensor.dim(); ++i)
    resp.data()[i] = tensor.data()[i];

  return resp;
}

template <typename T>
txeo::Tensor<T> Matrix<T>::to_tensor(Matrix<T> &&matrix) {
  txeo::Tensor<T> resp{std::move(matrix)};

  return resp;
}

template <typename T>
txeo::Tensor<T> Matrix<T>::to_tensor(const Matrix<T> &matrix) {
  txeo::Tensor<T> resp{matrix};

  return resp;
}

template class Matrix<short>;
template class Matrix<int>;
template class Matrix<bool>;
template class Matrix<long>;
template class Matrix<long long>;
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<size_t>;

} // namespace txeo