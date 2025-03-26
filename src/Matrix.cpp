#include "txeo/Matrix.h"
#include "txeo/Tensor.h"
#include "txeo/TensorFunc.h"
#include "txeo/TensorOp.h"
#include "txeo/detail/utils.h"

#include <algorithm>
#include <cstddef>

namespace txeo {

template <typename T>
Matrix<T>::Matrix() {
  this->create_from_shape(TensorShape({1, 1}));
  this->data()[0] = 0;
}

template <typename T>
Matrix<T>::Matrix(Tensor<T> &&tensor) : Tensor<T>(std::move(tensor)) {
  if (tensor.order() != 2)
    throw MatrixError("Tensor does not have order two.");
}

template <typename T>
void Matrix<T>::normalize_columns(NormalizationType type) {
  TensorFunc<T>::normalize_by(*this, 0, type);
};

template <typename T>
void Matrix<T>::normalize_rows(NormalizationType type) {
  TensorFunc<T>::normalize_by(*this, 1, type);
};

template <typename T>
void Matrix<T>::reshape(const TensorShape &shape) {
  if (shape.number_of_axes() != 2)
    throw MatrixError("Shape does not have two axes.");
  Tensor<T>::reshape(shape);
}

template <typename T>
Matrix<T> &Matrix<T>::transpose() {
  return TensorFunc<T>::transpose_by(*this);
}

template <typename T>
Matrix<T> Matrix<T>::dot(const Matrix<T> &matrix) const {
  return TensorOp<T>::dot(*this, matrix);
}

template <typename T>
Tensor<T> Matrix<T>::dot(const Vector<T> &vector) const {
  return TensorOp<T>::dot(*this, vector);
}

template <typename T>
Matrix<T> Matrix<T>::to_matrix(Tensor<T> &&tensor) {
  if (tensor.order() != 2)
    throw MatrixError("Tensor does not have order two.");

  Matrix<T> resp{std::move(tensor)};

  return resp;
}

template <typename T>
Matrix<T> Matrix<T>::to_matrix(const Tensor<T> &tensor) {
  if (tensor.order() != 2)
    throw MatrixError("Tensor does not have order two.");

  auto row_dim = detail::to_size_t(tensor.shape().axis_dim(0));
  auto col_dim = detail::to_size_t(tensor.shape().axis_dim(1));

  Matrix<T> resp{row_dim, col_dim};
  for (size_t i{0}; i < tensor.dim(); ++i)
    resp.data()[i] = tensor.data()[i];

  return resp;
}

template <typename T>
Tensor<T> Matrix<T>::to_tensor(Matrix<T> &&matrix) {
  Tensor<T> resp{std::move(matrix)};

  return resp;
}

template <typename T>
Tensor<T> Matrix<T>::to_tensor(const Matrix<T> &matrix) {
  Tensor<T> resp{matrix};

  return resp;
}

template <typename U>
Matrix<U> operator+(const Matrix<U> &left, const Matrix<U> &right) {
  return Matrix<U>::to_matrix(TensorOp<U>::sum(left, right));
};

template <typename U>
Matrix<U> operator+(const Matrix<U> &left, const U &right) {
  return Matrix<U>::to_matrix(TensorOp<U>::sum(left, right));
};

template <typename U>
Matrix<U> operator-(const Matrix<U> &left, const Matrix<U> &right) {
  return Matrix<U>::to_matrix(TensorOp<U>::subtract(left, right));
};

template <typename U>
Matrix<U> operator-(const U &left, const Matrix<U> &right) {
  return Matrix<U>::to_matrix(TensorOp<U>::subtract(left, right));
};

template <typename U>
Matrix<U> operator-(const Matrix<U> &left, const U &right) {
  return Matrix<U>::to_matrix(TensorOp<U>::subtract(left, right));
};

template <typename U>
Matrix<U> operator*(const Matrix<U> &matrix, const U &scalar) {
  return Matrix<U>::to_matrix(TensorOp<U>::multiply(matrix, scalar));
};

template <typename U>
Matrix<U> operator*(const U &scalar, const Matrix<U> &matrix) {
  return Matrix<U>::to_matrix(TensorOp<U>::multiply(matrix, scalar));
};

template <typename U>
Matrix<U> operator/(const Matrix<U> &matrix, const U &scalar) {
  return Matrix<U>::to_matrix(TensorOp<U>::divide(matrix, scalar));
};

template <typename U>
Matrix<U> operator/(const U &left, const Matrix<U> &right) {
  return Matrix<U>::to_matrix(TensorOp<U>::divide(left, right));
};

template class Matrix<short>;
template class Matrix<int>;
template class Matrix<bool>;
template class Matrix<long>;
template class Matrix<long long>;
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<size_t>;

template Matrix<short> operator+(const Matrix<short> &, const Matrix<short> &);
template Matrix<int> operator+(const Matrix<int> &, const Matrix<int> &);
template Matrix<bool> operator+(const Matrix<bool> &, const Matrix<bool> &);
template Matrix<long> operator+(const Matrix<long> &, const Matrix<long> &);
template Matrix<long long> operator+(const Matrix<long long> &, const Matrix<long long> &);
template Matrix<float> operator+(const Matrix<float> &, const Matrix<float> &);
template Matrix<double> operator+(const Matrix<double> &, const Matrix<double> &);
template Matrix<size_t> operator+(const Matrix<size_t> &, const Matrix<size_t> &);

template Matrix<short> operator+(const Matrix<short> &, const short &);
template Matrix<int> operator+(const Matrix<int> &, const int &);
template Matrix<bool> operator+(const Matrix<bool> &, const bool &);
template Matrix<long> operator+(const Matrix<long> &, const long &);
template Matrix<long long> operator+(const Matrix<long long> &, const long long &);
template Matrix<float> operator+(const Matrix<float> &, const float &);
template Matrix<double> operator+(const Matrix<double> &, const double &);
template Matrix<size_t> operator+(const Matrix<size_t> &, const size_t &);

template Matrix<short> operator-(const Matrix<short> &, const Matrix<short> &);
template Matrix<int> operator-(const Matrix<int> &, const Matrix<int> &);
template Matrix<bool> operator-(const Matrix<bool> &, const Matrix<bool> &);
template Matrix<long> operator-(const Matrix<long> &, const Matrix<long> &);
template Matrix<long long> operator-(const Matrix<long long> &, const Matrix<long long> &);
template Matrix<float> operator-(const Matrix<float> &, const Matrix<float> &);
template Matrix<double> operator-(const Matrix<double> &, const Matrix<double> &);
template Matrix<size_t> operator-(const Matrix<size_t> &, const Matrix<size_t> &);

template Matrix<short> operator-(const Matrix<short> &, const short &);
template Matrix<int> operator-(const Matrix<int> &, const int &);
template Matrix<bool> operator-(const Matrix<bool> &, const bool &);
template Matrix<long> operator-(const Matrix<long> &, const long &);
template Matrix<long long> operator-(const Matrix<long long> &, const long long &);
template Matrix<float> operator-(const Matrix<float> &, const float &);
template Matrix<double> operator-(const Matrix<double> &, const double &);
template Matrix<size_t> operator-(const Matrix<size_t> &, const size_t &);

template Matrix<short> operator-(const short &, const Matrix<short> &);
template Matrix<int> operator-(const int &, const Matrix<int> &);
template Matrix<bool> operator-(const bool &, const Matrix<bool> &);
template Matrix<long> operator-(const long &, const Matrix<long> &);
template Matrix<long long> operator-(const long long &, const Matrix<long long> &);
template Matrix<float> operator-(const float &, const Matrix<float> &);
template Matrix<double> operator-(const double &, const Matrix<double> &);
template Matrix<size_t> operator-(const size_t &, const Matrix<size_t> &);

template Matrix<short> operator*(const Matrix<short> &, const short &);
template Matrix<int> operator*(const Matrix<int> &, const int &);
template Matrix<bool> operator*(const Matrix<bool> &, const bool &);
template Matrix<long> operator*(const Matrix<long> &, const long &);
template Matrix<long long> operator*(const Matrix<long long> &, const long long &);
template Matrix<float> operator*(const Matrix<float> &, const float &);
template Matrix<double> operator*(const Matrix<double> &, const double &);
template Matrix<size_t> operator*(const Matrix<size_t> &, const size_t &);

template Matrix<short> operator/(const Matrix<short> &, const short &);
template Matrix<int> operator/(const Matrix<int> &, const int &);
template Matrix<bool> operator/(const Matrix<bool> &, const bool &);
template Matrix<long> operator/(const Matrix<long> &, const long &);
template Matrix<long long> operator/(const Matrix<long long> &, const long long &);
template Matrix<float> operator/(const Matrix<float> &, const float &);
template Matrix<double> operator/(const Matrix<double> &, const double &);
template Matrix<size_t> operator/(const Matrix<size_t> &, const size_t &);

template Matrix<short> operator/(const short &, const Matrix<short> &);
template Matrix<int> operator/(const int &, const Matrix<int> &);
template Matrix<bool> operator/(const bool &, const Matrix<bool> &);
template Matrix<long> operator/(const long &, const Matrix<long> &);
template Matrix<long long> operator/(const long long &, const Matrix<long long> &);
template Matrix<float> operator/(const float &, const Matrix<float> &);
template Matrix<double> operator/(const double &, const Matrix<double> &);
template Matrix<size_t> operator/(const size_t &, const Matrix<size_t> &);

template Matrix<short> operator*(const short &, const Matrix<short> &);
template Matrix<int> operator*(const int &, const Matrix<int> &);
template Matrix<bool> operator*(const bool &, const Matrix<bool> &);
template Matrix<long> operator*(const long &, const Matrix<long> &);
template Matrix<long long> operator*(const long long &, const Matrix<long long> &);
template Matrix<float> operator*(const float &, const Matrix<float> &);
template Matrix<double> operator*(const double &, const Matrix<double> &);
template Matrix<size_t> operator*(const size_t &, const Matrix<size_t> &);

} // namespace txeo