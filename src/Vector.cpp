#include "txeo/Vector.h"
#include "txeo/Tensor.h"
#include "txeo/TensorFunc.h"
#include "txeo/TensorOp.h"
#include "txeo/detail/utils.h"

namespace txeo {

template <typename T>
Vector<T>::Vector() {
  this->create_from_shape(txeo::TensorShape({1}));
  this->data()[0] = 0;
}

template <typename T>
Vector<T>::Vector(txeo::Tensor<T> &&tensor) : txeo::Tensor<T>(std::move(tensor)) {
  if (tensor.order() != 1)
    throw txeo::VectorError("Tensor does not have order one.");
}

template <typename T>
void Vector<T>::reshape(const txeo::TensorShape &shape) {
  if (shape.number_of_axes() != 1)
    throw txeo::VectorError("Shape does not have one axis.");
  txeo::Tensor<T>::reshape(shape);
}

template <typename T>
Vector<T> Vector<T>::to_vector(txeo::Tensor<T> &&tensor) {
  if (tensor.order() != 1)
    throw txeo::VectorError("Tensor does not have order one.");

  Vector<T> resp{std::move(tensor)};

  return resp;
}

template <typename T>
Vector<T> Vector<T>::to_vector(const txeo::Tensor<T> &tensor) {
  if (tensor.order() != 1)
    throw txeo::VectorError("Tensor does not have order one.");

  auto dim = txeo::detail::to_size_t(tensor.shape().axis_dim(0));

  Vector<T> resp(dim);
  for (size_t i{0}; i < tensor.dim(); ++i)
    resp.data()[i] = tensor.data()[i];

  return resp;
}

template <typename T>
txeo::Tensor<T> Vector<T>::to_tensor(Vector<T> &&vector) {
  txeo::Tensor<T> resp{std::move(vector)};

  return resp;
}

template <typename T>
txeo::Tensor<T> Vector<T>::to_tensor(const Vector<T> &vector) {
  txeo::Tensor<T> resp{vector};

  return resp;
}

template <typename T>
void Vector<T>::normalize(txeo::NormalizationType type) {
  txeo::TensorFunc<T>::normalize_by(*this, type);
}

template <typename U>
Vector<U> operator+(const Vector<U> &left, const Vector<U> &right) {
  return Vector<U>::to_vector(TensorOp<U>::sum(left, right));
};

template <typename U>
Vector<U> operator+(const Vector<U> &left, const U &right) {
  return Vector<U>::to_vector(TensorOp<U>::sum(left, right));
};

template <typename U>
Vector<U> operator-(const Vector<U> &left, const Vector<U> &right) {
  return Vector<U>::to_vector(TensorOp<U>::subtract(left, right));
};

template <typename U>
Vector<U> operator-(const U &left, const Vector<U> &right) {
  return Vector<U>::to_vector(TensorOp<U>::subtract(left, right));
};

template <typename U>
Vector<U> operator-(const Vector<U> &left, const U &right) {
  return Vector<U>::to_vector(TensorOp<U>::subtract(left, right));
};

template <typename U>
Vector<U> operator*(const Vector<U> &tensor, const U &scalar) {
  return Vector<U>::to_vector(TensorOp<U>::multiply(tensor, scalar));
};

template <typename U>
Vector<U> operator/(const Vector<U> &tensor, const U &scalar) {
  return Vector<U>::to_vector(TensorOp<U>::divide(tensor, scalar));
};

template <typename U>
Vector<U> operator/(const U &left, const Vector<U> &right) {
  return Vector<U>::to_vector(TensorOp<U>::divide(left, right));
};

template class Vector<short>;
template class Vector<int>;
template class Vector<bool>;
template class Vector<long>;
template class Vector<long long>;
template class Vector<float>;
template class Vector<double>;
template class Vector<size_t>;

template Vector<short> operator+(const Vector<short> &, const Vector<short> &);
template Vector<int> operator+(const Vector<int> &, const Vector<int> &);
template Vector<bool> operator+(const Vector<bool> &, const Vector<bool> &);
template Vector<long> operator+(const Vector<long> &, const Vector<long> &);
template Vector<long long> operator+(const Vector<long long> &, const Vector<long long> &);
template Vector<float> operator+(const Vector<float> &, const Vector<float> &);
template Vector<double> operator+(const Vector<double> &, const Vector<double> &);
template Vector<size_t> operator+(const Vector<size_t> &, const Vector<size_t> &);

template Vector<short> operator+(const Vector<short> &, const short &);
template Vector<int> operator+(const Vector<int> &, const int &);
template Vector<bool> operator+(const Vector<bool> &, const bool &);
template Vector<long> operator+(const Vector<long> &, const long &);
template Vector<long long> operator+(const Vector<long long> &, const long long &);
template Vector<float> operator+(const Vector<float> &, const float &);
template Vector<double> operator+(const Vector<double> &, const double &);
template Vector<size_t> operator+(const Vector<size_t> &, const size_t &);

template Vector<short> operator-(const Vector<short> &, const Vector<short> &);
template Vector<int> operator-(const Vector<int> &, const Vector<int> &);
template Vector<bool> operator-(const Vector<bool> &, const Vector<bool> &);
template Vector<long> operator-(const Vector<long> &, const Vector<long> &);
template Vector<long long> operator-(const Vector<long long> &, const Vector<long long> &);
template Vector<float> operator-(const Vector<float> &, const Vector<float> &);
template Vector<double> operator-(const Vector<double> &, const Vector<double> &);
template Vector<size_t> operator-(const Vector<size_t> &, const Vector<size_t> &);

template Vector<short> operator-(const Vector<short> &, const short &);
template Vector<int> operator-(const Vector<int> &, const int &);
template Vector<bool> operator-(const Vector<bool> &, const bool &);
template Vector<long> operator-(const Vector<long> &, const long &);
template Vector<long long> operator-(const Vector<long long> &, const long long &);
template Vector<float> operator-(const Vector<float> &, const float &);
template Vector<double> operator-(const Vector<double> &, const double &);
template Vector<size_t> operator-(const Vector<size_t> &, const size_t &);

template Vector<short> operator-(const short &, const Vector<short> &);
template Vector<int> operator-(const int &, const Vector<int> &);
template Vector<bool> operator-(const bool &, const Vector<bool> &);
template Vector<long> operator-(const long &, const Vector<long> &);
template Vector<long long> operator-(const long long &, const Vector<long long> &);
template Vector<float> operator-(const float &, const Vector<float> &);
template Vector<double> operator-(const double &, const Vector<double> &);
template Vector<size_t> operator-(const size_t &, const Vector<size_t> &);

template Vector<short> operator*(const Vector<short> &, const short &);
template Vector<int> operator*(const Vector<int> &, const int &);
template Vector<bool> operator*(const Vector<bool> &, const bool &);
template Vector<long> operator*(const Vector<long> &, const long &);
template Vector<long long> operator*(const Vector<long long> &, const long long &);
template Vector<float> operator*(const Vector<float> &, const float &);
template Vector<double> operator*(const Vector<double> &, const double &);
template Vector<size_t> operator*(const Vector<size_t> &, const size_t &);

template Vector<short> operator/(const Vector<short> &, const short &);
template Vector<int> operator/(const Vector<int> &, const int &);
template Vector<bool> operator/(const Vector<bool> &, const bool &);
template Vector<long> operator/(const Vector<long> &, const long &);
template Vector<long long> operator/(const Vector<long long> &, const long long &);
template Vector<float> operator/(const Vector<float> &, const float &);
template Vector<double> operator/(const Vector<double> &, const double &);
template Vector<size_t> operator/(const Vector<size_t> &, const size_t &);

template Vector<short> operator/(const short &, const Vector<short> &);
template Vector<int> operator/(const int &, const Vector<int> &);
template Vector<bool> operator/(const bool &, const Vector<bool> &);
template Vector<long> operator/(const long &, const Vector<long> &);
template Vector<long long> operator/(const long long &, const Vector<long long> &);
template Vector<float> operator/(const float &, const Vector<float> &);
template Vector<double> operator/(const double &, const Vector<double> &);
template Vector<size_t> operator/(const size_t &, const Vector<size_t> &);

} // namespace txeo