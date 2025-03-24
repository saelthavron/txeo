
#include "txeo/Tensor.h"
#include "txeo/TensorFunc.h"
#include "txeo/TensorOp.h"
#include "txeo/TensorPart.h"
#include "txeo/TensorShape.h"
#include "txeo/detail/TensorPrivate.h"
#include "txeo/detail/TensorShapePrivate.h"
#include "txeo/detail/utils.h"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

namespace txeo {

namespace tf = tensorflow;

template <typename T>
void Tensor<T>::check_indexes(const std::vector<size_t> &indexes) {
  for (size_t i{0}; i < indexes.size(); ++i) {
    auto aux = detail::to_int64(indexes[i]);
    if (aux < 0 || aux >= _impl->txeo_shape.axis_dim(i))
      throw TensorError("Index out of bounds!");
  }
}

template <typename T>
template <typename P>
void Tensor<T>::create_from_shape(P &&shape) {
  auto aux = std::forward<P>(shape);
  auto shp = aux._impl->tf_shape != nullptr ? *aux._impl->tf_shape : *aux._impl->ext_tf_shape;
  _impl->tf_tensor = std::make_unique<tf::Tensor>(detail::get_tf_dtype<T>(), shp);
  _impl->txeo_shape._impl->tf_shape = nullptr;
  _impl->txeo_shape._impl->ext_tf_shape = &_impl->tf_tensor->shape();
  _impl->txeo_shape._impl->stride = detail::calc_stride(*_impl->txeo_shape._impl->ext_tf_shape);
}

template <typename T>
Tensor<T>::Tensor() : _impl{std::make_unique<Impl>()} {
  this->create_from_shape(TensorShape({}));
  this->data()[0] = 0;
}

template <typename T>
Tensor<T>::Tensor(const Tensor &tensor) : _impl{std::make_unique<Impl>()} {
  create_from_shape(tensor.shape().clone());
  for (size_t i{0}; i < this->dim(); ++i)
    this->data()[i] = tensor.data()[i];
}

template <typename T>
Tensor<T>::Tensor(Tensor &&tensor) noexcept : _impl{std::make_unique<Impl>()} {
  if (this != &tensor) {
    _impl->tf_tensor = std::move(tensor._impl->tf_tensor);
    _impl->txeo_shape = std::move(tensor._impl->txeo_shape);
  }
}

// Defined here after "Impl" implementation in order to avoid incompleteness
template <typename T>
Tensor<T>::~Tensor() = default;

template <typename T>
Tensor<T>::Tensor(const TensorShape &shape) : _impl{std::make_unique<Impl>()} {
  this->create_from_shape(shape);
}

template <typename T>
Tensor<T>::Tensor(TensorShape &&shape) : _impl{std::make_unique<Impl>()} {
  this->create_from_shape(std::move(shape));
}

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t> &shape) : _impl{std::make_unique<Impl>()} {
  this->create_from_shape(TensorShape(shape));
}

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> &&shape) : _impl{std::make_unique<Impl>()} {
  this->create_from_shape(TensorShape(std::move(shape)));
}

template <typename T>
Tensor<T>::Tensor(const TensorShape &shape, const T &fill_value) : _impl{std::make_unique<Impl>()} {
  this->create_from_shape(shape);
  this->fill(fill_value);
}

template <typename T>
Tensor<T>::Tensor(TensorShape &&shape, const T &fill_value) : _impl{std::make_unique<Impl>()} {
  this->create_from_shape(std::move(shape));
  this->fill(fill_value);
}

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t> &shape, const T &fill_value)
    : _impl{std::make_unique<Impl>()} {
  this->create_from_shape(TensorShape(shape));
  this->fill(fill_value);
}

template <typename T>
Tensor<T>::Tensor(std::vector<size_t> &&shape, const T &fill_value)
    : _impl{std::make_unique<Impl>()} {
  this->create_from_shape(TensorShape(std::move(shape)));
  this->fill(fill_value);
}

template <typename T>
Tensor<T>::Tensor(const TensorShape &shape, const std::vector<T> &values)
    : _impl{std::make_unique<Impl>()} {
  if (values.size() != shape.calculate_capacity())
    throw TensorError("Shape and number of values are incompatible!");
  create_from_shape(shape);
  std::copy(std::begin(values), std::end(values), this->data());
}

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t> &shape, const std::vector<T> &values)
    : _impl{std::make_unique<Impl>()} {
  TensorShape aux(shape);
  if (values.size() != aux.calculate_capacity())
    throw TensorError("Shape and number of values are incompatible!");
  create_from_shape(aux);
  std::copy(std::begin(values), std::end(values), this->data());
}

template <typename T>
Tensor<T>::Tensor(const std::initializer_list<std::initializer_list<T>> &values)
    : _impl{std::make_unique<Impl>()} {
  std::vector<T> flat_data;
  std::vector<size_t> shape;
  this->fill_data_shape(values, flat_data, shape);
  create_from_shape(TensorShape(shape));
  std::copy(std::begin(flat_data), std::end(flat_data), this->data());
}

template <typename T>
Tensor<T>::Tensor(
    const std::initializer_list<std::initializer_list<std::initializer_list<T>>> &values)
    : _impl{std::make_unique<Impl>()} {
  std::vector<T> flat_data;
  std::vector<size_t> shape;
  this->fill_data_shape(values, flat_data, shape);
  create_from_shape(TensorShape(shape));
  std::copy(std::begin(flat_data), std::end(flat_data), this->data());
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(const Tensor &tensor) {
  this->create_from_shape(tensor.shape().clone());
  for (size_t i{0}; i < this->dim(); ++i)
    this->data()[i] = tensor.data()[i];

  return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(Tensor &&tensor) noexcept {
  if (this != &tensor) {
    _impl->tf_tensor = std::move(tensor._impl->tf_tensor);
    _impl->txeo_shape = std::move(tensor._impl->txeo_shape);
  }
  return (*this);
}

template <typename T>
bool Tensor<T>::operator==(const Tensor &tensor) {
  if (_impl->tf_tensor->shape() != tensor._impl->tf_tensor->shape())
    return false;
  for (size_t i{0}; i < this->dim(); ++i) {
    if (!detail::is_zero(this->data()[i] - tensor.data()[i]))
      return false;
  }

  return true;
}

template <typename T>
bool Tensor<T>::operator!=(const Tensor &tensor) {
  if (_impl->tf_tensor->shape() != tensor._impl->tf_tensor->shape())
    return true;
  for (size_t i{0}; i < this->dim(); ++i) {
    if (this->data()[i] != tensor.data()[i])
      return true;
  }

  return false;
}

template <typename T>
const TensorShape &Tensor<T>::shape() const {
  return _impl->txeo_shape;
}

template <typename T>
int Tensor<T>::order() const {
  return _impl->txeo_shape.number_of_axes();
}

template <typename T>
size_t Tensor<T>::dim() const {
  return _impl->txeo_shape.calculate_capacity();
}

template <typename T>
size_t Tensor<T>::memory_size() const {
  return _impl->tf_tensor->TotalBytes();
}

template <typename T>
const T *Tensor<T>::data() const {
  return static_cast<T *>(_impl->tf_tensor->data());
}

template <typename T>
T &Tensor<T>::operator()() {
  return this->data()[0];
}

template <typename T>
T &Tensor<T>::at() {
  if (this->order() != 0)
    throw TensorError("This tensor is not a scalar.");
  return (*this)();
}

template <typename T>
const T &Tensor<T>::operator()() const {
  return this->data()[0];
}

template <typename T>
const T &Tensor<T>::at() const {
  if (this->order() != 0)
    throw TensorError("This tensor is not a scalar.");
  return (*this)();
}

template <typename T>
void Tensor<T>::reshape(const TensorShape &shape) {
  auto old_tensor = std::move(_impl->tf_tensor);
  create_from_shape(shape);
  if (!_impl->tf_tensor->CopyFrom(*old_tensor, _impl->tf_tensor->shape()))
    throw TensorError("The number of axes do not match the dimension of this tensor!");
}

template <typename T>
void Tensor<T>::reshape(const std::vector<size_t> &shape) {
  reshape(TensorShape(shape));
}

template <typename T>
Tensor<T> Tensor<T>::slice(size_t first_axis_begin, size_t first_axis_end) const {
  try {
    return TensorPart<T>::slice(*this, first_axis_begin, first_axis_end);
  } catch (TensorPartError e) {
    throw TensorError(e.what());
  }
}

template <typename T>
void Tensor<T>::view_of(const Tensor<T> &tensor, const TensorShape &shape) {
  if (this->dim() == 0)
    return;
  if (this->dim() != tensor.dim() || this->dim() != shape.calculate_capacity())
    throw TensorError("Parameters do not match the dimension of this tensor!");
  this->reshape(shape);
  if (!_impl->tf_tensor->CopyFrom(*tensor._impl->tf_tensor, *shape._impl->tf_shape))
    throw TensorError("This tensor could not be shared!");
}

template <typename T>
Tensor<T> Tensor<T>::flatten() const {
  Tensor<T> resp(TensorShape({this->dim()}));
  if (this->dim() != 0)
    if (!resp._impl->tf_tensor->CopyFrom(*_impl->tf_tensor, resp._impl->tf_tensor->shape()))
      throw TensorError("This tensor could not be flatten!");

  return resp;
}

template <typename T>
void Tensor<T>::fill(const T &value) {
  for (size_t i{0}; i < this->dim(); ++i)
    this->data()[i] = value;
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(const T &value) {
  this->fill(value);

  return (*this);
}

template <typename T>
T *Tensor<T>::data() {
  return static_cast<T *>(_impl->tf_tensor->data());
}

template <typename T>
void Tensor<T>::shuffle() {
  if (this->dim() == 0)
    return;
  std::mt19937_64 engine{std::random_device{}()};

  auto *data = this->data();
  std::shuffle(data, data + this->dim(), engine);
}

template <typename T>
void Tensor<T>::squeeze() {
  std::vector<size_t> new_shape;
  const auto &aux = this->shape().axes_dims();
  for (auto &item : aux)
    if (item != 1)
      new_shape.emplace_back(detail::to_size_t(item));

  this->reshape(TensorShape(new_shape));
}

template <typename T>
Tensor<T> &Tensor<T>::increase_dimension(size_t axis, T value) {
  return TensorPart<T>::increase_dimension_by(*this, axis, value);
}

template <typename T>
txeo::Tensor<T> &Tensor<T>::power(const T &exponent) {
  return TensorFunc<T>::power_elem_by(*this, exponent);
}

template <typename T>
txeo::Tensor<T> &Tensor<T>::square() {
  return TensorFunc<T>::square_by(*this);
}

template <typename T>
txeo::Tensor<T> &Tensor<T>::sqrt() {
  return TensorFunc<T>::sqrt_by(*this);
}

template <typename T>
txeo::Tensor<T> &Tensor<T>::abs() {
  return TensorFunc<T>::abs_by(*this);
}

template <typename T>
txeo::Tensor<T> &Tensor<T>::permute(const std::vector<size_t> &axes) {
  return TensorFunc<T>::permute_by(*this, axes);
}

template <typename T>
txeo::Tensor<T> &Tensor<T>::normalize(size_t axis, txeo::NormalizationType type) {
  return TensorFunc<T>::normalize_by(*this, axis, type);
}

template <typename T>
txeo::Tensor<T> &Tensor<T>::normalize(txeo::NormalizationType type) {
  return TensorFunc<T>::normalize_by(*this, type);
}

template <typename T>
T Tensor<T>::dot(const Tensor<T> &tensor) const {
  return TensorOp<T>::dot(*this, tensor);
}

template <typename T>
Tensor<T> Tensor<T>::clone() const {
  Tensor<T> resp{*this};
  return resp;
}

template <typename U>
std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor) {
  os << *tensor._impl->tf_tensor;
  return os;
}

template <typename T>
void Tensor<T>::fill_with_uniform_random(const T &min, const T &max, const size_t &seed1,
                                         const size_t &seed2) {
  if (this->dim() == 0)
    return;
  if (max <= min)
    throw TensorError("The max value is not greater than the min value");

  auto aux_min = static_cast<double>(min);
  auto aux_max = static_cast<double>(max);

  std::mt19937 engine{};
  std::seed_seq sseq{seed1, seed2};
  engine.seed(sseq);

  std::uniform_real_distribution<double> scaler{aux_min, aux_max};
  for (size_t i{0}; i < this->dim(); ++i)
    this->data()[i] = static_cast<T>(scaler(engine));
}

template <typename T>
void Tensor<T>::fill_with_uniform_random(const T &min, const T &max) {
  if (this->dim() == 0)
    return;
  if (max <= min)
    throw TensorError("The max value is not greater than the min value");

  auto aux_min = static_cast<double>(min);
  auto aux_max = static_cast<double>(max);

  std::mt19937 engine{std::random_device{}()};

  std::uniform_real_distribution<double> scaler{aux_min, aux_max};
  for (size_t i{0}; i < this->dim(); ++i)
    this->data()[i] = static_cast<T>(scaler(engine));
}

template <typename U>
Tensor<U> operator+(const Tensor<U> &left, const Tensor<U> &right) {
  return TensorOp<U>::sum(left, right);
};

template <typename U>
Tensor<U> operator+(const Tensor<U> &left, const U &right) {
  return TensorOp<U>::sum(left, right);
};

template <typename U>
Tensor<U> operator-(const Tensor<U> &left, const Tensor<U> &right) {
  return TensorOp<U>::subtract(left, right);
};

template <typename U>
Tensor<U> operator-(const U &left, const Tensor<U> &right) {
  return TensorOp<U>::subtract(left, right);
};

template <typename U>
Tensor<U> operator-(const Tensor<U> &left, const U &right) {
  return TensorOp<U>::subtract(left, right);
};

template <typename U>
Tensor<U> operator*(const Tensor<U> &tensor, const U &scalar) {
  return TensorOp<U>::multiply(tensor, scalar);
};

template <typename U>
Tensor<U> operator*(const U &scalar, const Tensor<U> &tensor) {
  return TensorOp<U>::multiply(tensor, scalar);
};

template <typename U>
Tensor<U> operator/(const Tensor<U> &tensor, const U &scalar) {
  return TensorOp<U>::divide(tensor, scalar);
};

template <typename U>
Tensor<U> operator/(const U &left, const Tensor<U> &right) {
  return TensorOp<U>::divide(left, right);
};

template <typename T>
Tensor<T> &Tensor<T>::operator+=(const Tensor<T> &tensor) {
  TensorOp<T>::sum_by(*this, tensor);
  return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::operator-=(const Tensor<T> &tensor) {
  TensorOp<T>::subtract_by(*this, tensor);
  return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::operator*=(const T &scalar) {
  TensorOp<T>::multiply_by(*this, scalar);
  return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::operator/=(const T &scalar) {
  TensorOp<T>::divide_by(*this, scalar);
  return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::operator+=(const T &scalar) {
  TensorOp<T>::sum_by(*this, scalar);
  return *this;
}

template <typename T>
Tensor<T> &Tensor<T>::operator-=(const T &scalar) {
  TensorOp<T>::subtract_by(*this, scalar);
  return *this;
}

template <typename T>
TensorIterator<T> Tensor<T>::begin() {
  return TensorIterator<T>{this->data()};
}

template <typename T>
TensorIterator<T> Tensor<T>::end() {
  return TensorIterator<T>{this->data() + this->dim()};
}

template <typename T>
TensorIterator<const T> Tensor<T>::begin() const {
  return TensorIterator<const T>{this->data()};
}

template <typename T>
TensorIterator<const T> Tensor<T>::end() const {
  return TensorIterator<const T>{this->data() + this->dim()};
}

// Avoiding problems in linking

template class Tensor<size_t>;
template class Tensor<short>;
template class Tensor<int>;
template class Tensor<bool>;
template class Tensor<long>;
template class Tensor<long long>;
template class Tensor<float>;
template class Tensor<double>;

template std::ostream &operator<<(std::ostream &, const Tensor<short> &);
template std::ostream &operator<<(std::ostream &, const Tensor<int> &);
template std::ostream &operator<<(std::ostream &, const Tensor<bool> &);
template std::ostream &operator<<(std::ostream &, const Tensor<long> &);
template std::ostream &operator<<(std::ostream &, const Tensor<long long> &);
template std::ostream &operator<<(std::ostream &, const Tensor<float> &);
template std::ostream &operator<<(std::ostream &, const Tensor<double> &);
template std::ostream &operator<<(std::ostream &, const Tensor<size_t> &);

template Tensor<short> operator+(const Tensor<short> &, const Tensor<short> &);
template Tensor<int> operator+(const Tensor<int> &, const Tensor<int> &);
template Tensor<bool> operator+(const Tensor<bool> &, const Tensor<bool> &);
template Tensor<long> operator+(const Tensor<long> &, const Tensor<long> &);
template Tensor<long long> operator+(const Tensor<long long> &, const Tensor<long long> &);
template Tensor<float> operator+(const Tensor<float> &, const Tensor<float> &);
template Tensor<double> operator+(const Tensor<double> &, const Tensor<double> &);
template Tensor<size_t> operator+(const Tensor<size_t> &, const Tensor<size_t> &);

template Tensor<short> operator+(const Tensor<short> &, const short &);
template Tensor<int> operator+(const Tensor<int> &, const int &);
template Tensor<bool> operator+(const Tensor<bool> &, const bool &);
template Tensor<long> operator+(const Tensor<long> &, const long &);
template Tensor<long long> operator+(const Tensor<long long> &, const long long &);
template Tensor<float> operator+(const Tensor<float> &, const float &);
template Tensor<double> operator+(const Tensor<double> &, const double &);
template Tensor<size_t> operator+(const Tensor<size_t> &, const size_t &);

template Tensor<short> operator-(const Tensor<short> &, const Tensor<short> &);
template Tensor<int> operator-(const Tensor<int> &, const Tensor<int> &);
template Tensor<bool> operator-(const Tensor<bool> &, const Tensor<bool> &);
template Tensor<long> operator-(const Tensor<long> &, const Tensor<long> &);
template Tensor<long long> operator-(const Tensor<long long> &, const Tensor<long long> &);
template Tensor<float> operator-(const Tensor<float> &, const Tensor<float> &);
template Tensor<double> operator-(const Tensor<double> &, const Tensor<double> &);
template Tensor<size_t> operator-(const Tensor<size_t> &, const Tensor<size_t> &);

template Tensor<short> operator-(const Tensor<short> &, const short &);
template Tensor<int> operator-(const Tensor<int> &, const int &);
template Tensor<bool> operator-(const Tensor<bool> &, const bool &);
template Tensor<long> operator-(const Tensor<long> &, const long &);
template Tensor<long long> operator-(const Tensor<long long> &, const long long &);
template Tensor<float> operator-(const Tensor<float> &, const float &);
template Tensor<double> operator-(const Tensor<double> &, const double &);
template Tensor<size_t> operator-(const Tensor<size_t> &, const size_t &);

template Tensor<short> operator-(const short &, const Tensor<short> &);
template Tensor<int> operator-(const int &, const Tensor<int> &);
template Tensor<bool> operator-(const bool &, const Tensor<bool> &);
template Tensor<long> operator-(const long &, const Tensor<long> &);
template Tensor<long long> operator-(const long long &, const Tensor<long long> &);
template Tensor<float> operator-(const float &, const Tensor<float> &);
template Tensor<double> operator-(const double &, const Tensor<double> &);
template Tensor<size_t> operator-(const size_t &, const Tensor<size_t> &);

template Tensor<short> operator*(const Tensor<short> &, const short &);
template Tensor<int> operator*(const Tensor<int> &, const int &);
template Tensor<bool> operator*(const Tensor<bool> &, const bool &);
template Tensor<long> operator*(const Tensor<long> &, const long &);
template Tensor<long long> operator*(const Tensor<long long> &, const long long &);
template Tensor<float> operator*(const Tensor<float> &, const float &);
template Tensor<double> operator*(const Tensor<double> &, const double &);
template Tensor<size_t> operator*(const Tensor<size_t> &, const size_t &);

template Tensor<short> operator/(const Tensor<short> &, const short &);
template Tensor<int> operator/(const Tensor<int> &, const int &);
template Tensor<bool> operator/(const Tensor<bool> &, const bool &);
template Tensor<long> operator/(const Tensor<long> &, const long &);
template Tensor<long long> operator/(const Tensor<long long> &, const long long &);
template Tensor<float> operator/(const Tensor<float> &, const float &);
template Tensor<double> operator/(const Tensor<double> &, const double &);
template Tensor<size_t> operator/(const Tensor<size_t> &, const size_t &);

template Tensor<short> operator/(const short &, const Tensor<short> &);
template Tensor<int> operator/(const int &, const Tensor<int> &);
template Tensor<bool> operator/(const bool &, const Tensor<bool> &);
template Tensor<long> operator/(const long &, const Tensor<long> &);
template Tensor<long long> operator/(const long long &, const Tensor<long long> &);
template Tensor<float> operator/(const float &, const Tensor<float> &);
template Tensor<double> operator/(const double &, const Tensor<double> &);
template Tensor<size_t> operator/(const size_t &, const Tensor<size_t> &);

template Tensor<short> operator*(const short &, const Tensor<short> &);
template Tensor<int> operator*(const int &, const Tensor<int> &);
template Tensor<bool> operator*(const bool &, const Tensor<bool> &);
template Tensor<long> operator*(const long &, const Tensor<long> &);
template Tensor<long long> operator*(const long long &, const Tensor<long long> &);
template Tensor<float> operator*(const float &, const Tensor<float> &);
template Tensor<double> operator*(const double &, const Tensor<double> &);
template Tensor<size_t> operator*(const size_t &, const Tensor<size_t> &);

} // namespace txeo
