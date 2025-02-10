#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <utility>
#include <vector>

#include "txeo/TensorShape.h"
#include "txeo/detail/TensorPrivate.h"
#include "txeo/detail/TensorShapePrivate.h"
#include "txeo/detail/utils.h"

namespace txeo {

namespace tf = tensorflow;

template <typename T>
inline void Tensor<T>::check_indexes(const std::vector<size_t> &indexes) {
  for (size_t i{0}; i < indexes.size(); ++i) {
    if (_impl->txeo_shape.axis_dim(i) >= txeo::detail::to_int64(indexes[i]))
      throw TensorError("Axis " + std::to_string(i) + " not in the range [0," +
                        std::to_string(_impl->txeo_shape.axis_dim(i) - 1) + "]");
  }
}

template <typename T>
template <typename P>
void Tensor<T>::create_from_shape(P &&shape) {
  auto aux = std::forward<P>(shape);
  auto shp = aux._impl->tf_shape != nullptr ? *aux._impl->tf_shape : *aux._impl->ext_tf_shape;
  _impl->tf_tensor = std::make_unique<tf::Tensor>(txeo::detail::get_tf_dtype<T>(), shp);
  _impl->txeo_shape._impl->ext_tf_shape = &_impl->tf_tensor->shape();
  _impl->txeo_shape._impl->stride =
      txeo::detail::calc_stride(*_impl->txeo_shape._impl->ext_tf_shape);
}

template <typename T>
inline Tensor<T>::Tensor() : _impl{std::make_unique<Impl>()} {
}

template <typename T>
inline Tensor<T>::Tensor(const Tensor &tensor) : _impl{std::make_unique<Impl>()} {
  create_from_shape(tensor.shape().clone());
  for (size_t i{0}; i < this->dim(); ++i)
    this->data()[i] = tensor.data()[i];
}

template <typename T>
inline Tensor<T>::Tensor(Tensor &&tensor) noexcept : _impl{std::make_unique<Impl>()} {
  if (this != &tensor) {
    _impl->tf_tensor = std::move(tensor._impl->tf_tensor);
    _impl->txeo_shape = std::move(tensor._impl->txeo_shape);
  }
}

// Defined here after "Impl" implementation in order to avoid incompleteness
template <typename T>
Tensor<T>::~Tensor() = default;

template <typename T>
Tensor<T>::Tensor(const txeo::TensorShape &shape) : _impl{std::make_unique<Impl>()} {
  this->create_from_shape(shape);
}

template <typename T>
Tensor<T>::Tensor(txeo::TensorShape &&shape) : _impl{std::make_unique<Impl>()} {
  this->create_from_shape(std::move(shape));
}

template <typename T>
inline Tensor<T>::Tensor(const txeo::TensorShape &shape, const T &fill_value)
    : _impl{std::make_unique<Impl>()} {
  this->create_from_shape(shape);
  this->fill(fill_value);
}

template <typename T>
inline Tensor<T>::Tensor(txeo::TensorShape &&shape, const T &fill_value)
    : _impl{std::make_unique<Impl>()} {
  this->create_from_shape(std::move(shape));
  this->fill(fill_value);
}

template <typename T>
Tensor<T>::Tensor(const txeo::TensorShape &shape, const std::vector<T> &values)
    : _impl{std::make_unique<Impl>()} {
  if (values.size() != shape.calculate_capacity())
    throw txeo::TensorError("Shape and number of values are incompatible!");
  create_from_shape(shape);
  std::copy(std::begin(values), std::end(values), this->data());
}

template <typename T>
inline Tensor<T>::Tensor(const std::initializer_list<std::initializer_list<T>> &values)
    : _impl{std::make_unique<Impl>()} {
  std::vector<T> flat_data;
  std::vector<size_t> shape;
  this->fill_data_shape(values, flat_data, shape);
  create_from_shape(txeo::TensorShape(shape));
  std::copy(std::begin(flat_data), std::end(flat_data), this->data());
}

template <typename T>
inline Tensor<T>::Tensor(
    const std::initializer_list<std::initializer_list<std::initializer_list<T>>> &values)
    : _impl{std::make_unique<Impl>()} {
  std::vector<T> flat_data;
  std::vector<size_t> shape;
  this->fill_data_shape(values, flat_data, shape);
  create_from_shape(txeo::TensorShape(shape));
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
  if (_impl->tf_tensor->dtype() != tensor._impl->tf_tensor->dtype())
    return false;
  if (_impl->tf_tensor->shape() != tensor._impl->tf_tensor->shape())
    return false;
  for (size_t i{0}; i < this->dim(); ++i) {
    if (this->data()[i] != tensor.data()[i])
      return false;
  }

  return true;
}

template <typename T>
bool Tensor<T>::operator!=(const Tensor &tensor) {
  if (_impl->tf_tensor->dtype() != tensor._impl->tf_tensor->dtype())
    return true;
  if (_impl->tf_tensor->shape() != tensor._impl->tf_tensor->shape())
    return true;
  for (size_t i{0}; i < this->dim(); ++i) {
    if (this->data()[i] != tensor.data()[i])
      return true;
  }

  return false;
}

template <typename T>
inline const txeo::TensorShape &Tensor<T>::shape() const {
  return _impl->txeo_shape;
}

template <typename T>
inline int Tensor<T>::order() const {
  return _impl->txeo_shape.number_of_axes();
}

template <typename T>
template <typename U>
inline bool Tensor<T>::is_equal_shape(const Tensor<U> &other) const {
  return _impl->txeo_shape == other._impl->txeo_shape;
}

template <typename T>
inline size_t Tensor<T>::dim() const {
  return _impl->txeo_shape.calculate_capacity();
}

template <typename T>
inline size_t Tensor<T>::memory_size() const {
  return _impl->tf_tensor->TotalBytes();
}

template <typename T>
inline const T *Tensor<T>::data() const {
  return static_cast<T *>(_impl->tf_tensor->data());
}

template <typename T>
inline T &Tensor<T>::operator()() {
  return this->data()[0];
}

template <typename T>
inline T &Tensor<T>::at() {
  if (this->order() != 0)
    throw TensorError("This tensor is not a scalar.");
  return (*this)();
}

template <typename T>
inline const T &Tensor<T>::operator()() const {
  return this->data()[0];
}

template <typename T>
inline const T &Tensor<T>::at() const {
  if (this->order() != 0)
    throw TensorError("This tensor is not a scalar.");
  return (*this)();
}

template <typename T>
inline void Tensor<T>::reshape(const txeo::TensorShape &shape) {
  auto old_tensor = std::move(_impl->tf_tensor);
  create_from_shape(shape);
  if (!_impl->tf_tensor->CopyFrom(*old_tensor, _impl->tf_tensor->shape()))
    throw txeo::TensorError("The number of axes do not match the dimension of this tensor!");
}

template <typename T>
inline void Tensor<T>::reshape(const std::vector<size_t> &shape) {
  reshape(txeo::TensorShape(shape));
}

template <typename T>
inline Tensor<T> Tensor<T>::slice(size_t first_axis_begin, size_t first_axis_end) const {
  if (first_axis_end < first_axis_begin)
    throw txeo::TensorError("The end index can not be less than the initial index!");
  if (txeo::detail::to_int64(first_axis_end) >= _impl->txeo_shape.axis_dim(0))
    throw txeo::TensorError(
        "The end index can not be greater than or equal to the dimension of first axis!");

  auto t_slice = _impl->tf_tensor->Slice(first_axis_begin, first_axis_end);
  Tensor<T> resp{txeo::detail::to_txeo_tensor_shape(t_slice.shape())};
  if (!resp._impl->tf_tensor->CopyFrom(t_slice, t_slice.shape()))
    throw txeo::TensorError("This tensor could not be sliced!");

  return resp;
}

template <typename T>
inline void Tensor<T>::share_from(const Tensor<T> &tensor, const txeo::TensorShape &shape) {
  if (this->dim() == 0)
    return;
  if (this->dim() != tensor.dim() || this->dim() != shape.calculate_capacity())
    throw txeo::TensorError("Parameters do not match the dimension of this tensor!");
  this->reshape(shape);

  if (!_impl->tf_tensor->CopyFrom(*_impl->tf_tensor, _impl->tf_tensor->shape()))
    throw txeo::TensorError("This tensor could not be copied!");
}

template <typename T>
inline Tensor<T> Tensor<T>::flatten() const {
  Tensor<T> resp(txeo::TensorShape({this->dim()}));
  if (this->dim() != 0)
    if (!resp._impl->tf_tensor->CopyFrom(*_impl->tf_tensor, _impl->tf_tensor->shape()))
      throw txeo::TensorError("This tensor could not be flatten!");

  return resp;
}

template <typename T>
inline void Tensor<T>::fill(const T &value) {
  for (size_t i{0}; i < this->dim(); ++i)
    this->data()[i] = value;
}

template <typename T>
inline Tensor<T> &Tensor<T>::operator=(const T &value) {
  this->fill(value);

  return (*this);
}

template <typename T>
inline T *Tensor<T>::data() {
  return static_cast<T *>(_impl->tf_tensor->data());
}

template <typename T>
inline void Tensor<T>::shuffle() {
  if (this->dim() == 0)
    return;
  std::mt19937_64 engine{std::random_device{}()};

  auto *data = this->data();
  std::shuffle(data, data + this->dim(), engine);
}

template <typename T>
inline void Tensor<T>::squeeze() {
  std::vector<size_t> new_shape;
  const auto &aux = this->shape().axes_dims();
  for (auto &item : aux)
    if (item != 1)
      new_shape.emplace_back(txeo::detail::to_size_t(item));

  this->reshape(txeo::TensorShape(new_shape));
}

template <typename T>
inline Tensor<T> Tensor<T>::clone() const {
  Tensor<T> resp{*this};
  return resp;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
  os << *tensor._impl->tf_tensor;
  return os;
}

template <typename T>
inline void Tensor<T>::fill_with_uniform_random(const T &min, const T &max, size_t seed1,
                                                size_t seed2) {
  if (this->dim() == 0)
    return;
  if (max <= min)
    throw txeo::TensorError("The max value is not greater than the min value");

  auto aux_min = static_cast<double>(min);
  auto aux_max = static_cast<double>(max);

  std::mt19937 engine{};
  std::seed_seq sseq{seed1, seed2};
  engine.seed(sseq);

  std::uniform_real_distribution<double> scaler{aux_min, aux_max};
  for (size_t i{0}; i < this->dim(); ++i)
    (*this)(i) = static_cast<T>(scaler(engine));
}

// Avoiding problems in linking

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

} // namespace txeo