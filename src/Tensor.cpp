#include "txeo/Tensor.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tensorflow/core/framework/tensor.h>
#include <utility>
#include <vector>

#include "txeo/detail/TensorPrivate.h"
#include "txeo/detail/TensorShapePrivate.h"
#include "txeo/detail/utils.h"

namespace txeo {

namespace tf = tensorflow;

template <typename T>
template <typename P>
void Tensor<T>::create_from_shape(P &&shape) {
  auto aux = std::forward<P>(shape);
  _impl->tf_tensor =
      std::make_unique<tf::Tensor>(txeo::detail::get_tf_dtype<T>(), *aux._impl->tf_shape);
  _impl->txeo_shape._impl->ext_tf_shape = &_impl->tf_tensor->shape();
}

template <typename T>
template <typename P>
void Tensor<T>::create_from_vector(P &&shape) {
  txeo::TensorShape aux{std::forward<P>(shape)};
  _impl->tf_tensor =
      std::make_unique<tf::Tensor>(txeo::detail::get_tf_dtype<T>(), *aux._impl->tf_shape);
  _impl->txeo_shape._impl->ext_tf_shape = &_impl->tf_tensor->shape();
}

template <typename T>
inline Tensor<T>::Tensor(const Tensor &tensor) : _impl{std::make_unique<Impl>()} {
  if (this != &tensor) {
    _impl->tf_tensor = std::make_unique<tf::Tensor>(*tensor._impl->tf_tensor);
    _impl->txeo_shape._impl->ext_tf_shape = &_impl->tf_tensor->shape();
  }
}

template <typename T>
inline Tensor<T>::Tensor(Tensor &&tensor) noexcept : _impl{std::make_unique<Impl>()} {
  if (this != &tensor) {
    _impl->tf_tensor = std::make_unique<tf::Tensor>(std::move(*tensor._impl->tf_tensor));
    _impl->txeo_shape._impl->ext_tf_shape = &_impl->tf_tensor->shape();
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
inline Tensor<T>::Tensor(const std::vector<int64_t> &shape) : _impl{std::make_unique<Impl>()} {
  this->create_from_vector(shape);
}

template <typename T>
inline Tensor<T>::Tensor(std::vector<int64_t> &&shape) : _impl{std::make_unique<Impl>()} {
  this->create_from_vector(std::move(shape));
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(const Tensor &tensor) {
  *_impl->tf_tensor = *tensor._impl->tf_tensor;

  return (*this);
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(Tensor &&tensor) noexcept {
  *_impl->tf_tensor = *tensor._impl->tf_tensor;

  return (*this);
}

template <typename T>
bool Tensor<T>::operator==(const Tensor &tensor) {
  if (_impl->tf_tensor->dtype() != tensor._impl->tf_tensor->dtype())
    return false;
  if (_impl->tf_tensor->shape() != tensor._impl->tf_tensor->shape())
    return false;
  if (_impl->tf_tensor->data() != tensor._impl->tf_tensor->data())
    return false;

  return true;
}

template <typename T>
bool Tensor<T>::operator!=(const Tensor &tensor) {
  if (_impl->tf_tensor->dtype() != tensor._impl->tf_tensor->dtype())
    return true;
  if (_impl->tf_tensor->shape() != tensor._impl->tf_tensor->shape())
    return true;
  if (_impl->tf_tensor->data() != tensor._impl->tf_tensor->data())
    return true;

  return false;
}

template <typename T>
Tensor<T>::Tensor(txeo::TensorShape &&shape) : _impl{std::make_unique<Impl>()} {
  this->create_from_shape(std::move(shape));
}

template <typename T>
inline const txeo::TensorShape &Tensor<T>::shape() {
  return _impl->txeo_shape;
}

template <typename T>
inline int Tensor<T>::order() const {
  return _impl->txeo_shape.number_of_axes();
}

template <typename T>
template <typename U>
inline bool Tensor<T>::is_equal_shape(const txeo::Tensor<U> &other) const {
  return _impl->txeo_shape == other._impl->txeo_shape;
}

template <typename T>
inline int64_t Tensor<T>::dim() const {
  return _impl->txeo_shape.calculate_capacity();
}

template <typename T>
inline size_t Tensor<T>::size_in_bytes() const {
  return _impl->tf_tensor->TotalBytes();
}

template <typename T>
inline const std::type_identity_t<T> *Tensor<T>::data() const {
  return static_cast<T *>(_impl->tf_tensor->data());
}

template <typename T>
inline T &Tensor<T>::operator()() {
  const auto &aux = _impl->tf_tensor->template scalar<T>();
  return aux();
}

template <typename T>
inline T &Tensor<T>::operator()(size_t x) {
  const auto &aux = _impl->tf_tensor->template tensor<T, 1>();
  return aux(x);
}

template <typename T>
inline T &Tensor<T>::operator()(size_t x, size_t y) {
  const auto &aux = _impl->tf_tensor->template tensor<T, 2>();
  return aux(x, y);
}

template <typename T>
inline T &Tensor<T>::operator()(size_t x, size_t y, size_t z) {
  const auto &aux = _impl->tf_tensor->template tensor<T, 3>();
  return aux(x, y, z);
}

template <typename T>
inline T &Tensor<T>::operator()(size_t x, size_t y, size_t z, size_t k) {
  const auto &aux = _impl->tf_tensor->template tensor<T, 4>();
  return aux(x, y, z, k);
}

template <typename T>
inline T &Tensor<T>::operator()(size_t x, size_t y, size_t z, size_t k, size_t w) {
  const auto &aux = _impl->tf_tensor->template tensor<T, 5>();
  return aux(x, y, z, k, w);
}

template <typename T>
inline T &Tensor<T>::at() {
  if (this->order() != 0)
    throw TensorError("This tensor is not a scalar.");
  return (*this)();
}

template <typename T>
inline T &Tensor<T>::at(size_t x) {
  if (this->order() != 1)
    throw TensorError("This tensor is not a vector.");
  if (_impl->txeo_shape.axis_dim(0) >= (int64_t)x)
    throw TensorError("Axis " + std::to_string(0) + " not in the range [0," +
                      std::to_string(_impl->txeo_shape.axis_dim(0) - 1) + "]");

  return (*this)(x);
}

template <typename T>
inline T &Tensor<T>::at(size_t x, size_t y) {
  if (this->order() != 2)
    throw TensorError("This tensor is not a matrix.");
  txeo::detail::tensor::check_indexes(_impl->txeo_shape, {x, y});

  return (*this)(x, y);
}

template <typename T>
inline T &Tensor<T>::at(size_t x, size_t y, size_t z) {
  if (this->order() != 3)
    throw TensorError("This is not a tensor of order 3.");
  txeo::detail::tensor::check_indexes(_impl->txeo_shape, {x, y, z});

  return (*this)(x, y, z);
}

template <typename T>
inline T &Tensor<T>::at(size_t x, size_t y, size_t z, size_t k) {
  if (this->order() != 4)
    throw TensorError("This is not a tensor of order 4.");
  txeo::detail::tensor::check_indexes(_impl->txeo_shape, {x, y, z, k});

  return (*this)(x, y, z, k);
}

template <typename T>
inline T &Tensor<T>::at(size_t x, size_t y, size_t z, size_t k, size_t w) {
  if (this->order() != 5)
    throw TensorError("This is not a tensor of order 5.");
  txeo::detail::tensor::check_indexes(_impl->txeo_shape, {x, y, z, k, w});

  return (*this)(x, y, z, k, x);
}

template <typename T>
inline const T &Tensor<T>::operator()() const {
  const auto &aux = _impl->tf_tensor->template scalar<T>();
  return aux();
}

template <typename T>
inline const T &Tensor<T>::operator()(size_t x) const {
  const auto &aux = _impl->tf_tensor->template tensor<T, 1>();
  return aux(x);
}

template <typename T>
inline const T &Tensor<T>::operator()(size_t x, size_t y) const {
  const auto &aux = _impl->tf_tensor->template tensor<T, 2>();
  return aux(x, y);
}

template <typename T>
inline const T &Tensor<T>::operator()(size_t x, size_t y, size_t z) const {
  const auto &aux = _impl->tf_tensor->template tensor<T, 3>();
  return aux(x, y, z);
}

template <typename T>
inline const T &Tensor<T>::operator()(size_t x, size_t y, size_t z, size_t k) const {
  const auto &aux = _impl->tf_tensor->template tensor<T, 4>();
  return aux(x, y, z, k);
}

template <typename T>
inline const T &Tensor<T>::operator()(size_t x, size_t y, size_t z, size_t k, size_t w) const {
  const auto &aux = _impl->tf_tensor->template tensor<T, 5>();
  return aux(x, y, z, k, w);
}

template <typename T>
inline const T &Tensor<T>::at() const {
  if (this->order() != 0)
    throw TensorError("This tensor is not a scalar.");
  return (*this)();
}

template <typename T>
inline const T &Tensor<T>::at(size_t x) const {
  if (this->order() != 1)
    throw TensorError("This tensor is not a vector.");
  if (_impl->txeo_shape.axis_dim(0) >= (int64_t)x)
    throw TensorError("Axis " + std::to_string(0) + " not in the range [0," +
                      std::to_string(_impl->txeo_shape.axis_dim(0) - 1) + "]");

  return (*this)(x);
}

template <typename T>
inline const T &Tensor<T>::at(size_t x, size_t y) const {
  if (this->order() != 2)
    throw TensorError("This tensor is not a matrix.");
  txeo::detail::tensor::check_indexes(_impl->txeo_shape, {x, y});

  return (*this)(x, y);
}

template <typename T>
inline const T &Tensor<T>::at(size_t x, size_t y, size_t z) const {
  if (this->order() != 3)
    throw TensorError("This is not a tensor of order 3.");
  txeo::detail::tensor::check_indexes(_impl->txeo_shape, {x, y, z});

  return (*this)(x, y, z);
}

template <typename T>
inline const T &Tensor<T>::at(size_t x, size_t y, size_t z, size_t k) const {
  if (this->order() != 4)
    throw TensorError("This is not a tensor of order 4.");
  txeo::detail::tensor::check_indexes(_impl->txeo_shape, {x, y, z, k});

  return (*this)(x, y, z, k);
}

template <typename T>
inline const T &Tensor<T>::at(size_t x, size_t y, size_t z, size_t k, size_t w) const {
  if (this->order() != 5)
    throw TensorError("This is not a tensor of order 5.");
  txeo::detail::tensor::check_indexes(_impl->txeo_shape, {x, y, z, k, w});

  return (*this)(x, y, z, k, x);
}

template <typename T>
template <typename... Args>
inline T &Tensor<T>::element_at(Args... args) {
  tf::Tensor x;
  auto aux = _impl->tf_tensor->template flat<T>();
  auto flat_index =
      txeo::detail::tensor::calc_flat_index({args...}, _impl->txeo_shape._impl->tf_shape);

  return aux(flat_index);
}

template <typename T>
template <typename... Args>
inline const T &Tensor<T>::element_at(Args... args) const {
  tf::Tensor x;
  auto aux = _impl->tf_tensor->template flat<T>();
  auto flat_index =
      txeo::detail::tensor::calc_flat_index({args...}, _impl->txeo_shape._impl->tf_shape);

  return aux(flat_index);
}

// Avoiding problems in linking
template class Tensor<short>;
template class Tensor<int>;
template class Tensor<bool>;
template class Tensor<long>;
template class Tensor<long long>;
template class Tensor<float>;
template class Tensor<double>;

} // namespace txeo