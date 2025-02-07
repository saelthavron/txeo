#include <cstddef>
#include <cstdint>
#include <memory>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/tensor_shape.h>
#include <type_traits>
#include <utility>
#include <vector>

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
inline Tensor<T>::Tensor() : _impl{std::make_unique<Impl>()} {
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
  _impl->txeo_shape = tensor._impl->txeo_shape;

  return (*this);
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(Tensor &&tensor) noexcept {
  *_impl->tf_tensor = std::move(*tensor._impl->tf_tensor);
  _impl->txeo_shape = std::move(tensor._impl->txeo_shape);

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
inline int64_t Tensor<T>::dim() const {
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
inline void Tensor<T>::reshape(const std::vector<int64_t> &shape) {
  auto &old_tensor = _impl->tf_tensor;
  create_from_vector(shape);
  if (!_impl->tf_tensor->CopyFrom(*old_tensor, _impl->tf_tensor->shape()))
    throw txeo::TensorError("The number of axes do not match the dimension of this tensor!");
}

template <typename T>
inline void Tensor<T>::reshape(const txeo::TensorShape &shape) {
  reshape(shape.axes_dims());
}

template <typename T>
inline Tensor<T> Tensor<T>::slice(size_t first_axis_begin, size_t first_axis_end) const {
  if (first_axis_end < first_axis_begin)
    throw txeo::TensorError("The end index can not be less than the initial index!");
  if (txeo::detail::to_int64(first_axis_end) >= _impl->txeo_shape.axis_dim(0))
    throw txeo::TensorError(
        "The end index can not be greater than or equal to the dimension of first axis!");

  auto t_slice = _impl->tf_tensor->Slice(first_axis_begin, first_axis_end);
  Tensor resp;
  resp._impl->tf_tensor = std::make_unique<tf::Tensor>(t_slice.dtype(), t_slice.shape());
  resp._impl->txeo_shape._impl->ext_tf_shape = &resp._impl->tf_tensor->shape();
  if (!resp._impl->tf_tensor->CopyFrom(t_slice, t_slice.shape()))
    throw txeo::TensorError("This tensor could not be sliced!");

  return resp;
}

template <typename T>
inline void Tensor<T>::copy_from(const Tensor<T> &tensor, const txeo::TensorShape &shape) {
  if (this->dim() != tensor.dim() || this->dim() != shape.calculate_capacity())
    throw txeo::TensorError("Parameters do not match the dimension of this tensor!");
  this->reshape(shape);
  if (!_impl->tf_tensor->CopyFrom(*_impl->tf_tensor, _impl->tf_tensor->shape()))
    throw txeo::TensorError("This tensor could not be copied!");
}

template <typename T>
inline Tensor<T> Tensor<T>::flatten() const {
  Tensor resp;
  resp._impl->tf_tensor =
      std::make_unique<tf::Tensor>(_impl->tf_tensor->dtype(), tf::TensorShape{this->dim()});
  resp._impl->txeo_shape._impl->ext_tf_shape = &resp._impl->tf_tensor->shape();
  if (!resp._impl->tf_tensor->CopyFrom(*_impl->tf_tensor, _impl->tf_tensor->shape()))
    throw txeo::TensorError("This tensor could not be flatten!");

  return resp;
}

template <typename T>
inline void Tensor<T>::fill(const T &value) {
  _impl->tf_tensor->template flat<T>().setConstant(value);
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
template <c_numeric N>
inline void Tensor<T>::fill_with_uniform_random(const N &min, const N &max, size_t seed1,
                                                size_t seed2) {
  if (max <= min)
    throw txeo::TensorError("The max value is not greater than the min value");

  std::mt19937 engine{};
  std::seed_seq sseq{seed1, seed2};
  engine.seed(sseq);
  if (std::is_floating_point_v<N>) {
    std::uniform_real_distribution<N> scaler{min, max};
    for (int64_t i{0}; i < this->dim(); ++i)
      (*this)(i) = scaler(engine);
  } else {
    std::uniform_int_distribution<N> scaler{min, max};
    for (int64_t i{0}; i < this->dim(); ++i)
      (*this)(i) = scaler(engine);
  }
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