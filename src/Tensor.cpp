#include "txeo/TensorShape.h"
#include "txeo/detail/TensorPrivate.h"
#include "txeo/detail/TensorShapePrivate.h"
#include "txeo/detail/utils.h"

#include <cstdint>
#include <memory>
#include <tensorflow/core/framework/tensor.h>
#include <utility>
#include <vector>

namespace txeo {

namespace tf = tensorflow;

template <typename T>
template <typename P>
void Tensor<T>::create_from_shape(P &&shape) {
  auto aux = std::forward<P>(shape);
  _impl->tf_tensor =
      std::make_unique<tf::Tensor>(txeo::detail::get_tf_dtype<T>(), *aux._impl->tf_shape);
  _impl->txeo_shape = std::make_unique<txeo::TensorShape>(std::vector<int64_t>{});

  txeo::detail::tensor::update_shape(_impl->tf_tensor, _impl->txeo_shape);
}

template <typename T>
template <typename P>
void Tensor<T>::create_from_vector(P &&shape) {
  txeo::TensorShape aux{std::forward<P>(shape)};
  _impl->tf_tensor =
      std::make_unique<tf::Tensor>(txeo::detail::get_tf_dtype<T>(), *aux._impl->tf_shape);
  _impl->txeo_shape = std::make_unique<txeo::TensorShape>(std::vector<int64_t>{});
  txeo::detail::tensor::update_shape(_impl->tf_tensor, _impl->txeo_shape);
}

template <typename T>
inline Tensor<T>::Tensor(const Tensor &tensor) : _impl{std::make_unique<Impl>()} {
  if (this != &tensor) {
    _impl->tf_tensor = std::make_unique<tf::Tensor>(*tensor._impl->tf_tensor);
    _impl->txeo_shape = std::make_unique<txeo::TensorShape>(std::vector<int64_t>{});
    txeo::detail::tensor::update_shape(_impl->tf_tensor, _impl->txeo_shape);
  }
}

template <typename T>
inline Tensor<T>::Tensor(Tensor &&tensor) noexcept : _impl{std::make_unique<Impl>()} {
  if (this != &tensor) {
    _impl->tf_tensor = std::make_unique<tf::Tensor>(std::move(*tensor._impl->tf_tensor));
    _impl->txeo_shape = std::make_unique<txeo::TensorShape>(std::vector<int64_t>{});
    txeo::detail::tensor::update_shape(_impl->tf_tensor, _impl->txeo_shape);
  }
}

// Defined here after Impl implementation in order to avoid incompleteness
template <typename T>
Tensor<T>::~Tensor() = default;

template <typename T>
Tensor<T>::Tensor(const txeo::TensorShape &shape) : _impl{std::make_unique<Impl>()} {
  this->create_from_shape(shape);
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
  return *_impl->txeo_shape;
}

template <typename T>
inline int Tensor<T>::order() const {
  return _impl->tf_tensor->shape().dims();
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