#include "txeo/TensorShape.h"
#include "txeo/detail/TensorPrivate.h"
#include "txeo/detail/TensorShapePrivate.h"
#include "txeo/detail/utils.h"

#include <memory>
#include <tensorflow/core/framework/tensor.h>
#include <utility>

namespace txeo {

namespace tf = tensorflow;

template <typename T>
template <typename P>
void Tensor<T>::create_from_shape(P &&shape) {
  auto aux = std::forward<P>(shape);
  _impl->tf_tensor =
      std::make_unique<tf::Tensor>(txeo::detail::get_tf_dtype<T>(), *aux._impl->tf_shape);
}

template <typename T>
template <typename P>
void Tensor<T>::create_from_vector(P &&shape) {
  txeo::TensorShape aux{std::forward<P>(shape)};
  _impl->tf_tensor =
      std::make_unique<tf::Tensor>(txeo::detail::get_tf_dtype<T>(), *aux._impl->tf_shape);
}

template <typename T>
Tensor<T>::Tensor(const txeo::TensorShape &shape) : _impl{std::make_unique<Impl>()} {
  this->create_from_shape(shape);
}

template <typename T>
Tensor<T>::Tensor(txeo::TensorShape &&shape) : _impl{std::make_unique<Impl>()} {
  this->create_from_shape(shape);
}

// Defined here after Impl implementation in order to avoid incompleteness
template <typename T>
Tensor<T>::~Tensor() = default;

template <typename T>
inline txeo::TensorShape Tensor<T>::shape() const {
  txeo::TensorShape res({});
  for (auto &item : _impl->tf_tensor->shape().dim_sizes())
    res.push_axis_back(item);

  return res;
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