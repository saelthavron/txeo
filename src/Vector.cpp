#include "txeo/Vector.h"
#include "txeo/Tensor.h"
#include "txeo/detail/utils.h"

namespace txeo {

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
inline Vector<T> Vector<T>::to_vector(txeo::Tensor<T> &&tensor) {
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

template class Vector<short>;
template class Vector<int>;
template class Vector<bool>;
template class Vector<long>;
template class Vector<long long>;
template class Vector<float>;
template class Vector<double>;
template class Vector<size_t>;

} // namespace txeo