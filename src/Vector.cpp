#include "txeo/Vector.h"
#include "txeo/Tensor.h"

namespace txeo {

template <typename T>
Vector<T>::Vector(txeo::Tensor<T> &&tensor) : txeo::Tensor<T>(std::move(tensor)) {
  if (tensor.order() != 1)
    throw txeo::VectorError("Tensor does not have order one.");
}

template <typename T>
void Vector<T>::reshape(const txeo::TensorShape &shape) {
  if (shape.number_of_axes() != 2)
    throw txeo::VectorError("Tensor does not have order one.");
  txeo::Tensor<T>::reshape(shape);
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