#include "txeo/Vector.h"

namespace txeo {

template <typename T>
Vector<T>::Vector(txeo::Tensor<T> &&tensor) : txeo::Tensor<T>(std::move(tensor)) {
  if (tensor.order() != 1)
    throw txeo::VectorError("Tensor does not have order one.");
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