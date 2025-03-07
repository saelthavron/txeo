#include "txeo/Matrix.h"
#include "txeo/Tensor.h"

namespace txeo {

template <typename T>
Matrix<T>::Matrix(txeo::Tensor<T> &&tensor) : txeo::Tensor<T>(std::move(tensor)) {
  if (tensor.order() != 2)
    throw txeo::MatrixError("Tensor does not have order two.");
}

template class Matrix<short>;
template class Matrix<int>;
template class Matrix<bool>;
template class Matrix<long>;
template class Matrix<long long>;
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<size_t>;

} // namespace txeo