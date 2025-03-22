#include "txeo/OlsGDTrainer.h"
#include "txeo/Matrix.h"
#include "txeo/TensorAgg.h"
#include "txeo/TensorFunc.h"
#include "txeo/TensorOp.h"
#include "txeo/TensorPart.h"

namespace txeo {

template <typename T>
  requires(std::floating_point<T>)
void OlsGDTrainer<T>::set_learning_rate(T learning_rate) {
  this->reset();
  _learning_rate = learning_rate;
}

template <typename T>
  requires(std::floating_point<T>)
T OlsGDTrainer<T>::learning_rate() const {
  return _learning_rate;
}

template <typename T>
  requires(std::floating_point<T>)
txeo::Tensor<T> OlsGDTrainer<T>::predict(const txeo::Tensor<T> input) {
  return input;
}

template <typename T>
  requires(std::floating_point<T>)
void OlsGDTrainer<T>::train(const txeo::Tensor<T> &X, const txeo::Tensor<T> &y) {
  if (X.order() != 2 || y.order() != 2)
    throw txeo::OlsGDTrainerError("One of the tensors is not second-order.");
  if (X.dim() == 0 || y.dim() == 0)
    throw txeo::OlsGDTrainerError("One of the tensors has dimension zero.");
  if (X.shape().axis_dim(0) != y.shape().axis_dim(0))
    throw txeo::OlsGDTrainerError("Tensors are incompatible.");

  size_t p = X.shape().axis_dim(0);
  size_t n = X.shape().axis_dim(1);
  size_t m = y.shape().axis_dim(1);

  auto train_in = txeo::Matrix<T>::to_matrix(txeo::TensorPart<T>::increment_dimension(X, 1, 1.0));
  auto train_out = txeo::Matrix<T>::to_matrix(y);
  txeo::TensorFunc<T>::transpose_by(train_out);

  auto Z = txeo::TensorFunc<T>::get_gram_matrix(train_in);
  auto K = txeo::TensorOp<T>::product(train_out, train_in);

  auto norm_X = txeo::TensorAgg<T>::reduce_euclidean_norm(train_in, {0, 1})();
  auto norm_Y = txeo::TensorAgg<T>::reduce_euclidean_norm(train_out, {0, 1})();
  txeo::Matrix<T> B_prev{m, n + 1, norm_Y / norm_X};

  if (_variable_lr)
    _learning_rate = 1.0 / (norm_X * norm_X);

  auto B = B_prev - ((txeo::TensorOp<T>::product(B_prev, Z) - K) * _learning_rate);
  auto L = B - B_prev;
  auto Y_t = y; // VERIFICAR SE PRECISA COPIAR;

  // for (size_t e{0}; e < epochs; ++e) {
  //   auto B_t = txeo::TensorFunc<float>::transpose(B);
  //   auto loss_value = loss.get_loss(txeo::TensorOp<float>::product(X, B_t));
  //   std::cout << "Epoch " << e << ", Loss: " << loss_value << ", Learning Rate: " <<
  //   learning_rate
  //             << std::endl;
  //   if (loss_value < epsilon)
  //     break;

  //   B_prev = B;
  //   if (enable_barzelay_borwein) {
  //     auto LZ = txeo::TensorOp<float>::product(L, Z);
  //     auto numerator = std::abs(txeo::TensorOp<float>::dot(L, LZ));
  //     auto denominator = txeo::TensorOp<float>::dot(LZ, LZ);
  //     learning_rate = numerator / denominator;
  //   };
  //   B -= (txeo::TensorOp<float>::product(B, Z) - K) * learning_rate;
  //   L = B - B_prev;
  // }
}

template class OlsGDTrainer<double>;
template class OlsGDTrainer<float>;

} // namespace txeo
