#include "txeo/OlsGDTrainer.h"
#include "txeo/Loss.h"
#include "txeo/Matrix.h"
#include "txeo/TensorAgg.h"
#include "txeo/TensorFunc.h"
#include "txeo/TensorOp.h"
#include "txeo/TensorPart.h"
#include <iostream>
#include <utility>

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
txeo::Tensor<T> OlsGDTrainer<T>::predict(const txeo::Tensor<T> &input) {
  return txeo::TensorOp<T>::product_tensors(input, this->weight_bias());
}

template <typename T>
  requires(std::floating_point<T>)
void OlsGDTrainer<T>::train(size_t epochs, txeo::LossFunc metric) {

  size_t n = this->_x_train->shape().axis_dim(1);
  size_t m = this->_y_train->shape().axis_dim(1);

  auto train_in =
      txeo::Matrix<T>::to_matrix(txeo::TensorPart<T>::increment_dimension(*this->_x_train, 1, 1.0));
  auto train_out = txeo::Matrix<T>::to_matrix(*this->_y_train);
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
  this->_is_converged = false;

  auto *valid_in = &train_in;
  if (this->_x_train != this->_x_valid) {
    *valid_in = txeo::Matrix<T>::to_matrix(
        txeo::TensorPart<T>::increment_dimension(*this->_x_valid, 1, 1.0));
  }
  txeo::Loss<T> loss{*this->_y_valid, metric};

  auto lim = this->_is_early_stop ? epochs + this->_patience : epochs;
  for (size_t e{0}; e < lim; ++e) {
    auto B_t = txeo::TensorFunc<T>::transpose(B);
    auto loss_value = loss.get_loss(txeo::TensorOp<T>::product_tensors(*valid_in, B_t));
    std::cout << "Epoch " << e << ", Loss: " << loss_value << ", Learning Rate: " << _learning_rate
              << std::endl;
    if (loss_value < this->_epsilon && this->_is_early_stop) {
      this->_is_converged = true;
      break;
    }

    B_prev = B;
    if (_variable_lr) {
      auto LZ = txeo::TensorOp<T>::product(L, Z);
      auto numerator = std::abs(txeo::TensorOp<T>::dot(L, LZ));
      auto denominator = txeo::TensorOp<T>::dot(LZ, LZ);
      _learning_rate = numerator / denominator;
    };
    B -= (txeo::TensorOp<T>::product(B, Z) - K) * _learning_rate;
    L = B - B_prev;
  }

  _weight_bias = std::move(txeo::TensorFunc<T>::transpose_by(B));
}

template <typename T>
  requires(std::floating_point<T>)
const txeo::Matrix<T> &OlsGDTrainer<T>::weight_bias() const {
  if (!this->_is_trained)
    throw txeo::OlsGDTrainerError("Trainer is not trained.");
  return _weight_bias;
}

template class OlsGDTrainer<double>;
template class OlsGDTrainer<float>;

} // namespace txeo
