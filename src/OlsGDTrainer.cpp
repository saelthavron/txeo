#include "txeo/OlsGDTrainer.h"
#include "txeo/Loss.h"
#include "txeo/Matrix.h"
#include "txeo/TensorAgg.h"
#include "txeo/TensorFunc.h"
#include "txeo/TensorOp.h"
#include "txeo/TensorPart.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <utility>

namespace txeo {

template <typename T>
  requires(std::floating_point<T>)
void OlsGDTrainer<T>::set_learning_rate(T learning_rate) {
  this->_is_trained = false;
  _learning_rate = learning_rate;
}

template <typename T>
  requires(std::floating_point<T>)
T OlsGDTrainer<T>::learning_rate() const {
  return _learning_rate;
}

template <typename T>
  requires(std::floating_point<T>)
Tensor<T> OlsGDTrainer<T>::predict(const Tensor<T> &input) {
  return TensorOp<T>::product_tensors(input, this->weight_bias());
}

template <typename T>
  requires(std::floating_point<T>)
void OlsGDTrainer<T>::train(size_t epochs, LossFunc metric) {

  size_t n = this->_x_train->shape().axis_dim(1);
  size_t m = this->_y_train->shape().axis_dim(1);

  auto train_in = Matrix<T>::to_matrix(TensorPart<T>::increase_dimension(*this->_x_train, 1, 1.0));
  auto train_out = Matrix<T>::to_matrix(*this->_y_train).transpose();

  auto Z = TensorFunc<T>::compute_gram_matrix(train_in);
  auto K = train_out.prod(train_in);

  auto norm_X = TensorAgg<T>::reduce_euclidean_norm(train_in, {0, 1})();
  auto norm_Y = TensorAgg<T>::reduce_euclidean_norm(train_out, {0, 1})();
  Matrix<T> B_prev{m, n + 1, norm_Y / norm_X};

  if (_variable_lr)
    _learning_rate = 1.0 / (norm_X * norm_X);

  auto B = B_prev - (_learning_rate * (B_prev.prod(Z) - K));
  auto L = B - B_prev;
  _is_converged = false;

  auto *valid_in = &train_in;
  if (this->_x_train != this->_x_valid)
    *valid_in = Matrix<T>::to_matrix(TensorPart<T>::increase_dimension(*this->_x_valid, 1, 1.0));

  Loss<T> loss{*this->_y_valid, metric};

  T loss_value_prev = std::numeric_limits<T>::max();
  size_t patience = 0;
  for (size_t e{0}; e < epochs; ++e) {
    auto B_t = TensorFunc<T>::transpose(B);
    auto loss_value = loss.get_loss(TensorOp<T>::product_tensors(*valid_in, B_t));
    std::cout << "Epoch " << e << ", Loss: " << loss_value << ", Learning Rate: " << _learning_rate
              << std::endl;
    if (std::isnan(loss_value)) {
      _is_converged = false;
      break;
    }
    if (loss_value >= loss_value_prev && this->_is_early_stop) {
      if (patience == this->_patience) {
        _is_converged = false;
        break;
      } else
        ++patience;
    } else {
      if (loss_value < _tolerance) {
        _is_converged = true;
        break;
      }
      patience = 0;
    }
    loss_value_prev = loss_value;
    B_prev = B;
    if (_variable_lr) {
      auto LZ = L.prod(Z);
      _learning_rate = std::fabs(L.dot(LZ)) / LZ.dot(LZ);
    };
    B -= _learning_rate * (B.prod(Z) - K);
    L = B - B_prev;
  }

  _weight_bias = std::move(TensorFunc<T>::transpose_by(B));
}

template <typename T>
  requires(std::floating_point<T>)
const Matrix<T> &OlsGDTrainer<T>::weight_bias() const {
  if (!this->_is_trained)
    throw OlsGDTrainerError("Trainer is not trained.");
  return _weight_bias;
}

template <typename T>
  requires(std::floating_point<T>)
void OlsGDTrainer<T>::set_tolerance(const T &tolerance) {
  this->_is_trained = false;
  _tolerance = tolerance;
}

template class OlsGDTrainer<double>;
template class OlsGDTrainer<float>;

} // namespace txeo
