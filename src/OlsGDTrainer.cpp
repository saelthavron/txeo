#include "txeo/OlsGDTrainer.h"
#include "txeo/Loss.h"
#include "txeo/Matrix.h"
#include "txeo/TensorAgg.h"
#include "txeo/TensorFunc.h"
#include "txeo/TensorOp.h"
#include "txeo/TensorPart.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <utility>

namespace txeo {
enum class LossFunc;

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
Tensor<T> OlsGDTrainer<T>::predict(const Tensor<T> &input) const {
  auto aux = TensorPart<T>::increase_dimension(input, 1, 1.0);
  return TensorOp<T>::product_tensors(aux, this->weight_bias());
}

template <typename T>
  requires(std::floating_point<T>)
void OlsGDTrainer<T>::train(size_t epochs, LossFunc metric) {

  auto &dt_norm = this->_data_table_norm;
  auto &dt = *this->_data_table;

  auto &&x_train = this->_is_norm_enabled ? dt_norm.x_train_normalized() : dt.x_train();
  auto &&y_train = this->_data_table->y_train();

  auto &&x_eval = dt.has_eval()
                      ? (this->_is_norm_enabled ? dt_norm.x_eval_normalized() : *dt.x_eval())
                      : x_train;
  auto &&y_eval = dt.has_eval() ? *dt.y_eval() : y_train;

  // Input and Output data variables
  size_t n = x_train.shape().axis_dim(1);
  size_t m = y_train.shape().axis_dim(1);

  auto &&X = Matrix<T>::to_matrix(TensorPart<T>::increase_dimension(x_train, 1, 1.0));
  auto &&Y = TensorFunc<T>::transpose(y_train);

  auto &&Z = TensorFunc<T>::compute_gram_matrix(X);
  auto &&K = Y.dot(X);

  _is_converged = false;

  // Initializing the loss class
  auto &&X_eval = Matrix<T>::to_matrix(TensorPart<T>::increase_dimension(x_eval, 1, 1.0));
  Loss<T> loss{y_eval, metric};

  // Initial Guesses
  T norm_X = TensorAgg<T>::reduce_euclidean_norm(X, {0, 1})();
  T norm_Y = TensorAgg<T>::reduce_euclidean_norm(Y, {0, 1})();
  Matrix<T> B_prev{m, n + 1, norm_Y / norm_X};

  if (_variable_lr)
    _learning_rate = 1.0 / (norm_X * norm_X);

  auto B = B_prev - (_learning_rate * (B_prev.dot(Z) - K));
  auto L = B - B_prev;

  // Declaring variables to capture trainer params
  T loss_value = std::numeric_limits<T>::max();
  T loss_value_prev = std::numeric_limits<T>::max();
  T min_loss = std::numeric_limits<T>::max();
  size_t patience = 0;
  Matrix<T> B_best{};
  bool found_best{false};

  // Iterate OLS
  for (size_t e{0}; e < epochs; ++e) {
    auto &&B_t = TensorFunc<T>::transpose(B);
    loss_value = loss.get_loss(TensorOp<T>::product_tensors(X_eval, B_t));
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
    if (loss_value < min_loss) {
      found_best = true;
      min_loss = loss_value;
      B_best = B;
    }
    loss_value_prev = loss_value;
    B_prev = B;
    if (_variable_lr) {
      auto LZ = L.dot(Z);
      _learning_rate = std::fabs(L.inner(LZ)) / LZ.inner(LZ);
    };
    B -= _learning_rate * (B.dot(Z) - K);
    L = B - B_prev;
  }

  if (found_best) {
    _min_loss = min_loss;
    _weight_bias = std::move(TensorFunc<T>::transpose_by(B_best));

  } else {
    _min_loss = loss_value;
    _weight_bias = std::move(TensorFunc<T>::transpose_by(B));
  }
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
T OlsGDTrainer<T>::min_loss() const {
  if (!this->_is_trained)
    throw OlsGDTrainerError("Trainer is not trained.");
  return _min_loss;
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
