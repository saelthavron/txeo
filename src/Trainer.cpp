#include "txeo/Trainer.h"
#include "txeo/Loss.h"

namespace txeo {
enum class LossFunc;

template <typename T>
void Trainer<T>::fit(size_t epochs, LossFunc metric) {
  this->train(epochs, metric);
  _is_trained = true;
};

template <typename T>
void Trainer<T>::fit(size_t epochs, LossFunc metric, size_t patience) {
  _is_early_stop = true;
  _patience = patience;
  this->fit(epochs, metric);
  _is_early_stop = false;
}

template <typename T>
void Trainer<T>::fit(size_t epochs, LossFunc metric, size_t patience, NormalizationType type) {
  this->enable_feature_norm(type);
  this->fit(epochs, metric, patience);
}

template <typename T>
T Trainer<T>::compute_test_loss(LossFunc metric) const {
  if (!this->_is_trained)
    throw TrainerError("Trainer is not trained.");

  auto *x_test = this->_data_table->x_test();
  auto *y_test = this->_data_table->y_test();

  if (x_test == nullptr)
    throw TrainerError("Test data was not specified.");

  Loss<T> loss{*y_test, metric};

  return loss.get_loss(this->predict(*x_test));
}

template <typename T>
void Trainer<T>::enable_feature_norm(NormalizationType type) {
  _data_table_norm = DataTableNorm<T>{*_data_table, type};
  _is_norm_enabled = true;
  _is_trained = false;
}

template class Trainer<size_t>;
template class Trainer<short>;
template class Trainer<int>;
template class Trainer<bool>;
template class Trainer<long>;
template class Trainer<long long>;
template class Trainer<float>;
template class Trainer<double>;

} // namespace txeo