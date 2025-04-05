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
T Trainer<T>::compute_test_loss(txeo::LossFunc metric) const {
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
void Trainer<T>::enable_data_table_norm(txeo::NormalizationType type) {
  _data_table_norm = txeo::DataTableNorm<T>{*_data_table, type};
  _is_norm_enabled = true;
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