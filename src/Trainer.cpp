#include "txeo/Trainer.h"

namespace txeo {

template <typename T>
inline void Trainer<T>::reset() {
  _is_trained = false;
  _is_converged = false;
}

template <typename T>
void Trainer<T>::fit(const txeo::Tensor<T> &X, const txeo::Tensor<T> &y) {
  this->train(X, y);
  _is_trained = true;
};

template <typename T>
void Trainer<T>::set_epochs(const size_t &epochs) {
  this->reset();
  _epochs = epochs;
}

template <typename T>
void Trainer<T>::set_epsilon(double epsilon) {
  this->reset();
  _epsilon = epsilon;
}

template <typename T>
void Trainer<T>::enable_early_stopping(double epsilon, size_t patience) {
  this->reset();
  _is_early_stop = true;
  _epsilon = epsilon;
  _patience = patience;
}

template <typename T>
inline void Trainer<T>::disable_early_stopping() {
  _is_early_stop = false;
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