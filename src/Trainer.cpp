#include "txeo/Trainer.h"

namespace txeo {

template <typename T>
inline Trainer<T>::Trainer(const txeo::Tensor<T> &x_train, const txeo::Tensor<T> &y_train,
                           const txeo::Tensor<T> &x_valid, const txeo::Tensor<T> &y_valid)
    : _x_train{&x_train}, _y_train{&y_train}, _x_valid{&x_valid}, _y_valid{&y_valid} {
  if (x_train.dim() == 0 || y_train.dim() == 0 || x_valid.dim() == 0 || y_valid.dim() == 0)
    throw TrainerError("One of the tensors has zero dimension.");

  if (x_train.shape().axis_dim(0) != y_train.shape().axis_dim(0) ||
      x_valid.shape().axis_dim(0) != y_valid.shape().axis_dim(0))
    throw TrainerError("Training or Validation tensor are incompatible.");
};

template <typename T>
void Trainer<T>::reset() {
  _is_trained = false;
  _is_converged = false;
}

template <typename T>
void Trainer<T>::fit(size_t epochs, txeo::LossFunc metric) {
  this->train(epochs, metric);
  _is_trained = true;
};

template <typename T>
inline void Trainer<T>::fit(size_t epochs, txeo::LossFunc metric, T epsilon, size_t patience) {
  if (epsilon < 0)
    throw TrainerError("Tolerance cannot be negative.");
  _is_early_stop = true;
  _epsilon = epsilon;
  _patience = patience;
  this->fit(epochs, metric);
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