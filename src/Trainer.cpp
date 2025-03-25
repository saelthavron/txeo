#include "txeo/Trainer.h"

namespace txeo {

template <typename T>
inline Trainer<T>::Trainer(const Tensor<T> &x_train, const Tensor<T> &y_train,
                           const Tensor<T> &x_eval, const Tensor<T> &y_eval)
    : _x_train{&x_train}, _y_train{&y_train}, _x_eval{&x_eval}, _y_eval{&y_eval} {
  if (x_train.dim() == 0 || y_train.dim() == 0 || x_eval.dim() == 0 || y_eval.dim() == 0)
    throw TrainerError("One of the tensors has zero dimension.");

  if (x_train.shape().axis_dim(0) != y_train.shape().axis_dim(0) ||
      x_eval.shape().axis_dim(0) != y_eval.shape().axis_dim(0))
    throw TrainerError("Training or Validation tensor are incompatible.");
};

template <typename T>
void Trainer<T>::fit(size_t epochs, LossFunc metric) {
  this->train(epochs, metric);
  _is_trained = true;
};

template <typename T>
inline void Trainer<T>::fit(size_t epochs, LossFunc metric, size_t patience) {
  _is_early_stop = true;
  _patience = patience;
  this->fit(epochs, metric);
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