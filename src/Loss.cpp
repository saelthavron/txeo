#include "txeo/Loss.h"

#include <cmath>

#include <cstdlib>

namespace txeo {

template <typename T>
void Loss<T>::set_loss(LossFunc func) {
  switch (func) {
  case LossFunc::MSE:
    _loss_func = [this](const Tensor<T> &pred) -> T { return this->mse(pred); };
    break;
  case LossFunc::MAE:
    _loss_func = [this](const Tensor<T> &pred) -> T { return this->mae(pred); };
    break;
  case LossFunc::MSLE:
    _loss_func = [this](const Tensor<T> &pred) -> T { return this->msle(pred); };
    break;
  case LossFunc::LCHE:
    _loss_func = [this](const Tensor<T> &pred) -> T { return this->lche(pred); };
    break;
  }
}

template <typename T>
Loss<T>::Loss(const Tensor<T> &valid, LossFunc func) : _valid{&valid} {
  if (_valid->dim() == 0)
    throw LossError("Tensor has dimension zero.");

  this->set_loss(func);
}

template <typename T>
void Loss<T>::verify_parameter(const Tensor<T> &pred) const {
  if (pred.dim() == 0)
    throw LossError("Tensor has dimension zero.");
  if (pred.shape() != _valid->shape())
    throw LossError("Incompatible shape.");
}

template <typename T>
T Loss<T>::mean_squared_error(const Tensor<T> &pred) const {
  this->verify_parameter(pred);

  T resp = 0;
  auto pred_flat = pred.data();
  auto valid_flat = _valid->data();

  for (size_t i{0}; i < pred.dim(); ++i) {
    auto aux = pred_flat[i] - valid_flat[i];
    resp += aux * aux;
  }

  return resp / pred.dim();
}

template <typename T>
T Loss<T>::mean_absolute_error(const Tensor<T> &pred) const {
  this->verify_parameter(pred);

  T resp = 0;
  auto pred_flat = pred.data();
  auto valid_flat = _valid->data();

  for (size_t i{0}; i < pred.dim(); ++i)
    resp += std::abs(pred_flat[i] - valid_flat[i]);

  return resp / pred.dim();
}

template <>
size_t Loss<size_t>::mean_absolute_error(const Tensor<size_t> &pred) const {
  this->verify_parameter(pred);

  size_t resp = 0;
  auto pred_flat = pred.data();
  auto valid_flat = _valid->data();

  for (size_t i{0}; i < pred.dim(); ++i) {
    resp +=
        pred_flat[i] > valid_flat[i] ? pred_flat[i] - valid_flat[i] : valid_flat[i] - pred_flat[i];
  }

  return resp / pred.shape().axis_dim(0);
}

template <typename T>
T Loss<T>::mean_squared_logarithmic_error(const Tensor<T> &pred) const {
  this->verify_parameter(pred);

  T resp = 0;
  auto pred_flat = pred.data();
  auto valid_flat = _valid->data();

  for (size_t i{0}; i < pred.dim(); ++i) {
    if (pred_flat[i] < 0 || valid_flat[i] < 0)
      throw LossError("A tensor element is negative.");

    auto aux = std::log1p(pred_flat[i]) - std::log1p(valid_flat[i]);
    resp += aux * aux;
  }

  return resp / pred.shape().axis_dim(0);
}

template <typename T>
T Loss<T>::log_cosh_error(const Tensor<T> &pred) const {

  this->verify_parameter(pred);

  T resp = 0;
  auto pred_flat = pred.data();
  auto valid_flat = _valid->data();

  for (size_t i{0}; i < pred.dim(); ++i)
    resp += std::log(std::cosh(pred_flat[i] - valid_flat[i]));

  return resp / pred.shape().axis_dim(0);
}

template <typename T>
T Loss<T>::get_loss(const Tensor<T> &pred) const {
  return _loss_func(pred);
}

template class Loss<size_t>;
template class Loss<short>;
template class Loss<int>;
template class Loss<bool>;
template class Loss<long>;
template class Loss<long long>;
template class Loss<float>;
template class Loss<double>;

} // namespace txeo
