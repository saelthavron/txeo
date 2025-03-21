#ifndef LOSS_H
#define LOSS_H
#pragma once

#include "txeo/Tensor.h"

#include <functional>

namespace txeo {

enum class LossFunc { MSE, MAE, MSLE, LCHE };

template <typename T>
class Loss {
  public:
    Loss(const Loss &) = default;
    Loss(Loss &&) = default;
    Loss &operator=(const Loss &) = default;
    Loss &operator=(Loss &&) = default;
    ~Loss() = default;

    explicit Loss(const txeo::Tensor<T> &valid, txeo::LossFunc func = txeo::LossFunc::MSE);

    T mean_squared_error(const txeo::Tensor<T> &pred) const;
    T mse(const txeo::Tensor<T> &pred) const { return mean_squared_error(pred); };

    T mean_absolute_error(const txeo::Tensor<T> &pred) const;
    T mae(const txeo::Tensor<T> &pred) const { return mean_absolute_error(pred); };

    T mean_squared_logarithmic_error(const txeo::Tensor<T> &pred) const;
    T msle(const txeo::Tensor<T> &pred) const { return mean_squared_logarithmic_error(pred); };

    T log_cosh_error(const txeo::Tensor<T> &pred) const;
    T lche(const txeo::Tensor<T> &pred) const { return log_cosh_error(pred); };

    T get_loss(const txeo::Tensor<T> &pred) const;
    void set_loss(txeo::LossFunc func);

  private:
    Loss() = default;
    const txeo::Tensor<T> *_valid;

    void verify_parameter(const txeo::Tensor<T> &pred) const;
    std::function<T(const txeo::Tensor<T> &)> _loss_func;
};

/**
 * @brief Exceptions concerning @ref txeo::Loss
 *
 */
class LossError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo

#endif