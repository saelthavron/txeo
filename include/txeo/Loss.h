#ifndef LOSS_H
#define LOSS_H
#pragma once

#include "txeo/Tensor.h"
#include "txeo/types.h"

#include <functional>

namespace txeo {

/**
 * @class Loss
 * @brief Computes error metrics between predicted and validation tensors.
 *
 * Supports multiple loss functions that can be selected at runtime:
 * - MSE (Mean Squared Error)
 * - MAE (Mean Absolute Error)
 * - MSLE (Mean Squared Logarithmic Error)
 * - LCHE (Log-Cosh Error)
 *
 * @tparam T Numeric type of tensor elements (float/double recommended)
 *
 * @note All operations validate tensor shape compatibility between predictions and validation data;
 * @note First axis's dimension of the tensors involved must refer to number of samples
 */
template <typename T>
class Loss {
  public:
    Loss(const Loss &) = default;
    Loss(Loss &&) = default;
    Loss &operator=(const Loss &) = default;
    Loss &operator=(Loss &&) = default;
    ~Loss() = default;

    /**
     * @brief Construct a new Loss object
     *
     * @param valid Validation tensor containing ground truth values
     * @param func Initial loss function (default: MSE)
     *
     * @par Example:
     * @code
     * // Create with MAE as default loss
     * txeo::Loss<float> loss(validation_tensor, txeo::LossFunc::MAE);
     * @endcode
     */
    explicit Loss(const txeo::Tensor<T> &valid, txeo::LossFunc func = txeo::LossFunc::MSE);

    /**
     * @brief Compute loss using currently selected function
     *
     * @param pred Prediction tensor
     * @return T Calculated loss value
     * @throw LossError If shapes mismatch or invalid input values
     *
     * @par Example:
     * @code
     * auto error = loss.get_loss(predictions);
     * std::cout << "Current loss: " << error << std::endl;
     * @endcode
     */
    T get_loss(const txeo::Tensor<T> &pred) const;

    /**
     * @brief Set the active loss function
     *
     * @param func Loss function to use for subsequent calculations
     *
     * @par Example:
     * @code
     * // Switch to log-cosh error
     * loss.set_loss(txeo::LossFunc::LCHE);
     * @endcode
     */
    void set_loss(txeo::LossFunc func);

    /**
     * @brief Compute Mean Squared Error (MSE)
     *
     * @f[ MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2 @f]
     *
     * @param pred Prediction tensor
     * @return T MSE value
     */
    T mean_squared_error(const txeo::Tensor<T> &pred) const;

    /**
     * @brief Compute Mean Absolute Error (MAE)
     *
     * @f[ MAE = \frac{1}{N}\sum_{i=1}^{N}|y_i - \hat{y}_i| @f]
     *
     * @param pred Prediction tensor
     * @return T MAE value
     */
    T mean_absolute_error(const txeo::Tensor<T> &pred) const;

    /**
     * @brief Compute Mean Squared Logarithmic Error (MSLE)
     *
     * @f[ MSLE = \frac{1}{N}\sum_{i=1}^{N}(\log(1+y_i) - \log(1+\hat{y}_i))^2 @f]
     *
     * @param pred Prediction tensor
     * @return T MSLE value
     * @throw LossError If any values are negative
     */
    T mean_squared_logarithmic_error(const txeo::Tensor<T> &pred) const;

    /**
     * @brief Compute Log-Cosh Error (LCHE)
     *
     * @f[ LCHE = \frac{1}{N}\sum_{i=1}^{N}\log(\cosh(y_i - \hat{y}_i)) @f]
     *
     * @param pred Prediction tensor
     * @return T LCHE value
     */
    T log_cosh_error(const txeo::Tensor<T> &pred) const;

    /// @name Shorthand Aliases
    /// @{
    T mse(const txeo::Tensor<T> &pred) const {
      return mean_squared_error(pred);
    }; ///< @see mean_squared_error

    T mae(const txeo::Tensor<T> &pred) const {
      return mean_absolute_error(pred);
    }; ///< @see mean_absolute_error

    T msle(const txeo::Tensor<T> &pred) const {
      return mean_squared_logarithmic_error(pred);
    }; ///< @see mean_squared_logarithmic_error

    T lche(const txeo::Tensor<T> &pred) const {
      return log_cosh_error(pred);
    }; ///< @see log_cosh_error
    /// @}

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