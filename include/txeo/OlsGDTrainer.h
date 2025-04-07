#ifndef OLSGDTRAINER_H
#define OLSGDTRAINER_H
#pragma once

#include "txeo/Matrix.h"
#include "txeo/Tensor.h"
#include "txeo/TensorShape.h"
#include "txeo/Trainer.h"

#include <concepts>
#include <cstddef>
#include <stdexcept>

namespace txeo {
enum class LossFunc;

/**
 * @class OlsGDTrainer
 * @brief Ordinary Least Squares trainer using Gradient Descent optimization
 *
 * @tparam T Floating-point type for calculations (float, double, etc.)
 *
 * Implements linear regression training through gradient descent with:
 * - Configurable learning rate
 * - Convergence tolerance
 * - Variable learning rate support (Barzilai-Borwein Method)
 * - Weight/bias matrix access
 *
 * Inherits from txeo::Trainer<T> and implements required virtual methods.
 *
 * Implements algorithms based on paper:
 * Algarte, R.D., "Tensor-Based Foundations of Ordinary Least Squares and Neural Network Regression
 * Models" (https://arxiv.org/abs/2411.12873)
 *
 * **Example Usage:**
 * @code
 * // Create training data (y = 2x + 1)
 * txeo::Matrix<double> X({{1.0}, {2.0}, {3.0}}); // 3x1
 * txeo::Matrix<double> y({{3.0}, {5.0}, {7.0}}); // 3x1
 *
 * OlsGDTrainer<double> trainer(X, y);
 * trainer.set_tolerance(1e-5);
 *
 * // Train with early stopping
 * trainer.fit(1000, LossFunc::MSE, 10);
 *
 * if(trainer.is_converged()) {
 *     auto weights = trainer.weight_bias();
 *     std::cout << "Model: y = " << weights(0,0) << "x + " << weights(1,0) << std::endl;
 *
 *     // Make prediction
 *     txeo::Matrix<double> test_input(1,1,{4.0});
 *     auto prediction = trainer.predict(test_input);
 *     std::cout << "Prediction for x=4: " << prediction(0,0) << std::endl;
 * }
 * @endcode
 */
template <typename T>
  requires(std::floating_point<T>)
class OlsGDTrainer : public txeo::Trainer<T> {
  public:
    OlsGDTrainer(const OlsGDTrainer &) = delete;
    OlsGDTrainer(OlsGDTrainer &&) = delete;
    OlsGDTrainer &operator=(const OlsGDTrainer &) = delete;
    OlsGDTrainer &operator=(OlsGDTrainer &&) = delete;
    ~OlsGDTrainer() = default;

    /**
     * @brief Construct a new OlsGD Trainer object from a data table
     *
     * @param data Training/Evaluation/Test data
     */
    OlsGDTrainer(txeo::DataTable<T> &&data) : txeo::Trainer<T>(std::move(data)) {};

    OlsGDTrainer(const txeo::DataTable<T> &data) : txeo::Trainer<T>(data) {};

    /**
     * @brief Makes predictions using learned weights
     *
     * @param input Feature matrix (shape: [samples, features])
     * @return Prediction matrix (shape: [samples, outputs])
     *
     * @throws OlsGDTrainerError
     */
    txeo::Tensor<T> predict(const txeo::Tensor<T> &input) const override;

    /**
     * @brief Gets current learning rate
     *
     * @return Current learning rate value
     */
    [[nodiscard]] T learning_rate() const;

    /**
     * @brief Sets learning rate for gradient descent
     *
     * @param learning_rate Must be > 0
     *
     * @throws OlsGDTrainerError for invalid values
     */
    void set_learning_rate(T learning_rate);

    /**
     * @brief Enables adaptive learning rate adjustment (Barzilai-Borwein Method).
     * When enabled, learning rate automatically reduces when loss plateaus. For the majority of the
     * cases, convergence drastically increases.
     */
    void enable_variable_lr() { _variable_lr = true; }

    /**
     * @brief Disables adaptive learning rate adjustment (Barzilai-Borwein Method)
     */
    void disable_variable_lr() { _variable_lr = false; }

    /**
     * @brief Gets weight/bias matrix related to the minimum loss during fit
     *
     * @return Matrix containing model parameters (shape: [features+1, outputs])
     *
     * @throws OlsGDTrainerError
     */
    const txeo::Matrix<T> &weight_bias() const;

    /**
     * @brief Gets convergence tolerance
     *
     * @return Current tolerance value
     */
    T tolerance() const { return _tolerance; }

    /**
     * @brief Sets convergence tolerance
     *
     * @param tolerance Minimum loss difference to consider converged (>0)
     */
    void set_tolerance(const T &tolerance);

    /**
     * @brief Checks convergence status
     *
     * @return true if training converged before max epochs
     */
    [[nodiscard]] bool is_converged() const { return _is_converged; }

    /**
     * @brief Gets the minimum loss during training
     *
     * @return Value of the minimum loss
     */
    T min_loss() const;

  private:
    T _learning_rate{0.01};
    T _tolerance{0.001};
    T _min_loss{0};
    txeo::Matrix<T> _weight_bias{};
    bool _variable_lr{false};
    bool _is_converged{false};

    OlsGDTrainer() = default;
    void train(size_t epochs, txeo::LossFunc metric) override;
};

/**
 * @brief Exceptions concerning @ref txeo::OlsGDTrainer
 *
 */
class OlsGDTrainerError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo

#endif