#ifndef OLSGDTRAINER_H
#define OLSGDTRAINER_H
#pragma once

#include "txeo/Matrix.h"
#include "txeo/TensorShape.h"
#include "txeo/Trainer.h"

#include <concepts>
#include <cstddef>
namespace txeo {

template <typename T>
  requires(std::floating_point<T>)
class OlsGDTrainer : public txeo::Trainer<T> {
  public:
    OlsGDTrainer(const OlsGDTrainer &) = delete;
    OlsGDTrainer(OlsGDTrainer &&) = delete;
    OlsGDTrainer &operator=(const OlsGDTrainer &) = delete;
    OlsGDTrainer &operator=(OlsGDTrainer &&) = delete;
    ~OlsGDTrainer() = default;

    OlsGDTrainer(const txeo::Matrix<T> &x_train, const txeo::Matrix<T> &y_train,
                 const txeo::Matrix<T> &x_valid, const txeo::Matrix<T> &y_valid)
        : txeo::Trainer<T>{x_train, y_train, x_valid, y_valid} {};

    OlsGDTrainer(const txeo::Matrix<T> &x_train, const txeo::Matrix<T> &y_train)
        : OlsGDTrainer{x_train, y_train, x_train, y_train} {}

    txeo::Tensor<T> predict(const txeo::Tensor<T> &input) override;

    [[nodiscard]] T learning_rate() const;
    void set_learning_rate(T learning_rate);

    void enable_variable_lr() { _variable_lr = true; }
    void disable_variable_lr() { _variable_lr = false; }

    const txeo::Matrix<T> &weight_bias() const;

    T tolerance() const { return _tolerance; }
    void set_tolerance(const T &tolerance);

    [[nodiscard]] bool is_converged() const { return _is_converged; }

  private:
    T _learning_rate{0.01};
    T _tolerance{0.001};
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