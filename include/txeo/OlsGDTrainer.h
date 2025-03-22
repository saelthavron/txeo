#ifndef OLSGDTRAINER_H
#define OLSGDTRAINER_H
#pragma once

#include "txeo/Loss.h"
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

    OlsGDTrainer(const txeo::Loss<T> &loss, size_t epochs) : txeo::Trainer<T>{loss, epochs} {};

    txeo::Tensor<T> predict(const txeo::Tensor<T> input) override;

    [[nodiscard]] T learning_rate() const;
    void set_learning_rate(T learning_rate);

    void enable_variable_lr() { _variable_lr = true; }
    void disable_variable_lr() { _variable_lr = false; }

  private:
    T _learning_rate{0.01};
    txeo::Tensor<T> _bias{};
    txeo::Tensor<T> _weight{};
    bool _variable_lr{false};

    OlsGDTrainer() = default;
    void train(const txeo::Tensor<T> &X, const txeo::Tensor<T> &y) override;
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