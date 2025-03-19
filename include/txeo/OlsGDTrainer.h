#ifndef OLSGDTRAINER_H
#define OLSGDTRAINER_H
#pragma once

#include "txeo/TensorShape.h"
#include "txeo/Trainer.h"

namespace txeo {

template <typename T>
class OlsGDTrainer : public txeo::Trainer<T> {
  public:
    OlsGDTrainer(const OlsGDTrainer &) = delete;
    OlsGDTrainer(OlsGDTrainer &&) = delete;
    OlsGDTrainer &operator=(const OlsGDTrainer &) = delete;
    OlsGDTrainer &operator=(OlsGDTrainer &&) = delete;
    ~OlsGDTrainer();

    OlsGDTrainer(double learning_rate) : _learning_rate{std::move(learning_rate)} {};

    txeo::Tensor<T> predict(const txeo::Tensor<T> input) override;

    [[nodiscard]] double learning_rate() const;
    void set_learning_rate(double learning_rate);

  private:
    double _learning_rate{};
    txeo::Tensor<T> _bias{};
    txeo::Tensor<T> _weight{};

    OlsGDTrainer() = default;
    void train(const txeo::Tensor<T> &X, const txeo::Tensor<T> &y) override;
};

} // namespace txeo

#endif