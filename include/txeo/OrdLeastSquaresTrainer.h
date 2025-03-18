#ifndef ORDLEASTSQUARESTRAINER_H
#define ORDLEASTSQUARESTRAINER_H
#pragma once

#include "txeo/TensorShape.h"
#include "txeo/Trainer.h"

namespace txeo {

template <typename T>
class OrdLeastSquaresTrainer : public txeo::Trainer<T> {
  public:
    OrdLeastSquaresTrainer(const OrdLeastSquaresTrainer &) = delete;
    OrdLeastSquaresTrainer(OrdLeastSquaresTrainer &&) = delete;
    OrdLeastSquaresTrainer &operator=(const OrdLeastSquaresTrainer &) = delete;
    OrdLeastSquaresTrainer &operator=(OrdLeastSquaresTrainer &&) = delete;
    ~OrdLeastSquaresTrainer();

    OrdLeastSquaresTrainer(double learning_rate) : _learning_rate{std::move(learning_rate)} {};

    txeo::Tensor<T> predict(const txeo::Tensor<T> input) override;

    [[nodiscard]] double learning_rate() const;
    void set_learning_rate(double learning_rate);

  private:
    double _learning_rate{};
    txeo::Tensor<T> _bias{};
    txeo::Tensor<T> _weight{};

    OrdLeastSquaresTrainer() = default;
    void train(const txeo::Tensor<T> &X, const txeo::Tensor<T> &y) override;
};

} // namespace txeo

#endif