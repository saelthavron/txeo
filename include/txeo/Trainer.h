#ifndef TRAINER_H
#define TRAINER_H
#include "txeo/Loss.h"
#pragma once

#include "txeo/Tensor.h"

#include <cstddef>

namespace txeo {

template <typename T>
class Trainer {
  public:
    Trainer(const Trainer &) = default;
    Trainer(Trainer &&) = default;
    Trainer &operator=(const Trainer &) = default;
    Trainer &operator=(Trainer &&) = default;
    virtual ~Trainer() = default;

    Trainer(const txeo::Loss<T> &loss, size_t epochs) : _loss{&loss}, _epochs{epochs} {}

    virtual void fit(const txeo::Tensor<T> &X, const txeo::Tensor<T> &y);

    virtual txeo::Tensor<T> predict(const txeo::Tensor<T> input) = 0;

    [[nodiscard]] size_t epochs() const { return _epochs; }
    [[nodiscard]] double epsilon() const { return _epsilon; }

    void set_epochs(const size_t &epochs);
    void set_epsilon(double epsilon);

    [[nodiscard]] bool is_trained() const { return _is_trained; }

    void enable_early_stopping(double epsilon, size_t patience = 5);

    void disable_early_stopping();

  protected:
    Trainer() = default;

    const txeo::Loss<T> *_loss;
    size_t _epochs{};
    double _epsilon{0.01};
    bool _is_trained{false};
    bool _is_converged{false};
    bool _is_early_stop{false};
    size_t _patience{5};

    virtual void train(const txeo::Tensor<T> &X, const txeo::Tensor<T> &y) = 0;

    void reset();
};

} // namespace txeo

#endif