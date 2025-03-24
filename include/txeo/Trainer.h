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

    Trainer(const txeo::Tensor<T> &x_train, const txeo::Tensor<T> &y_train,
            const txeo::Tensor<T> &x_valid, const txeo::Tensor<T> &y_valid);

    Trainer(const txeo::Tensor<T> &x_train, const txeo::Tensor<T> &y_train)
        : Trainer{x_train, y_train, x_train, y_train} {}

    virtual void fit(size_t epochs, txeo::LossFunc metric);

    virtual void fit(size_t epochs, txeo::LossFunc metric, size_t patience);

    virtual txeo::Tensor<T> predict(const txeo::Tensor<T> &input) = 0;

    [[nodiscard]] bool is_trained() const { return _is_trained; }

  protected:
    Trainer() = default;

    bool _is_trained{false};
    bool _is_early_stop{false};
    size_t _patience{0};

    const txeo::Tensor<T> *_x_train;
    const txeo::Tensor<T> *_y_train;
    const txeo::Tensor<T> *_x_valid;
    const txeo::Tensor<T> *_y_valid;

    virtual void train(size_t epochs, txeo::LossFunc loss_func) = 0;
};

/**
 * @brief Exceptions concerning @ref txeo::OlsGDTrainer
 *
 */
class TrainerError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo

#endif