#ifndef TRAINER_H
#define TRAINER_H
#pragma once

#include "txeo/DataTable.h"
#include "txeo/Tensor.h"
#include "types.h"

#include <cstddef>
#include <stdexcept>

namespace txeo {
enum class LossFunc;

/**
 * @class Trainer
 * @brief Abstract base class for machine learning trainers
 *
 * @tparam T Numeric type for tensor elements (e.g., float, double)
 *
 * This class provides the core interface for training machine learning models with:
 * - Training/evaluation data management
 * - Common training parameters
 * - Basic training lifecycle control
 *
 * Derived classes must implement the prediction logic and training algorithm.
 */
template <typename T>
class Trainer {
  public:
    Trainer(const Trainer &) = default;
    Trainer(Trainer &&) = default;
    Trainer &operator=(const Trainer &) = default;
    Trainer &operator=(Trainer &&) = default;
    virtual ~Trainer() = default;

    /**
     * @brief Construct a new Trainer object from a data table.
     *
     * @param data Training/Evaluation/Test data. If an rvalue is informed, no copy is made
     */
    Trainer(const txeo::DataTable<T> &data)
        : _data_table{std::make_unique<txeo::DataTable<T>>(std::move(data))} {};

    /**
     * @brief Trains the model for specified number of epochs
     *
     * @param epochs Number of training iterations
     * @param metric Loss function to optimize
     */
    virtual void fit(size_t epochs, txeo::LossFunc metric);

    /**
     * @brief Trains with early stopping based on validation performance
     *
     * @param epochs Maximum number of training iterations
     * @param metric Loss function to optimize
     * @param patience Number of epochs to wait without improvement before stopping
     */
    virtual void fit(size_t epochs, txeo::LossFunc metric, size_t patience);

    /**
     * @brief Evaluates test data based on a specified metric
     *
     *@ throws txeo::TrainerError
     *
     * @param metric Loss function type
     * @return T Loss value
     */
    T compute_test_loss(txeo::LossFunc metric) const;

    /**
     * @brief Makes predictions using the trained model (pure virtual)
     *
     * @param input Input tensor for prediction (shape: [samples, features])
     * @return Prediction tensor (shape: [samples, outputs])
     *
     * @throws txeo::TrainerError
     *
     * @note Must be implemented in derived classes
     */
    virtual txeo::Tensor<T> predict(const txeo::Tensor<T> &input) const = 0;

    /**
     * @brief Checks if model has been trained
     *
     * @return true if training has been completed, false otherwise
     */
    [[nodiscard]] bool is_trained() const { return _is_trained; }

    /**
     * @brief Returns the data table of this trainer
     *
     * @return const txeo::DataTable<T>&
     */
    const txeo::DataTable<T> &data_table() const { return *_data_table; }

  protected:
    Trainer() = default;

    bool _is_trained{false};
    bool _is_early_stop{false};
    size_t _patience{0};

    std::unique_ptr<txeo::DataTable<T>> _data_table;

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