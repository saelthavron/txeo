#ifndef DATATABLENORM_H
#define DATATABLENORM_H
#include <functional>
#pragma once

#include "txeo/DataTable.h"
#include "txeo/types.h"

namespace txeo {

/**
 * @class DataTableNorm
 * @brief A normalizer for DataTable objects that handles feature scaling
 *
 * This class provides normalization functionality for machine learning datasets
 * stored in DataTable format. It supports both Min-Max scaling and Z-Score
 * standardization, computing normalization parameters from the training data.
 *
 * @tparam T The data type of the table elements (e.g., float, double)
 *
 * @note The normalizer computes normalization parameters (min/max or mean/std)
 *       from the training portion of the DataTable. Make sure the DataTable
 *       is properly initialized with training data before use.
 */
template <typename T>
class DataTableNorm {
  public:
    DataTableNorm() = default;

    DataTableNorm(const DataTableNorm &) = delete;
    DataTableNorm(DataTableNorm &&) = default;
    DataTableNorm &operator=(const DataTableNorm &) = delete;
    DataTableNorm &operator=(DataTableNorm &&) = default;
    ~DataTableNorm() = default;

    /**
     * @brief Construct a new DataTableNorm object with associated DataTable
     *
     * @param data Reference to the DataTable containing training data
     * @param type Normalization type (MIN_MAX or Z_SCORE)
     *
     * @throws DataTableNormError
     *
     * **Example Usage:**
     * @code
     * txeo::DataTable<double> data = load_my_dataset();
     * txeo::DataTableNorm<double> normalizer(data, txeo::NormalizationType::Z_SCORE);
     *
     * // Normalize a new sample
     * txeo::Matrix<double> sample = {{1.5}, {2.0}, {3.5}};
     * auto normalized_sample = normalizer.normalize(sample);
     * @endcode
     */
    DataTableNorm(const txeo::DataTable<T> &data,
                  txeo::NormalizationType type = txeo::NormalizationType::MIN_MAX);

    /**
     * @brief Get the associated DataTable
     * @return const txeo::DataTable<T>& Reference to the stored DataTable
     *
     * **Example Usage:**
     * @code
     * // Inspect the data table used for normalization
     * const auto& dt = normalizer.data_table();
     * std::cout << "Original dataset size: " << dt.x_train().rows() << std::endl;
     * @endcode
     */
    const txeo::DataTable<T> &data_table() const { return *_data_table; }

    /**
     * @brief Set a new DataTable for normalization
     *
     * @param data New DataTable to use for normalization parameters
     *
     * @throws DataTableNormError If DataTable is invalid
     *
     * **Example Usage:**
     * @code
     * txeo::DataTable<double> new_data = load_updated_dataset();
     * normalizer.set_data_table(new_data);  // Update normalization parameters
     * @endcode
     */
    void set_data_table(const txeo::DataTable<T> &data);

    /**
     * @brief Get the current normalization type
     * @return txeo::NormalizationType Active normalization method
     *
     * **Example Usage:**
     * @code
     * if(normalizer.type() == txeo::NormalizationType::MIN_MAX) {
     *     std::cout << "Using Min-Max normalization" << std::endl;
     * }
     * @endcode
     */
    [[nodiscard]] txeo::NormalizationType type() const { return _type; }

    /**
     * @brief Normalize a matrix using rvalue semantics (move data)
     *
     * @param x Input matrix to normalize (will be moved from)
     * @return txeo::Matrix<T> Normalized matrix
     *
     * @throws DataTableNormError If normalization parameters not initialized
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<double> large_matrix = generate_large_data();
     * // Efficiently normalize without copying
     * auto normalized = normalizer.normalize(std::move(large_matrix));
     * @endcode
     */
    txeo::Matrix<T> normalize(txeo::Matrix<T> &&x) const;

    /**
     * @brief Normalize a copy of the input matrix
     *
     * @param x Input matrix to normalize
     * @return txeo::Matrix<T> Normalized copy of the matrix
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<double> original = {{1.0}, {2.0}, {3.0}};
     * auto normalized = normalizer.normalize(original);  // Preserve original
     * @endcode
     */
    txeo::Matrix<T> normalize(const txeo::Matrix<T> &x) const {
      return this->normalize(x.clone());
    };

    /**
     * @brief Get normalized training data
     * @return txeo::Matrix<T> Normalized training dataset
     *
     * @throws DataTableNormError If training data not available
     *
     * **Example Usage:**
     * @code
     * auto x_train_norm = normalizer.x_train_normalized();
     * model.train(x_train_norm, normalizer.data_table().y_train());
     * @endcode
     */
    txeo::Matrix<T> x_train_normalized();

    /**
     * @brief Get normalized evaluation data
     * @return txeo::Matrix<T> Normalized evaluation dataset
     *
     * **Example Usage:**
     * @code
     * auto x_eval_norm = normalizer.x_eval_normalized();
     * auto eval_score = model.evaluate(x_eval_norm, normalizer.data_table().y_eval());
     * @endcode
     */
    txeo::Matrix<T> x_eval_normalized();

    /**
     * @brief Get normalized test data
     * @return txeo::Matrix<T> Normalized test dataset
     *
     * **Example Usage:**
     * @code
     * auto x_test_norm = normalizer.x_test_normalized();
     * auto test_acc = model.test(x_test_norm, normalizer.data_table().y_test());
     * @endcode
     */
    txeo::Matrix<T> x_test_normalized();

  private:
    txeo::NormalizationType _type{txeo::NormalizationType::MIN_MAX};

    const txeo::DataTable<T> *_data_table{nullptr};

    std::vector<std::function<T(const T &)>> _funcs;
};

class DataTableNormError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo

#endif