#ifndef DATATABLE_H
#define DATATABLE_H
#pragma once

#include "txeo/Matrix.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace txeo {

/**
 * @class DataTable
 * @brief A container for managing training, evaluation, and test data splits.
 *
 * @tparam T The data type stored in the Matrix (e.g., float, double)
 *
 * This class handles the division of a dataset into training, evaluation,
 * and test sets based on specified column indices and percentage splits.
 *
 */
template <typename T>
class DataTable {
  public:
    ~DataTable() = default;

    DataTable(const DataTable &) = default;
    DataTable(DataTable &&) = default;
    DataTable &operator=(const DataTable &) = default;
    DataTable &operator=(DataTable &&) = default;

    /**
     * @brief Construct a DataTable with specified feature/label columns
     *
     * @param data Input matrix containing all data points
     * @param x_cols Column indices for feature columns
     * @param y_cols Column indices for label columns
     *
     * @throws DataTableError
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<double> data = {{1, 2, 3},
     *                             {4, 5, 6},
     *                             {7, 8, 9}};
     * // Columns 0-1 as features, column 2 as label
     * DataTable<double> dt(data, {0,1}, {2});
     * assert(dt.x_train().cols() == 2);
     * assert(dt.y_train().cols() == 1);
     * @endcode
     */
    DataTable(txeo::Matrix<T> &&data, std::vector<size_t> x_cols, std::vector<size_t> y_cols);

    DataTable(const txeo::Matrix<T> &data, std::vector<size_t> x_cols, std::vector<size_t> y_cols)
        : DataTable{std::move(data.clone()), x_cols, y_cols} {};

    /**
     * @brief Construct a DataTable with specified label columns. All the remaining columns are
     * considered features
     *
     * @param data Input matrix containing all data points
     * @param y_cols Column indices for label columns
     *
     * @throws DataTableError
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<float> data = {{1.1f, 2.2f, 3.3f, 4.4f},
     *                            {5.5f, 6.6f, 7.7f, 8.8f}};
     * // Columns 3 as label, 0-2 as features
     * DataTable<float> dt(data, {3});
     * assert(dt.x_dim() == 3);
     * assert(dt.y_dim() == 1);
     * @endcode
     *
     */
    DataTable(txeo::Matrix<T> &&data, std::vector<size_t> y_cols);

    DataTable(const txeo::Matrix<T> &data, std::vector<size_t> y_cols)
        : DataTable{std::move(data.clone()), y_cols} {};

    /**
     * @brief Construct a DataTable with specified feature/label columns and evaluation split
     * percentage
     *
     * @param data Input matrix containing all data points
     * @param x_cols Column indices for feature columns
     * @param y_cols Column indices for label columns
     * @param eval_percent Percentage of data reserved for evaluation ]0,100[
     *
     * @throws DataTableError
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<double> data(100, 5); // 100 samples, 5 columns
     * // 70% train, 30% eval (columns 0-2 as X, 3-4 as Y)
     * DataTable<double> dt(data, {0,1,2}, {3,4}, 30);
     *
     * // Should have evaluation data
     * assert(dt.x_eval().has_value());
     * assert(dt.y_eval().has_value());
     * @endcode
     */
    DataTable(txeo::Matrix<T> &&data, std::vector<size_t> x_cols, std::vector<size_t> y_cols,
              size_t eval_percent);

    DataTable(const txeo::Matrix<T> &data, std::vector<size_t> x_cols, std::vector<size_t> y_cols,
              size_t eval_percent)
        : DataTable{std::move(data.clone()), x_cols, y_cols, eval_percent} {};

    /**
     * @brief Construct a DataTable with specified label columns and evaluation split
     * percentage. All the remaining columns are considered x features.
     *
     * @param data Input matrix containing all data points
     * @param y_cols Column indices for label columns
     * @param eval_percent Percentage of data reserved for evaluation ]0,100[
     *
     * @throws DataTableError
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<int> data = {{1, 2, 3, 4},
     *                          {5, 6, 7, 8}};
     * // 50% eval split, column 3 as label
     * DataTable<int> dt(data, {3}, 50);
     *
     * // Verify split sizes
     * assert(dt.row_size() == 1); // 50% of 2 rows = 1
     * assert(dt.x_eval()->rows() == 1);
     * @endcode
     */
    DataTable(txeo::Matrix<T> &&data, std::vector<size_t> y_cols, size_t eval_percent);

    DataTable(const txeo::Matrix<T> &data, std::vector<size_t> y_cols, size_t eval_percent)
        : DataTable{std::move(data.clone()), y_cols, eval_percent} {};

    /**
     * @brief Construct a DataTable with specified feature/label columns and evaluation/test split
     * percentages
     *
     * @param data Input matrix containing all data points
     * @param x_cols Column indices for feature columns
     * @param y_cols Column indices for label columns
     * @param eval_percent Percentage of data reserved for evaluation ]0,100[
     * @param eval_test Percentage of data reserved for test ]0,100[
     *
     * @throws DataTableError
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<double> data(1000, 10); // 1000 samples
     * // Split: 70% train, 20% eval, 10% test
     * DataTable<double> dt(data, {0,1,2,3}, {4,5}, 20, 10);
     *
     * assert(dt.x_train().rows() == 700);
     * assert(dt.x_eval()->rows() == 200);
     * assert(dt.x_test()->rows() == 100);
     * @endcode
     */
    DataTable(txeo::Matrix<T> &&data, std::vector<size_t> x_cols, std::vector<size_t> y_cols,
              size_t eval_percent, size_t eval_test);

    DataTable(const txeo::Matrix<T> &data, std::vector<size_t> x_cols, std::vector<size_t> y_cols,
              size_t eval_percent, size_t eval_test)
        : DataTable{std::move(data.clone()), x_cols, y_cols, eval_percent, eval_test} {};

    /**
     * @brief Construct a DataTable with specified label columns and evaluation/test split
     * percentages. All the remaining columns are considered x features.
     *
     * @param data Input matrix containing all data points
     * @param y_cols Column indices for label columns
     * @param eval_percent Percentage of data reserved for evaluation ]0,100[
     * @param eval_test Percentage of data reserved for test ]0,100[
     *
     * @throws DataTableError
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<float> data(500, 6);
     * // 60% train, 25% eval, 15% test
     * // Column 5 as label, rest as features
     * DataTable<float> dt(data, {5}, 25, 15);
     *
     * assert(dt.x_train().rows() == 300);  // 60% of 500
     * assert(dt.x_eval()->rows() == 125);  // 25% of 500
     * assert(dt.x_test()->rows() == 75);   // 15% of 500
     * @endcode
     */
    DataTable(txeo::Matrix<T> &&data, std::vector<size_t> y_cols, size_t eval_percent,
              size_t eval_test);

    DataTable(const txeo::Matrix<T> &data, std::vector<size_t> y_cols, size_t eval_percent,
              size_t eval_test)
        : DataTable<T>{std::move(data.clone()), y_cols, eval_percent, eval_test} {};

    /**
     * @brief Construct a DataTable with explicit training/evaluation/test splits.
     *
     * @param x_train Training input matrix (features)
     * @param y_train Training output matrix (labels)
     * @param x_eval Evaluation input matrix (features)
     * @param y_eval Evaluation output matrix (labels)
     * @param x_test Test input matrix (features)
     * @param y_test Test output matrix (labels)
     *
     * @example
     * @code
     * txeo::Matrix<double> X_train = {{1.0, 2.0}, {3.0, 4.0}};
     * txeo::Matrix<double> y_train = {{0.5}, {1.2}};
     * txeo::Matrix<double> X_eval = {{5.0, 6.0}};
     * txeo::Matrix<double> y_eval = {{2.1}};
     * txeo::Matrix<double> X_test = {{7.0, 8.0}};
     * txeo::Matrix<double> y_test = {{3.0}};
     *
     * // Create dataset with pre-split data
     * DataTable<double> dt_full(X_train, y_train, X_eval, y_eval, X_test, y_test);
     * @endcode
     */
    DataTable(txeo::Matrix<T> &&x_train, txeo::Matrix<T> &&y_train, txeo::Matrix<T> &&x_eval,
              txeo::Matrix<T> &&y_eval, txeo::Matrix<T> &&x_test, txeo::Matrix<T> &&y_test);

    DataTable(const txeo::Matrix<T> &x_train, const txeo::Matrix<T> &y_train,
              const txeo::Matrix<T> &x_eval, const txeo::Matrix<T> &y_eval,
              const txeo::Matrix<T> &x_test, const txeo::Matrix<T> &y_test)
        : DataTable<T>{std::move(x_train.clone()), std::move(y_train.clone()),
                       std::move(x_eval.clone()),  std::move(y_eval.clone()),
                       std::move(x_test.clone()),  std::move(y_test.clone())} {};

    /**
     * @brief Construct a DataTable with explicit training and evaluation splits.
     *
     * @param x_train Training input matrix (features)
     * @param y_train Training output matrix (labels)
     * @param x_eval Evaluation input matrix (features)
     * @param y_eval Evaluation output matrix (labels)
     *
     * @example
     * @code
     * txeo::Matrix<double> X_train = {{1.0, 2.0}, {3.0, 4.0}};
     * txeo::Matrix<double> y_train = {{0.5}, {1.2}};
     * txeo::Matrix<double> X_eval = {{5.0, 6.0}};
     * txeo::Matrix<double> y_eval = {{2.1}};
     *
     * // Create dataset with training and evaluation splits
     * DataTable<double> dt_eval(X_train, y_train, X_eval, y_eval);
     * @endcode
     */
    DataTable(txeo::Matrix<T> &&x_train, txeo::Matrix<T> &&y_train, txeo::Matrix<T> &&x_eval,
              txeo::Matrix<T> &&y_eval);

    DataTable(const txeo::Matrix<T> &x_train, const txeo::Matrix<T> &y_train,
              const txeo::Matrix<T> &x_eval, const txeo::Matrix<T> &y_eval)
        : DataTable(std::move(x_train.clone()), std::move(y_train.clone()),
                    std::move(x_eval.clone()), std::move(y_eval.clone())) {};

    /**
     * @brief Construct a DataTable with training data only.
     *
     * @param x_train Training input matrix (features)
     * @param y_train Training output matrix (labels)
     *
     * @example
     * @code
     * txeo::Matrix<double> X_train = {{1.0, 2.0}, {3.0, 4.0}};
     * txeo::Matrix<double> y_train = {{0.5}, {1.2}};
     *
     * // Create minimal dataset with only training data
     * DataTable<double> dt_simple(X_train, y_train);
     *
     * // Verify evaluation data is empty
     * assert(!dt_simple.x_eval().has_value());
     * @endcode
     */
    DataTable(txeo::Matrix<T> &&x_train, txeo::Matrix<T> &&y_train);

    DataTable(const txeo::Matrix<T> &x_train, const txeo::Matrix<T> &y_train)
        : DataTable{std::move(x_train.clone()), std::move(y_train.clone())} {};

    /**
     * @brief Returns training inputs matrix.
     *
     * @return Matrix of shape [rows, x_dim]
     */
    const txeo::Matrix<T> &x_train() const { return _x_train; }

    /**
     * @brief Returns training outputs matrix.
     *
     * @return Matrix of shape [rows, y_dim]
     */
    const txeo::Matrix<T> &y_train() const { return _y_train; }

    /**
     * @brief Returns a pointer to evaluation input matrix.
     *
     * @return Matrix of shape [rows, x_dim]. A nullptr means evaluation was not specified.
     */
    const txeo::Matrix<T> *x_eval() const;

    /**
     * @brief Returns a pointer to evaluation output  matrix.
     *
     * @return Matrix of shape [rows, y_dim]. A nullptr means evaluation data was not specified.
     */
    const txeo::Matrix<T> *y_eval() const;

    /**
     * @brief Returns a pointer to test input matrix.
     *
     * @return Matrix of shape [rows, x_dim]. A nullptr means test data was not specified.
     */
    const txeo::Matrix<T> *x_test() const;

    /**
     * @brief Returns a pointer to test output matrix.
     *
     * @return Matrix of shape [rows, y_dim]. A nullptr means test data was not specified.
     */
    const txeo::Matrix<T> *y_test() const;

    /**
     * @brief Returns the number of input columns.
     *
     * @return Feature dimension.
     */
    [[nodiscard]] size_t x_dim() const;

    /**
     * @brief Returns the number of output columns.
     *
     * @return Label dimension.
     */
    [[nodiscard]] size_t y_dim() const;

    /**
     * @brief Returns the number of training rows.
     *
     * @return Number of training samples.
     */
    [[nodiscard]] size_t row_size() const;

    [[nodiscard]] bool has_eval() const { return _has_eval; }

    [[nodiscard]] bool has_test() const { return _has_test; }

    DataTable<T> clone() const;

  private:
    DataTable() = default;

    txeo::Matrix<T> _x_train;
    txeo::Matrix<T> _y_train;
    txeo::Matrix<T> _x_eval;
    txeo::Matrix<T> _y_eval;
    txeo::Matrix<T> _x_test;
    txeo::Matrix<T> _y_test;
    bool _has_eval = false;
    bool _has_test = false;
};

class DataTableError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo

#endif