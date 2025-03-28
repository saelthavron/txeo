#ifndef DATATABLE_H
#define DATATABLE_H
#pragma once

#include "txeo/Matrix.h"

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

namespace txeo {

template <typename T>
class DataTable {
  public:
    ~DataTable();

    DataTable(const DataTable &) = default;
    DataTable(DataTable &&) = default;
    DataTable &operator=(const DataTable &) = default;
    DataTable &operator=(DataTable &&) = default;

    DataTable(const txeo::Matrix<T> &data, std::vector<size_t> x_cols, std::vector<size_t> y_cols);

    DataTable(const txeo::Matrix<T> &data, std::vector<size_t> y_cols);

    DataTable(const txeo::Matrix<T> &data, std::vector<size_t> x_cols, std::vector<size_t> y_cols,
              size_t eval_percent);

    DataTable(const txeo::Matrix<T> &data, std::vector<size_t> y_cols, size_t eval_percent);

    DataTable(const txeo::Matrix<T> &data, std::vector<size_t> x_cols, std::vector<size_t> y_cols,
              size_t eval_percent, size_t eval_test);

    DataTable(const txeo::Matrix<T> &data, std::vector<size_t> y_cols, size_t eval_percent,
              size_t eval_test);

    const txeo::Matrix<T> &x_train() const { return _x_train; }

    const txeo::Matrix<T> &y_train() const { return _y_train; }

    std::optional<const txeo::Matrix<T> &> x_eval() const;
    std::optional<const txeo::Matrix<T> &> y_eval() const;
    std::optional<const txeo::Matrix<T> &> x_test() const;
    std::optional<const txeo::Matrix<T> &> y_test() const;

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