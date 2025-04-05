#ifndef DATATABLENORM_H
#define DATATABLENORM_H
#pragma once

#include "txeo/DataTable.h"
#include "txeo/types.h"

namespace txeo {

template <typename T>
class DataTableNorm {
  public:
    DataTableNorm() = default;

    DataTableNorm(const DataTableNorm &) = delete;
    DataTableNorm(DataTableNorm &&) = default;
    DataTableNorm &operator=(const DataTableNorm &) = delete;
    DataTableNorm &operator=(DataTableNorm &&) = default;
    ~DataTableNorm() = default;

    DataTableNorm(const txeo::DataTable<T> &data,
                  txeo::NormalizationType type = txeo::NormalizationType::MIN_MAX);

    const txeo::DataTable<T> &data_table() const { return *_data_table; }

    void set_data_table(const txeo::DataTable<T> &data);

    [[nodiscard]] txeo::NormalizationType type() const { return _type; }

    // txeo::Matrix<T> normalize(txeo::Matrix<T> &&x) const;

    // txeo::Matrix<T> normalize(const txeo::Matrix<T> &x) const {
    //   return this->normalize(std::move(x.clone()));
    // };

    template <typename U>
      requires std::is_convertible_v<U, txeo::Matrix<T>>
    [[nodiscard]] txeo::Matrix<T> normalize(U &&x) const;

    txeo::Matrix<T> x_train_normalized();

    txeo::Matrix<T> x_eval_normalized();

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