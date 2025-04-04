#ifndef DATATABLENORM_H
#define DATATABLENORM_H
#pragma once

#include "txeo/DataTable.h"
#include "txeo/types.h"

namespace txeo {

template <typename T>
class DataTableNorm {
  public:
    DataTableNorm(const DataTableNorm &) = delete;
    DataTableNorm(DataTableNorm &&) = default;
    DataTableNorm &operator=(const DataTableNorm &) = delete;
    DataTableNorm &operator=(DataTableNorm &&) = default;
    ~DataTableNorm() = default;

    DataTableNorm(const txeo::DataTable<T> &data,
                  txeo::NormalizationType type = txeo::NormalizationType::MIN_MAX);

    void normalize_features(const txeo::DataTable<T> &data);

    [[nodiscard]] txeo::NormalizationType type() const { return _type; }

  private:
    DataTableNorm() = default;

    bool _is_normalized{false};
    txeo::NormalizationType _type{};

    std::vector<std::function<T(const T &)>> _funcs;
};

} // namespace txeo

#endif