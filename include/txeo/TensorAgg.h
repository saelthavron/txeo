#ifndef TENSORAGG_H
#define TENSORAGG_H
#include <initializer_list>
#pragma once

#include "txeo/Tensor.h"

#include <cstddef>

namespace txeo {

template <typename T>
class TensorAgg {

  public:
    TensorAgg() = delete;
    TensorAgg(const TensorAgg &) = delete;
    TensorAgg(TensorAgg &&) = delete;
    TensorAgg &operator=(const TensorAgg &) = delete;
    TensorAgg &operator=(TensorAgg &&) = delete;
    ~TensorAgg() = default;

    static txeo::Tensor<T> reduce_sum(const txeo::Tensor<T> &tensor,
                                      std::initializer_list<size_t> axes);
};

/**
 * @brief Exceptions concerning @ref txeo::TensorAgg
 *
 */
class TensorAggError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo

#endif