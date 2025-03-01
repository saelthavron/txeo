#ifndef TENSORAGG_H
#define TENSORAGG_H
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

    static txeo::Tensor<T> reduce_sum(const txeo::Tensor<T> &tensor, std::vector<size_t> axes);

    static txeo::Tensor<T> reduce_mean(const txeo::Tensor<T> &tensor, std::vector<size_t> axes);

    static txeo::Tensor<T> reduce_max(const txeo::Tensor<T> &tensor, std::vector<size_t> axes);

    static txeo::Tensor<T> reduce_min(const txeo::Tensor<T> &tensor, std::vector<size_t> axes);

    static txeo::Tensor<size_t> arg_max(const txeo::Tensor<T> &tensor, size_t axis);

    static txeo::Tensor<size_t> arg_min(const txeo::Tensor<T> &tensor, size_t axis);

    static txeo::Tensor<T> abs(const txeo::Tensor<T> &tensor);

    static T variance(const txeo::Tensor<T> &tensor);

    static T standard_deviation(const txeo::Tensor<T> &tensor);
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