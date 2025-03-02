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

    static txeo::Tensor<T> reduce_prod(const txeo::Tensor<T> &tensor, std::vector<size_t> axes);

    static txeo::Tensor<T> reduce_mean(const txeo::Tensor<T> &tensor, std::vector<size_t> axes);

    static txeo::Tensor<T> reduce_max(const txeo::Tensor<T> &tensor, std::vector<size_t> axes);

    static txeo::Tensor<T> reduce_min(const txeo::Tensor<T> &tensor, std::vector<size_t> axes);

    static txeo::Tensor<T> reduce_euclidean_norm(const txeo::Tensor<T> &tensor,
                                                 std::vector<size_t> axes);

    static txeo::Tensor<T> reduce_all(const txeo::Tensor<T> &tensor, std::vector<size_t> axes)
      requires(std::convertible_to<T, bool>);

    static txeo::Tensor<T> reduce_any(const txeo::Tensor<T> &tensor, std::vector<size_t> axes)
      requires(std::convertible_to<T, bool>);

    static txeo::Tensor<T> cumulative_sum(const txeo::Tensor<T> &tensor, size_t axis);

    static txeo::Tensor<T> cumulative_prod(const txeo::Tensor<T> &tensor, size_t axis);

    static txeo::Tensor<size_t> arg_max(const txeo::Tensor<T> &tensor, size_t axis);

    static txeo::Tensor<size_t> arg_min(const txeo::Tensor<T> &tensor, size_t axis);

    static txeo::Tensor<T> abs(const txeo::Tensor<T> &tensor);

    static T variance(const txeo::Tensor<T> &tensor);

    static T standard_deviation(const txeo::Tensor<T> &tensor);

    static T sum_all(const txeo::Tensor<T> &tensor);

  private:
    static void verify_parameters(const txeo::Tensor<T> &tensor, std::vector<size_t> axes);
};

/**
 * @brief Exceptions concerning @ref txeo::TensorAgg
 *
 */
class TensorAggError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// Median
// Mode
// Geometric mean
// Weighted mean
// Skewness and kurtosis
// Count non-zero elements
// Normalize (Min-Max, Z-Score)
// Summation over the entire tensor
// Other norms (L1, L∞, etc.)

} // namespace txeo

#endif