#ifndef TENSORAGG_H
#define TENSORAGG_H
#include <functional>
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
                                      const std::vector<size_t> &axes);

    static txeo::Tensor<T> reduce_prod(const txeo::Tensor<T> &tensor,
                                       const std::vector<size_t> &axes);

    static txeo::Tensor<T> reduce_mean(const txeo::Tensor<T> &tensor,
                                       const std::vector<size_t> &axes);

    static txeo::Tensor<T> reduce_max(const txeo::Tensor<T> &tensor,
                                      const std::vector<size_t> &axes);

    static txeo::Tensor<T> reduce_min(const txeo::Tensor<T> &tensor,
                                      const std::vector<size_t> &axes);

    static txeo::Tensor<T> reduce_euclidean_norm(const txeo::Tensor<T> &tensor,
                                                 const std::vector<size_t> &axes);

    static txeo::Tensor<T> reduce_maximum_norm(const txeo::Tensor<T> &tensor, size_t axis);

    static txeo::Tensor<T> reduce_variance(const txeo::Tensor<T> &tensor, size_t axis);

    static txeo::Tensor<T> reduce_standard_deviation(const txeo::Tensor<T> &tensor, size_t axis);

    static txeo::Tensor<T> reduce_median(const txeo::Tensor<T> &tensor, size_t axis);

    static txeo::Tensor<T> reduce_geometric_mean(const txeo::Tensor<T> &tensor, size_t axis);

    static txeo::Tensor<T> reduce_all(const txeo::Tensor<T> &tensor,
                                      const std::vector<size_t> &axes)
      requires(std::convertible_to<T, bool>);

    static txeo::Tensor<T> reduce_any(const txeo::Tensor<T> &tensor,
                                      const std::vector<size_t> &axes)
      requires(std::convertible_to<T, bool>);

    static txeo::Tensor<T> cumulative_sum(const txeo::Tensor<T> &tensor, size_t axis);

    static txeo::Tensor<T> cumulative_prod(const txeo::Tensor<T> &tensor, size_t axis);

    static txeo::Tensor<size_t> arg_max(const txeo::Tensor<T> &tensor, size_t axis);

    static txeo::Tensor<size_t> arg_min(const txeo::Tensor<T> &tensor, size_t axis);

    static txeo::Tensor<size_t> count_non_zero(const txeo::Tensor<T> &tensor, size_t axis);

    static T sum_all(const txeo::Tensor<T> &tensor);

  private:
    static void verify_parameters(const txeo::Tensor<T> &tensor, const std::vector<size_t> &axes);

    static txeo::Tensor<T> accumulate(const txeo::Tensor<T> &tensor, size_t axis,
                                      std::function<T(std::vector<T> &)>);

    static txeo::Tensor<size_t> count(const txeo::Tensor<T> &tensor, size_t axis,
                                      std::function<size_t(std::vector<T> &)>);

    static T median(std::vector<T> &values);
    static T geometric_mean(std::vector<T> &values);
    static T variance(std::vector<T> &values);
    static T maximum_norm(std::vector<T> &values);
    static size_t count_non_zero(std::vector<T> &values);
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