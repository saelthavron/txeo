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

    /**
     * @brief Computes the sum of tensor elements along the specified axes.
     *
     * @param tensor The input tensor.
     * @param axes The axes along which to compute the sum.
     * @return A new tensor containing the sum along the specified axes.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});
     * auto result = TensorAgg<int>::reduce_sum(tensor, {1});
     * // result = [6, 15]  (sum along axis 1)
     * @endcode
     */
    static txeo::Tensor<T> reduce_sum(const txeo::Tensor<T> &tensor,
                                      const std::vector<size_t> &axes);

    /**
     * @brief Computes the product of tensor elements along the specified axes.
     *
     * @param tensor The input tensor.
     * @param axes The axes along which to compute the product.
     * @return A new tensor containing the product along the specified axes.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});
     * auto result = TensorAgg<int>::reduce_prod(tensor, {1});
     * // result = [6, 120]  (product along axis 1)
     * @endcode
     */
    static txeo::Tensor<T> reduce_prod(const txeo::Tensor<T> &tensor,
                                       const std::vector<size_t> &axes);

    /**
     * @brief Computes the mean of tensor elements along the specified axes.
     *
     * @param tensor The input tensor.
     * @param axes The axes along which to compute the mean.
     * @return A new tensor containing the mean along the specified axes.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<double> tensor({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
     * auto result = TensorAgg<double>::reduce_mean(tensor, {1});
     * // result = [2.0, 5.0]  (mean along axis 1)
     * @endcode
     */
    static txeo::Tensor<T> reduce_mean(const txeo::Tensor<T> &tensor,
                                       const std::vector<size_t> &axes);

    /**
     * @brief Computes the maximum of tensor elements along the specified axes.
     *
     * @param tensor The input tensor.
     * @param axes The axes along which to compute the maximum.
     * @return A new tensor containing the maximum along the specified axes.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});
     * auto result = TensorAgg<int>::reduce_max(tensor, {1});
     * // result = [3, 6]  (max along axis 1)
     * @endcode
     */
    static txeo::Tensor<T> reduce_max(const txeo::Tensor<T> &tensor,
                                      const std::vector<size_t> &axes);

    /**
     * @brief Computes the minimum of tensor elements along the specified axes.
     *
     * @param tensor The input tensor.
     * @param axes The axes along which to compute the minimum.
     * @return A new tensor containing the minimum along the specified axes.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});
     * auto result = TensorAgg<int>::reduce_min(tensor, {1});
     * // result = [1, 4]  (min along axis 1)
     * @endcode
     */
    static txeo::Tensor<T> reduce_min(const txeo::Tensor<T> &tensor,
                                      const std::vector<size_t> &axes);

    /**
     * @brief Computes the Euclidean norm of tensor elements along the specified axes.
     *
     * @param tensor The input tensor.
     * @param axes The axes along which to compute the Euclidean norm.
     * @return A new tensor containing the Euclidean norm along the specified axes.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<double> tensor({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
     * auto result = TensorAgg<double>::reduce_euclidean_norm(tensor, {1});
     * // result = [sqrt(1^2 + 2^2 + 3^2), sqrt(4^2 + 5^2 + 6^2)] = [3.74166, 8.77496]
     * @endcode
     */
    static txeo::Tensor<T> reduce_euclidean_norm(const txeo::Tensor<T> &tensor,
                                                 const std::vector<size_t> &axes);

    /**
     * @brief Computes the maximum norm of tensor elements along the specified axis.
     *
     * @param tensor The input tensor.
     * @param axis The axis along which to compute the maximum norm.
     * @return A new tensor containing the maximum norm along the specified axis.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({2, 3}, {1, -2, 3, -4, 5, -6});
     * auto result = TensorAgg<int>::reduce_maximum_norm(tensor, 1);
     * // result = [3, 6]  (max norm along axis 1)
     * @endcode
     */
    static txeo::Tensor<T> reduce_maximum_norm(const txeo::Tensor<T> &tensor, size_t axis);

    /**
     * @brief Computes the variance of tensor elements along the specified axis.
     *
     * @param tensor The input tensor.
     * @param axis The axis along which to compute the variance.
     * @return A new tensor containing the variance along the specified axis.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<double> tensor({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
     * auto result = TensorAgg<double>::reduce_variance(tensor, 1);
     * // result = [1.0, 1.0]  (variance along axis 1)
     * @endcode
     */
    static txeo::Tensor<T> reduce_variance(const txeo::Tensor<T> &tensor, size_t axis);

    /**
     * @brief Computes the standard deviation of tensor elements along the specified axis.
     *
     * @param tensor The input tensor.
     * @param axis The axis along which to compute the standard deviation.
     * @return A new tensor containing the standard deviation along the specified axis.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<double> tensor({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
     * auto result = TensorAgg<double>::reduce_standard_deviation(tensor, 1);
     * // result = [1.0, 1.0]  (standard deviation along axis 1)
     * @endcode
     */
    static txeo::Tensor<T> reduce_standard_deviation(const txeo::Tensor<T> &tensor, size_t axis);

    /**
     * @brief Computes the median of tensor elements along the specified axis.
     *
     * @param tensor The input tensor.
     * @param axis The axis along which to compute the median.
     * @return A new tensor containing the median along the specified axis.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});
     * auto result = TensorAgg<int>::reduce_median(tensor, 1);
     * // result = [2, 5]  (median along axis 1)
     * @endcode
     */
    static txeo::Tensor<T> reduce_median(const txeo::Tensor<T> &tensor, size_t axis);

    /**
     * @brief Computes the geometric mean of tensor elements along the specified axis.
     *
     * @param tensor The input tensor.
     * @param axis The axis along which to compute the geometric mean.
     * @return A new tensor containing the geometric mean along the specified axis.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<double> tensor({2, 3}, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
     * auto result = TensorAgg<double>::reduce_geometric_mean(tensor, 1);
     * // result = [1.81712, 4.93242]  (geometric mean along axis 1)
     * @endcode
     */
    static txeo::Tensor<T> reduce_geometric_mean(const txeo::Tensor<T> &tensor, size_t axis);

    /**
     * @brief Computes the logical AND (`all`) of tensor elements along the specified axes.
     *
     * @param tensor The input tensor.
     * @param axes The axes along which to compute the logical AND.
     * @return A new tensor containing the logical AND along the specified axes.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<bool> tensor({2, 3}, {true, false, true, true, true, false});
     * auto result = TensorAgg<bool>::reduce_all(tensor, {1});
     * // result = [false, false]  (logical AND along axis 1)
     * @endcode
     */
    static txeo::Tensor<T> reduce_all(const txeo::Tensor<T> &tensor,
                                      const std::vector<size_t> &axes)
      requires(std::convertible_to<T, bool>);

    /**
     * @brief Computes the logical OR (`any`) of tensor elements along the specified axes.
     *
     * @param tensor The input tensor.
     * @param axes The axes along which to compute the logical OR.
     * @return A new tensor containing the logical OR along the specified axes.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<bool> tensor({2, 3}, {true, false, true, true, true, false});
     * auto result = TensorAgg<bool>::reduce_any(tensor, {1});
     * // result = [true, true]  (logical OR along axis 1)
     * @endcode
     */
    static txeo::Tensor<T> reduce_any(const txeo::Tensor<T> &tensor,
                                      const std::vector<size_t> &axes)
      requires(std::convertible_to<T, bool>);

    /**
     * @brief Computes the cumulative sum of tensor elements along the specified axis.
     *
     * @param tensor The input tensor.
     * @param axis The axis along which to compute the cumulative sum.
     * @return A new tensor containing the cumulative sum along the specified axis.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});
     * auto result = TensorAgg<int>::cumulative_sum(tensor, 1);
     * // result = [ [1, 3, 6], [4, 9, 15] ]  (cumulative sum along axis 1)
     * @endcode
     */
    static txeo::Tensor<T> cumulative_sum(const txeo::Tensor<T> &tensor, size_t axis);

    /**
     * @brief Computes the cumulative product of tensor elements along the specified axis.
     *
     * @param tensor The input tensor.
     * @param axis The axis along which to compute the cumulative product.
     * @return A new tensor containing the cumulative product along the specified axis.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});
     * auto result = TensorAgg<int>::cumulative_prod(tensor, 1);
     * // result = [ [1, 2, 6], [4, 20, 120] ]  (cumulative product along axis 1)
     * @endcode
     */
    static txeo::Tensor<T> cumulative_prod(const txeo::Tensor<T> &tensor, size_t axis);

    /**
     * @brief Finds the indices of the maximum values along the specified axis.
     *
     * @param tensor The input tensor.
     * @param axis The axis along which to find the maximum indices.
     * @return A new tensor containing the indices of the maximum values along the specified axis.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});
     * auto result = TensorAgg<int>::arg_max(tensor, 1);
     * // result = [2, 2]  (indices of max values along axis 1)
     * @endcode
     */
    static txeo::Tensor<size_t> arg_max(const txeo::Tensor<T> &tensor, size_t axis);

    /**
     * @brief Finds the indices of the minimum values along the specified axis.
     *
     * @param tensor The input tensor.
     * @param axis The axis along which to find the minimum indices.
     * @return A new tensor containing the indices of the minimum values along the specified axis.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});
     * auto result = TensorAgg<int>::arg_min(tensor, 1);
     * // result = [0, 0]  (indices of min values along axis 1)
     * @endcode
     */
    static txeo::Tensor<size_t> arg_min(const txeo::Tensor<T> &tensor, size_t axis);

    /**
     * @brief Counts the number of non-zero elements along the specified axis.
     *
     * @param tensor The input tensor.
     * @param axis The axis along which to count non-zero elements.
     * @return A new tensor containing the count of non-zero elements along the specified axis.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({2, 3}, {1, 0, 3, 0, 5, 0});
     * auto result = TensorAgg<int>::count_non_zero(tensor, 1);
     * // result = [2, 1]  (count of non-zero elements along axis 1)
     * @endcode
     */
    static txeo::Tensor<size_t> count_non_zero(const txeo::Tensor<T> &tensor, size_t axis);

    /**
     * @brief Computes the sum of all elements in the tensor.
     *
     * @param tensor The input tensor.
     * @return The sum of all elements in the tensor.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});
     * auto result = TensorAgg<int>::sum_all(tensor);
     * // result = 21  (sum of all elements)
     * @endcode
     */
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