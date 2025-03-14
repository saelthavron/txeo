#ifndef TENSORFUNC_H
#define TENSORFUNC_H
#include <functional>
#pragma once

#include "txeo/Matrix.h"
#include "txeo/Tensor.h"

#include <cstddef>

namespace txeo {

/**
 * @class TensorFunc
 * @brief A utility class for common math functions on tensors.
 *
 * This class provides static methods for common tensor functions,
 * such as square.
 *
 * @tparam T The data type of the tensor elements (e.g., int, double).
 */

template <typename T>
class TensorFunc {
  public:
    TensorFunc(const TensorFunc &) = delete;
    TensorFunc(TensorFunc &&) = delete;
    TensorFunc &operator=(const TensorFunc &) = default;
    TensorFunc &operator=(TensorFunc &&) = delete;
    ~TensorFunc() = default;

    /**
     * @brief Returns the element-wise potentiation of a tensor
     *
     * @param tensor Tensor to be powered
     * @param exponent Exponent of the potentiation
     * @return txeo::Tensor<T> Result
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<float> a({3}, {2.0f, 3.0f, 4.0f});
     * auto b = TensorOp<float>::power_elem(a, 2.0f);  // Result: [4.0f, 9.0f, 16.0f]
     * @endcode
     */
    static txeo::Tensor<T> power_elem(const txeo::Tensor<T> &tensor, const T &exponent);

    /**
     * @brief Performs element-wise potentiation of the tensor (in-place)
     *
     * @param tensor Tensor to be modified
     * @param exponent Exponent of the potentiation
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<double> a({2}, {3.0, 4.0});
     * TensorOp<double>::power_elem_by(a, 3.0);  // a becomes [27.0, 64.0]
     * @endcode
     */
    static txeo::Tensor<T> power_elem_by(txeo::Tensor<T> &tensor, const T &exponent);

    /**
     * @brief Computes the element-wise square of a tensor.
     *
     * @param tensor The input tensor.
     * @return A new tensor containing the squared values.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({3}, {1, 2, 3});
     * auto result = TensorOp<int>::square(tensor);
     * // result = [1, 4, 9]
     * @endcode
     */
    static txeo::Tensor<T> square(const txeo::Tensor<T> &tensor);

    /**
     * @brief Computes the element-wise square of a tensor in-place.
     *
     * @param tensor The input tensor to be modified.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({3}, {1, 2, 3});
     * TensorOp<int>::square_by(tensor);
     * // tensor = [1, 4, 9]
     * @endcode
     */
    static txeo::Tensor<T> &square_by(txeo::Tensor<T> &tensor);

    /**
     * @brief Computes the element-wise square root of a tensor.
     *
     * @param tensor The input tensor.
     * @return A new tensor containing the square root values.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<double> tensor({3}, {1.0, 4.0, 9.0});
     * auto result = TensorOp<double>::sqrt(tensor);
     * // result = [1.0, 2.0, 3.0]
     * @endcode
     */
    static txeo::Tensor<T> sqrt(const txeo::Tensor<T> &tensor);

    /**
     * @brief Computes the element-wise square root of a tensor in-place.
     *
     * @param tensor The input tensor to be modified.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<double> tensor({3}, {1.0, 4.0, 9.0});
     * TensorOp<double>::sqrt_by(tensor);
     * // tensor = [1.0, 2.0, 3.0]
     * @endcode
     */
    static txeo::Tensor<T> &sqrt_by(txeo::Tensor<T> &tensor);

    /**
     * @brief Computes the element-wise absolute value of a tensor.
     *
     * @param tensor The input tensor to be modified.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({3}, {-1, 2, -3});
     * TensorOp<int>::abs_by(tensor);
     * // tensor = [1, 2, 3]
     * @endcode
     */
    static txeo::Tensor<T> abs(const txeo::Tensor<T> &tensor);

    /**
     * @brief Computes the element-wise absolute value of a tensor in-place.
     *
     * @param tensor The input tensor to be modified.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({3}, {-1, 2, -3});
     * TensorOp<int>::abs_by(tensor);
     * // tensor = [1, 2, 3]
     * @endcode
     */
    static txeo::Tensor<T> &abs_by(txeo::Tensor<T> &tensor);

    /**
     * @brief Permutes the axes of a tensor.
     *
     *
     * @param tensor The input tensor.
     * @param axes The new order of the tensor axes. Must be a valid permutation of the tensor's
     * dimensions.
     * @return A new tensor with the axes permuted.
     *
     * @throws std::invalid_argument If the axes are invalid (e.g., size mismatch or out of range).
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({2, 3, 4}, {
     *     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
     *     13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
     * });
     *  // The new postion of axis 1 is zero, of axis 2 is one and of axis zero is 2
     * auto result = TensorFunc<int>::permute(tensor, {1, 2, 0});
     * // result shape: (3, 4, 2)
     * @endcode
     */
    static txeo::Tensor<T> permute(const txeo::Tensor<T> &tensor, const std::vector<size_t> &axes);

    /**
     * @brief Permutes the axes of a tensor in-place.
     *
     * @param tensor The input tensor to be modified.
     * @param axes The new order of axes. Must be a valid permutation of the tensor's dimensions.
     * @return A reference to the modified tensor.
     *
     * @throws std::invalid_argument If the axes are invalid (e.g., size mismatch or out of range).
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({2, 3, 4}, {
     *     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
     *     13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
     * });
     *  // The new postion of axis 1 is zero, of axis 2 is one and of axis zero is 2
     * TensorFunc<int>::permute_by(tensor, {1, 2, 0});
     * // tensor shape after permutation: (3, 4, 2)
     * @endcode
     */
    static txeo::Tensor<T> &permute_by(txeo::Tensor<T> &tensor, const std::vector<size_t> &axes);

    static txeo::Tensor<T> &min_max_normalize_by(txeo::Tensor<T> &tensor, size_t axis);
    static txeo::Tensor<T> min_max_normalize(const txeo::Tensor<T> &tensor, size_t axis);

    /**
     * @brief Transposes a matrix.
     *
     * @param matrix The input matrix.
     * @return A new matrix that is the transpose of the input matrix.
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<int> matrix(2, 3, {1, 2, 3, 4, 5, 6});
     * auto result = TensorFunc<int>::transpose(matrix);
     * // result shape: (3, 2)
     * @endcode
     */
    static txeo::Matrix<T> transpose(const txeo::Matrix<T> &matrix);

    /**
     * @brief Transposes a matrix in-place.
     *
     * @param matrix The input matrix to be modified.
     * @return A reference to the modified matrix.
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<int> matrix(2, 3, {1, 2, 3, 4, 5, 6});
     * TensorFunc<int>::transpose_by(matrix);
     * // matrix shape after transpose: (3, 2)
     * @endcode
     */
    static txeo::Matrix<T> &transpose_by(txeo::Matrix<T> &matrix);

  private:
    TensorFunc() = default;

    static void
    axis_func(txeo::Tensor<T> &tensor, size_t axis,
              std::function<void(const std::vector<T> &, const std::vector<T *> &)> func);
    static void min_max_normalize(const std::vector<T> &values, const std::vector<T *> &adresses);
    static void z_score_normalize(const std::vector<T> &values, const std::vector<T *> &adresses);
};

/**
 * @brief Exceptions concerning @ref txeo::TensorOp
 *
 */
class TensorFuncError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo

#endif