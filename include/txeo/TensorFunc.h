#ifndef TENSORFUNC_H
#define TENSORFUNC_H
#pragma once

#include "txeo/Matrix.h"
#include "txeo/Tensor.h"
#include "txeo/types.h"

#include <cstddef>
#include <functional>

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
     * @throws std::TensorFuncError
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

    /**
     * @brief Normalizes the input tensor along a specified axis in-place
     *
     * @param[in,out] tensor The tensor to be normalized (modified in-place)
     * @param axis The dimension along which to apply normalization
     * @param type The normalization method from NormalizationType enum:
     *             - MIN_MAX: Scales values to [0, 1] range
     *             - Z_SCORE: Standardizes to mean=0, std=1
     * @return Reference to the modified input tensor
     *
     * @throws std::TensorFuncError
     *
     * **Example Usage:**
     * @code
     * // Example: Z-score normalization along columns (axis=1)
     * txeo::Tensor<double> matrix({{1.0, 2.0}, {3.0, 4.0}}); // 2x2 matrix
     * TensorFunc<double>::normalize_by(matrix, 1, txeo::NormalizationType::Z_SCORE);
     * // Column 0 becomes [-1.0, 1.0], Column 1 becomes [-1.0, 1.0]
     * @endcode
     */
    static txeo::Tensor<T> &normalize_by(txeo::Tensor<T> &tensor, size_t axis,
                                         txeo::NormalizationType type);

    /**
     * @brief Creates a normalized copy of the input tensor along a specified axis
     *
     * @param tensor The input tensor to normalize
     * @param axis The dimension along which to apply normalization
     * @param type The normalization method (MIN_MAX or Z_SCORE)
     * @return New tensor containing normalized values
     *
     * @throws std::TensorFuncError
     *
     * **Example Usage:**
     * @code
     * // Example: Min-max normalization of a vector
     * txeo::Tensor<float> vec({2.0f, 4.0f, 6.0f}); // min=2, max=6
     * auto normalized = TensorFunc<float>::normalize(vec, 0, txeo::NormalizationType::MIN_MAX);
     * // normalized contains [0.0, 0.5, 1.0]
     * @endcode
     */
    static txeo::Tensor<T> normalize(const txeo::Tensor<T> &tensor, size_t axis,
                                     txeo::NormalizationType type);

    /**
     * @brief Normalizes the entire tensor in-place (global normalization)
     *
     * @param[in,out] tensor The tensor to be normalized (modified in-place)
     * @param type The normalization method (MIN_MAX or Z_SCORE)
     * @return Reference to the modified input tensor
     *
     * @throws std::TensorFuncError
     *
     * **Example Usage:**
     * @code
     * // Example: Global min-max normalization
     * txeo::Tensor<double> data({{10.0, 20.0}, {30.0, 40.0}}); // min=10, max=40
     * TensorFunc<double>::normalize_by(data, txeo::NormalizationType::MIN_MAX);
     * // data now contains [[0.0, 0.333], [0.666, 1.0]]
     * @endcode
     */
    static txeo::Tensor<T> &normalize_by(txeo::Tensor<T> &tensor, txeo::NormalizationType type);

    /**
     * @brief Creates a normalized copy of the entire tensor (global normalization)
     *
     * @param tensor The input tensor to normalize
     * @param type The normalization method (MIN_MAX or Z_SCORE)
     * @return New tensor containing normalized values
     *
     * @throws std::TensorFuncError
     *
     * **Example Usage:**
     * @code
     * // Example: Global Z-score normalization
     * txeo::Tensor<float> cube({{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}});
     * auto result = TensorFunc<float>::normalize(cube, txeo::NormalizationType::Z_SCORE);
     * // result contains values with μ=4.5 and σ=2.449
     * @endcode
     */
    static txeo::Tensor<T> normalize(const txeo::Tensor<T> &tensor, txeo::NormalizationType type);

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

    static txeo::Matrix<T> get_gram_matrix(const txeo::Matrix<T> &matrix);

  private:
    TensorFunc() = default;

    static void
    axis_func(txeo::Tensor<T> &tensor, size_t axis,
              std::function<void(const std::vector<T> &, const std::vector<T *> &)> func);
    static void min_max_normalize(const std::vector<T> &values, const std::vector<T *> &adresses);
    static void z_score_normalize(const std::vector<T> &values, const std::vector<T *> &adresses);

    static void min_max_normalize(txeo::Tensor<T> &tensor);
    static void z_score_normalize(txeo::Tensor<T> &tensor);
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