#ifndef TENSOROP_H
#define TENSOROP_H
#pragma once

#include "txeo/Matrix.h"
#include "txeo/Tensor.h"
#include "txeo/Vector.h"

namespace txeo {

/**
 * @class TensorOp
 * @brief A utility class for performing operations on tensors and vectors.
 *
 * This class provides static methods for common tensor and vector operations,
 * such as dot product.
 *
 * @tparam T The data type of the tensor/vector elements (e.g., int, double).
 */
template <typename T>
class TensorOp {
  public:
    TensorOp(const TensorOp &) = delete;
    TensorOp(TensorOp &&) = delete;
    TensorOp &operator=(const TensorOp &) = delete;
    TensorOp &operator=(TensorOp &&) = delete;
    ~TensorOp() = default;

    /**
     * @brief Returns the sum of two tensors
     *
     * @param left Left operand
     * @param right Right operand
     * @return txeo::Tensor<T> Result
     *
     * @exception TensorOpError Thrown if shapes mismatch
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<float> a({2,2}, {1.0f, 2.0f, 3.0f, 4.0f});
     * txeo::Tensor<float> b({2,2}, {5.0f, 6.0f, 7.0f, 8.0f});
     * auto c = TensorOp<float>::sum(a, b);  // Result: [[6,8],[10,12]]
     * @endcode
     */
    static txeo::Tensor<T> sum(const txeo::Tensor<T> &left, const txeo::Tensor<T> &right);

    /**
     * @brief Sums the left operand with the right operand (in-place)
     *
     * @param left Operand to be modified
     * @param right Operand to add
     *
     * @exception TensorOpError Thrown if shapes mismatch
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<double> a({3}, {1.0, 2.0, 3.0});
     * txeo::Tensor<double> b({3}, {4.0, 5.0, 6.0});
     * TensorOp<double>::sum_by(a, b);  // a becomes [5.0, 7.0, 9.0]
     * @endcode
     */
    static txeo::Tensor<T> &sum_by(txeo::Tensor<T> &left, const txeo::Tensor<T> &right);

    /**
     * @brief Element-wise sum of tensor and scalar (out-of-place)
     *
     * @param left Input tensor (shape NxMx...)
     * @param right Scalar value to add
     * @return New tensor with same shape as input
     *
     * **Example Usage:**
     *@code
     * txeo::Tensor<float> A({3}, {1.0f, 2.0f, 3.0f});
     * auto B = TensorOp<float>::sum(A, 5.0f);
     * // B contains [6.0, 7.0, 8.0], shape [3]
     *@endcode
     */
    static txeo::Tensor<T> sum(const txeo::Tensor<T> &left, const T &right);

    /**
     * @brief In-place element-wise addition of scalar to tensor
     * @param left Tensor to modify
     * @param right Scalar to add
     *
     * **Example Usage:**
     *@code
     * txeo::Tensor<int> t({2, 2}, {1, 2, 3, 4});
     * TensorOp<int>::sum_by(t, 5);
     * // t now contains [6, 7, 8, 9] with shape [2,2]
     * @endcode
     */
    static txeo::Tensor<T> &sum_by(txeo::Tensor<T> &left, const T &right);

    /**
     * @brief Returns the subtraction of two tensors
     *
     * @param left Left operand
     * @param right Right operand
     * @return txeo::Tensor<T> Result
     *
     * @exception TensorOpError Thrown if shapes mismatch
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> a({2}, {10, 20});
     * txeo::Tensor<int> b({2}, {3, 5});
     * auto c = TensorOp<int>::subtract(a, b);  // Result: [7, 15]
     * @endcode
     */
    static txeo::Tensor<T> subtract(const txeo::Tensor<T> &left, const txeo::Tensor<T> &right);

    /**
     * @brief Subtracts the left operand by the right operand (in-place)
     *
     * @param left Operand to be modified
     * @param right Operand to subtract
     *
     * @exception TensorOpError Thrown if shapes mismatch
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<float> a({2,2}, {5.0f, 10.0f, 15.0f, 20.0f});
     * txeo::Tensor<float> b({2,2}, {1.0f, 2.0f, 3.0f, 4.0f});
     * TensorOp<float>::subtract_by(a, b);  // a becomes [[4,8],[12,16]]
     * @endcode
     */
    static txeo::Tensor<T> &subtract_by(txeo::Tensor<T> &left, const txeo::Tensor<T> &right);

    /**
     * @brief Element-wise subtraction of scalar from tensor (out-of-place)
     * @param left Input tensor (shape NxMx...)
     * @param right Scalar value to subtract
     * @return New tensor with same shape as input
     *
     * **Example Usage:**
     *@code
     * txeo::Tensor<int> A({2, 2}, {10, 20, 30, 40});
     * auto B = TensorOp<int>::subtract(A, 5);
     * // B contains [5, 15, 25, 35], shape [2,2]
     *@endcode
     */
    static txeo::Tensor<T> subtract(const txeo::Tensor<T> &left, const T &right);

    /**
     * @brief In-place element-wise subtraction of scalar from tensor
     *
     * @param left Tensor to modify
     * @param right Scalar to subtract
     *
     * **Example Usage:**
     *@code
     * txeo::Tensor<float> t({3}, {5.5f, 6.6f, 7.7f});
     * TensorOp<float>::subtract_by(t, 2.2f);
     * // t now contains [3.3, 4.4, 5.5]
     *@endcode
     */
    static txeo::Tensor<T> &subtract_by(txeo::Tensor<T> &left, const T &right);

    /**
     * @brief Element-wise subtraction of tensor from scalar (out-of-place)
     *
     * @param left Scalar value
     * @param right Tensor to subtract
     * @return New tensor where each element = left - right[i]
     *
     * **Example Usage:**
     *@code
     * txeo::Tensor<double> t({2}, {1.5, 3.0});
     * auto result = TensorOp<double>::subtract(10.0, t);
     * // result contains [8.5, 7.0] with shape [2]
     *@endcode
     */
    static txeo::Tensor<T> subtract(const T &left, const txeo::Tensor<T> &right);

    /**
     * @brief In-place element-wise subtraction of tensor from scalar
     *
     * @param left Scalar value (minuend)
     * @param right Tensor to modify (subtrahend)
     *
     * **Example Usage:**
     *@code
     * txeo::Tensor<int> t({3}, {2, 3, 4});
     * TensorOp<int>::subtract_by(10, t);
     * // t now contains [8, 7, 6]
     *@endcode
     */
    static const T &subtract_by(const T &left, txeo::Tensor<T> &right);

    /**
     * @brief Returns the multiplication of a tensor and a scalar
     *
     * @param left Tensor operand
     * @param right Scalar multiplier
     * @return txeo::Tensor<T> Result
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<double> a({3}, {1.5, 2.5, 3.5});
     * auto b = TensorOp<double>::multiply(a, 2.0);  // Result: [3.0, 5.0, 7.0]
     * @endcode
     */
    static txeo::Tensor<T> multiply(const txeo::Tensor<T> &left, const T &right);

    /**
     * @brief Multiplies the tensor by a scalar (in-place)
     *
     * @param left Operand to be modified
     * @param right Scalar multiplier
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> a({2,2}, {1, 2, 3, 4});
     * TensorOp<int>::multiply_by(a, 3);  // a becomes [[3,6],[9,12]]
     * @endcode
     */
    static txeo::Tensor<T> &multiply_by(txeo::Tensor<T> &left, const T &right);

    /**
     * @brief Element-wise division of tensor by scalar (out-of-place)
     *
     * @param left Dividend tensor
     * @param right Scalar divisor
     * @return New tensor with division results
     *
     * **Example Usage:**
     *@code
     * txeo::Tensor<float> t({2, 2}, {10.0f, 20.0f, 30.0f, 40.0f});
     * auto result = TensorOp<float>::divide(t, 2.0f);
     * // result contains [5.0, 10.0, 15.0, 20.0]
     *@endcode
     */
    static txeo::Tensor<T> divide(const txeo::Tensor<T> &left, const T &right);

    /**
     * @brief In-place element-wise division of tensor by scalar
     * @param left Tensor to modify (dividend)
     * @param right Scalar divisor
     *
     * **Example Usage:**
     *@code
     * txeo::Tensor<double> t({3}, {15.0, 30.0, 45.0});
     * TensorOp<double>::divide_by(t, 3.0);
     * // t now contains [5.0, 10.0, 15.0]
     *@endcode
     */
    static txeo::Tensor<T> &divide_by(txeo::Tensor<T> &left, const T &right);

    /**
     * @brief Element-wise division of scalar by tensor (out-of-place)
     *
     * @param left Scalar dividend
     * @param right Tensor divisor (shape NxMx...)
     * @return New tensor with same shape as input
     *
     * **Example Usage:**
     *@code
     * txeo::Tensor<int> A({3}, {2, 5, 10});
     * auto B = TensorOp<int>::divide(100, A);
     * // B contains [50, 20, 10], shape [3]
     * @endcode
     */
    static txeo::Tensor<T> divide(const T &left, const txeo::Tensor<T> &right);

    /**
     * @brief In-place element-wise division of scalar by tensor elements
     *
     * @param scalar Scalar dividend
     * @param tensor Tensor divisor (modified with results)
     *
     * **Example Usage:**
     *@code
     * txeo::Tensor<int> t({4}, {2, 5, 10, 25});
     * TensorOp<int>::divide_by(100, t);
     * // t now contains [50, 20, 10, 4]
     * @endcode
     */
    static txeo::Tensor<T> &divide_by(const T &scalar, txeo::Tensor<T> &tensor);

    /**
     * @brief Returns the element-wise product (Hadamard Product) of two tensors
     *
     * @param left Left operand
     * @param right Right operand
     * @return txeo::Tensor<T> Result
     *
     * @exception TensorOpError Thrown if shapes mismatch
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<float> a({2,2}, {1.0f, 2.0f, 3.0f, 4.0f});
     * txeo::Tensor<float> b({2,2}, {2.0f, 3.0f, 4.0f, 5.0f});
     * auto c = TensorOp<float>::hadamard_prod(a, b);  // Result: [[2,6],[12,20]]
     * @endcode
     */
    static txeo::Tensor<T> hadamard_prod(const txeo::Tensor<T> &left, const txeo::Tensor<T> &right);

    /**
     * @brief Performs element-wise multiplication of the left operand by the right operand
     * (in-place)
     *
     * @param left Operand to be modified
     * @param right Operand to multiply with
     *
     * @exception TensorOpError Thrown if shapes mismatch
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<double> a({3}, {2.0, 3.0, 4.0});
     * txeo::Tensor<double> b({3}, {5.0, 6.0, 7.0});
     * TensorOp<double>::hadamard_prod_by(a, b);  // a becomes [10.0, 18.0, 28.0]
     * @endcode
     */
    static txeo::Tensor<T> &hadamard_prod_by(txeo::Tensor<T> &left, const txeo::Tensor<T> &right);

    /**
     * @brief Element-wise Hadamard division (out-of-place)
     *
     * @param left Dividend tensor (shape NxMx...)
     * @param right Divisor tensor (must match left's shape)
     * @return New tensor with same shape as inputs
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<float> A({2}, {10.0f, 20.0f});
     * txeo::Tensor<float> B({2}, {2.0f, 5.0f});
     * auto C = TensorOp<float>::hadamard_div(A, B);
     * // C contains [5.0, 4.0], shape [2]
     * @endcode
     */
    static txeo::Tensor<T> hadamard_div(const txeo::Tensor<T> &left, const txeo::Tensor<T> &right);

    /**
     * @brief In-place element-wise Hadamard division
     *
     * @param left Dividend tensor (modified with results)
     * @param right Divisor tensor
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<float> a({3}, {10.0f, 20.0f, 30.0f});
     * txeo::Tensor<float> b({3}, {2.0f, 5.0f, 10.0f});
     * TensorOp<float>::hadamard_div_by(a, b);
     * // a now contains [5.0, 4.0, 3.0]
     * @endcode
     */
    static txeo::Tensor<T> &hadamard_div_by(txeo::Tensor<T> &left, const txeo::Tensor<T> &right);

    /**
     * @brief Computes the inner product of two tensors.
     *
     * @param left The first tensor.
     * @param right The second tensors.
     * @return The dot product of the two tensors.
     *
     * @throws txeo::TensorOpError
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> left({1, 2, 3});  // Tensor [1, 2, 3]
     * txeo::Tensor<int> right({4, 5, 6}); // Tensor [4, 5, 6]
     * auto result = TensorOp<int>::inner(left, right);
     * // result = 1*4 + 2*5 + 3*6 = 32
     * @endcode
     */
    static T inner(const txeo::Tensor<T> &left, const txeo::Tensor<T> &right);

    /**
     * @brief Computes the matrix product of two second order tensors.
     *
     * @param left The left tensor (m x n).
     * @param right The right tensor (n x p).
     * @return A new tensor (m x p) containing the result of the matrix product.
     *
     * @throws txeo::TensorOpError
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> left({2, 3}, {1, 2, 3, 4, 5, 6});  // 2x3 tensor
     * txeo::Tensor<int> right({3, 2}, {7, 8, 9, 10, 11, 12});  // 3x2 tensor
     * auto result = TensorOp<int>::product_tensors(left, right);
     * // result = [ [58, 64], [139, 154] ]
     * @endcode
     */
    static txeo::Tensor<T> product_tensors(const txeo::Tensor<T> &left,
                                           const txeo::Tensor<T> &right);

    /**
     * @brief Computes the matrix product of two matrices.
     *
     * @param left The left matrix (m x n).
     * @param right The right matrix (n x p).
     * @return A new matrix (m x p) containing the result of the matrix product.
     *
     * @throws txeo::TensorOpError
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<int> left(2, 3, {1, 2, 3, 4, 5, 6});  // 2x3 matrix
     * txeo::Matrix<int> right(3, 2, {7, 8, 9, 10, 11, 12});  // 3x2 matrix
     * auto result = TensorOp<int>::product(left, right);
     * // result = [ [58, 64], [139, 154] ]
     * @endcode
     */
    static txeo::Matrix<T> product(const txeo::Matrix<T> &left, const txeo::Matrix<T> &right);

    /**
     * @brief Computes the matrix product of a matrix and a vector.
     *
     * @param left The left matrix (m x n).
     * @param right The right vector (n).
     * @return A new tensor (m x 1) containing the result of the matrix product.
     *
     * @throws txeo::TensorOpError
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<int> left(2, 3, {1, 2, 3, 4, 5, 6});  // 2x3 matrix
     * txeo::Vector<int> right(3, {7, 8, 9});  // vector of size 3
     * auto result = TensorOp<int>::product(left, right);
     * @endcode
     */
    static txeo::Tensor<T> product(const txeo::Matrix<T> &left, const txeo::Vector<T> &right);

  private:
    TensorOp() = default;
};

/**
 * @brief Exceptions concerning @ref txeo::TensorOp
 *
 */
class TensorOpError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo
#endif
