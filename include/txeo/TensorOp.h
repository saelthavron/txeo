#ifndef TENSOROP_H
#define TENSOROP_H

#pragma once

#include "txeo/Tensor.h"

namespace txeo {

/**
 * @brief Class that centralizes mathematical operations and functions on tensors
 *
 * @tparam T type of the tensor or tensors involved
 */
template <typename T>
class TensorOp {

  public:
    TensorOp() = delete;
    TensorOp(const TensorOp &) = delete;
    TensorOp(TensorOp &&) = delete;
    TensorOp &operator=(const TensorOp &) = delete;
    TensorOp &operator=(TensorOp &&) = delete;
    ~TensorOp();

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
     * @brief Adds the left operand with the right operand (in-place)
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
    static void sum_by(txeo::Tensor<T> &left, const txeo::Tensor<T> &right);

    static txeo::Tensor<T> sum(const txeo::Tensor<T> &left, const T &right);
    static void sum_by(txeo::Tensor<T> &left, const T &right);

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
    static void subtract_by(txeo::Tensor<T> &left, const txeo::Tensor<T> &right);

    static txeo::Tensor<T> subtract(const txeo::Tensor<T> &left, const T &right);
    static void subtract_by(txeo::Tensor<T> &left, const T &right);

    static txeo::Tensor<T> subtract(const T &left, const txeo::Tensor<T> &right);
    static void subtract_by(const T &left, txeo::Tensor<T> &right);

    /**
     * @brief Returns the multiplication of a tensor and a scalar
     *
     * @param tensor Tensor operand
     * @param scalar Scalar multiplier
     * @return txeo::Tensor<T> Result
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<double> a({3}, {1.5, 2.5, 3.5});
     * auto b = TensorOp<double>::multiply(a, 2.0);  // Result: [3.0, 5.0, 7.0]
     * @endcode
     */
    static txeo::Tensor<T> multiply(const txeo::Tensor<T> &tensor, const T &scalar);

    /**
     * @brief Multiplies the tensor by a scalar (in-place)
     *
     * @param tensor Operand to be modified
     * @param scalar Scalar multiplier
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> a({2,2}, {1, 2, 3, 4});
     * TensorOp<int>::multiply_by(a, 3);  // a becomes [[3,6],[9,12]]
     * @endcode
     */
    static void multiply_by(txeo::Tensor<T> &tensor, const T &scalar);

    static txeo::Tensor<T> divide(const txeo::Tensor<T> &tensor, const T &scalar);
    static void divide_by(txeo::Tensor<T> &tensor, const T &scalar);

    static txeo::Tensor<T> divide(const T &scalar, const txeo::Tensor<T> &tensor);
    static void divide_by(const T &scalar, txeo::Tensor<T> &tensor);

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
    static void hadamard_prod_by(txeo::Tensor<T> &left, const txeo::Tensor<T> &right);

    static txeo::Tensor<T> hadamard_div(const txeo::Tensor<T> &left, const txeo::Tensor<T> &right);

    static void hadamard_div_by(txeo::Tensor<T> &left, const txeo::Tensor<T> &right);

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
     * @param tensor Tensor to be modified
     * @param exponent Exponent of the potentiation
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<double> a({2}, {3.0, 4.0});
     * TensorOp<double>::power_elem_by(a, 3.0);  // a becomes [27.0, 64.0]
     * @endcode
     */
    static void power_elem_by(txeo::Tensor<T> &tensor, const T &exponent);
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

// / Element-wise division between two tensors
// static txeo::Tensor<T> divide(const txeo::Tensor<T> &left, const txeo::Tensor<T> &right);

// // In-place element-wise division
// static void divide_by(txeo::Tensor<T> &left, const txeo::Tensor<T> &right);

// // Tensor divided by scalar
// static txeo::Tensor<T> divide(const txeo::Tensor<T> &tensor, const T &scalar);

// // Scalar divided by tensor
// static txeo::Tensor<T> divide_scalar(const T &scalar, const txeo::Tensor<T> &tensor);

// // In-place tensor divided by scalar
// static void divide_by(txeo::Tensor<T> &tensor, const T &scalar);

// // Add scalar to tensor
// static txeo::Tensor<T> add_scalar(const txeo::Tensor<T> &tensor, const T &scalar);
// static void add_scalar_by(txeo::Tensor<T> &tensor, const T &scalar);

// // Subtract scalar from tensor
// static txeo::Tensor<T> subtract_scalar(const txeo::Tensor<T> &tensor, const T &scalar);
// static void subtract_scalar_by(txeo::Tensor<T> &tensor, const T &scalar);

// // Element-wise tensor^tensor exponentiation
// static txeo::Tensor<T> power_elem_tensor(const txeo::Tensor<T> &base, const txeo::Tensor<T>
// &exponent); static void power_elem_tensor_by(txeo::Tensor<T> &base, const txeo::Tensor<T>
// &exponent);

// // Unary operations (example: absolute value)
// static txeo::Tensor<T> abs(const txeo::Tensor<T> &tensor);
// static void abs_by(txeo::Tensor<T> &tensor);