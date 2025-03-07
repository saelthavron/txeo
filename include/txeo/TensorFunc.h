#ifndef TENSORFUNC_H
#define TENSORFUNC_H

#include "txeo/Tensor.h"
#pragma once

namespace txeo {

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

  private:
    TensorFunc() = default;
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