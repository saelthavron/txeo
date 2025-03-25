#ifndef TENSORPART_H
#define TENSORPART_H
#include "txeo/Matrix.h"
#pragma once

#include "txeo/Tensor.h"

namespace txeo {

/**
 * @class TensorPart
 * @brief A utility class for partitioning tensors.
 *
 * This class provides static methods for operations such as unstacking tensors along
 * a specified axis and slicing tensors along the first axis.
 *
 * @tparam T The data type of the tensor elements (e.g., int, double).
 */
template <typename T>
class TensorPart {
  public:
    TensorPart(const TensorPart &) = delete;
    TensorPart(TensorPart &&) = delete;
    TensorPart &operator=(const TensorPart &) = delete;
    TensorPart &operator=(TensorPart &&) = delete;
    ~TensorPart();

    /**
     * @brief Unstacks a tensor along a specified axis into a list of tensors.
     *
     *
     * @tparam T The data type of the tensor elements.
     * @param tensor The input tensor to unstack.
     * @param axis The axis along which to unstack the tensor. Must be a valid axis for the input
     * tensor.
     *
     * @return std::vector<txeo::Tensor<T>> A list of tensors resulting from the unstack operation.
     *
     * @throws txeo::TensorPartError
     *
     * **Example Usage:**
     * @code
     * #include "txeo/TensorPart.h"
     * #include "txeo/Tensor.h"
     * #include <iostream>
     *
     * int main() {
     *     // Create a 3D tensor with shape (2, 2, 3)
     *     txeo::Tensor<int> tensor({{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}});
     *
     *     // Unstack the tensor along axis 0
     *     auto unstacked_tensors = txeo::TensorPart<int>::unstack(tensor, 0);
     *
     *     // Print the unstacked tensors
     *     for (size_t i = 0; i < unstacked_tensors.size(); ++i) {
     *         std::cout << "Unstacked Tensor " << i << ":\n" << unstacked_tensors[i] << std::endl;
     *     }
     *
     *     return 0;
     * }
     * @endcode
     *
     * **Output:**
     * @code
     * Unstacked Tensor 0:
     * [[1 2 3]
     *  [4 5 6]]
     *
     * Unstacked Tensor 1:
     * [[7 8 9]
     *  [10 11 12]]
     * @endcode
     */
    static std::vector<txeo::Tensor<T>> unstack(const txeo::Tensor<T> &tensor, size_t axis);

    /**
     * @brief Returns a view of the tensor from a specified range of dimensions of the first axis
     *
     * @details This function creates a new tensor that views the content of the tensor according
     * to the specified parameters. There is no element copying.
     *
     * @param first_axis_begin Initial index along the first axis (inclusive).
     * @param first_axis_end Final index along the first axis (exclusive).
     * @return Tensor<T>
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
     *     txeo::Tensor<int> sliced_tensor = TensorPart::slice(tensor, 0, 2);
     *
     *     std::cout << "Sliced Tensor: " << sliced_tensor << std::endl; // {{1, 2}, {4, 5}, {7, 8}}
     *     return 0;
     * }
     * @endcode
     */
    static txeo::Tensor<T> slice(const txeo::Tensor<T> &tensor, size_t first_axis_begin,
                                 size_t first_axis_end);

    /**
     * @brief Increments the dimension of the tensor at the specified axis
     *
     * @param tensor Tensor which elements will generate the modified tensor
     * @param axis Position where new dimension will be inserted
     * @param value Value to fill the new dimension elements with
     * @return A new modified modified tensor
     *
     * **Example Usage:**
     * @code
     * // Add new dimension to 2x3 matrix making it 2x1x3
     * txeo::Tensor<float> t({2, 3}, {1,2,3,4,5,6});
     * auto resp = txeo::TensorPart<float>::increase_dimension(t,1, -1.0f);
     * // New shape: [2, 4]
     * // resp(0,2) == -1.0f, resp(1,2) == -1.0f
     * @endcode
     */
    static txeo::Tensor<T> increase_dimension(const txeo::Tensor<T> &tensor, size_t axis, T value);

    /**
     * @brief Increments the dimension of the tensor at the specified axis (in-place)
     *
     * @param tensor Tensor which shape will be altered
     * @param axis Position where new dimension will be inserted
     * @param value Value to fill the new dimension elements with
     * @return Reference to the modified tensor
     *
     * **Example Usage:**
     * @code
     * // Add new dimension to 2x3 matrix making it 2x1x3
     * txeo::Tensor<float> t({2, 3}, {1,2,3,4,5,6});
     * txeo::TensorPart<float>::increase_dimension_by(t,1, -1.0f);
     * // New shape: [2, 4]
     * // t(0,2) == -1.0f, t(1,2) == -1.0f
     * @endcode
     */
    static txeo::Tensor<T> &increase_dimension_by(txeo::Tensor<T> &tensor, size_t axis, T value);

    /**
     * @brief Creates a submatrix containing specified columns
     *
     * @param matrix Source matrix
     * @param cols Vector of column indices to select
     * @return New matrix with selected columns
     * @throws TensorPartError
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<double> mat(2, 3, {1.1, 2.2, 3.3, 4.4, 5.5, 6.6});
     * auto sub = TensorPart<double>::sub_matrix_cols(mat, {0, 2});
     * // Resulting 2x2 matrix:
     * // [1.1, 3.3]
     * // [4.4, 6.6]
     * @endcode
     */
    static txeo::Matrix<T> sub_matrix_cols(const txeo::Matrix<T> &matrix,
                                           const std::vector<size_t> &cols);

    /**
     * @brief Creates a submatrix excluding the specified columns
     *
     * @param matrix Source matrix
     * @param cols Vector of column indices to exclude
     * @return New matrix with excluded columns
     *
     * @throws TensorPartError
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<double> mat(2, 3, {1.1, 2.2, 3.3, 4.4, 5.5, 6.6});
     * auto sub = TensorPart<double>::sub_matrix_cols_exclude(mat, {0, 2});
     * // Resulting 2x1 matrix:
     * // [2.2]
     * // [5.5]
     * @endcode
     */
    static txeo::Matrix<T> sub_matrix_cols_exclude(const txeo::Matrix<T> &matrix,
                                                   const std::vector<size_t> &cols);

    /**
     * @brief Creates a submatrix containing specified rows
     *
     * @param matrix Source matrix
     * @param rows Vector of row indices to select
     * @return New matrix with selected rows
     *
     * @throws TensorPartError
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<int> mat(3, 2, {1,2,3,4,5,6});
     * auto sub = TensorPart<int>::sub_matrix_rows(mat, {2, 0});
     * // Resulting 2x2 matrix:
     * // [5, 6]
     * // [1, 2]
     * @endcode
     */
    static txeo::Matrix<T> sub_matrix_rows(const txeo::Matrix<T> &matrix,
                                           const std::vector<size_t> &rows);

  private:
    TensorPart() = default;
};

/**
 * @brief Exceptions concerning @ref txeo::Tensor
 *
 */
class TensorPartError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo

#endif