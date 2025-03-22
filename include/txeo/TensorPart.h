#ifndef TENSORPART_H
#define TENSORPART_H
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

    static txeo::Tensor<T> increment_dimension(const txeo::Tensor<T> &tensor, size_t axis, T value);

    static txeo::Tensor<T> &increment_dimension_by(txeo::Tensor<T> &tensor, size_t axis, T value);

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