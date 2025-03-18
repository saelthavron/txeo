#ifndef MATRIX_H
#define MATRIX_H
#pragma once

#include "txeo/Tensor.h"
#include "txeo/TensorShape.h"
#include "txeo/types.h"

#include <cstddef>
#include <initializer_list>

namespace txeo {

namespace detail {
class TensorHelper;
}

template <typename T>
class Predictor;

template <typename T>
class TensorAgg;

template <typename T>
class TensorPart;

template <typename T>
class TensorOp;

template <typename T>
class TensorFunc;

/**
 * @class Matrix
 * @brief A class representing a matrix, derived from Tensor.
 *
 * This class provides constructors for creating matrices with various initialization methods.
 * It inherits from `txeo::Tensor<T>` and specializes it for 2nd-order data.
 *
 * @tparam T The data type of the matrix elements (e.g., int, double).
 */
template <typename T>
class Matrix : public txeo::Tensor<T> {
  public:
    Matrix() = default;
    ~Matrix() = default;

    Matrix(const Matrix &matrix) : txeo::Tensor<T>{matrix} {};
    Matrix(Matrix &&matrix) noexcept : txeo::Tensor<T>{std::move(matrix)} {};

    Matrix &operator=(const Matrix &matrix) {
      txeo::Tensor<T>::operator=(matrix);
      return *this;
    };

    Matrix &operator=(Matrix &&matrix) noexcept {
      txeo::Tensor<T>::operator=(std::move(matrix));
      return *this;
    };

    /**
     * @brief Constructs a matrix with the specified row and column sizes.
     *
     * This constructor creates a matrix of the specified row and column sizes, with uninitialized
     * elements.
     *
     * @param row_size The number of rows in the matrix.
     * @param col_size The number of columns in the matrix.
     *
     * @code
     * txeo::Matrix<int> matrix(2, 3);  // Creates a 2x3 matrix
     * @endcode
     */
    explicit Matrix(size_t row_size, size_t col_size)
        : txeo::Tensor<T>{txeo::TensorShape({row_size, col_size})} {};

    /**
     * @brief Constructs a matrix with the specified row and column sizes and a fill value.
     *
     * @param row_size The number of rows in the matrix.
     * @param col_size The number of columns in the matrix.
     * @param fill_value The value to fill the matrix with.
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<int> matrix(2, 3, 5);  // Creates a 2x3 matrix filled with 5
     * @endcode
     */
    explicit Matrix(size_t row_size, size_t col_size, const T &fill_value)
        : txeo::Tensor<T>({row_size, col_size}, fill_value) {};

    /**
     * @brief Constructs a matrix with the specified row and column sizes and values.
     *
     * @param row_size The number of rows in the matrix.
     * @param col_size The number of columns in the matrix.
     * @param values The values to initialize the matrix with.
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<int> matrix(2, 3, {1, 2, 3, 4, 5, 6});  // Creates a 2x3 matrix with values [1,
     * 2, 3, 4, 5, 6]
     * @endcode
     */
    explicit Matrix(size_t row_size, size_t col_size, const std::vector<T> &values)
        : txeo::Tensor<T>({row_size, col_size}, values) {};

    /**
     * @brief Constructs a matrix with the specified row and column sizes and initializer list.
     *
     * @param row_size The number of rows in the matrix.
     * @param col_size The number of columns in the matrix.
     * @param values The values to initialize the matrix with.
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<int> matrix(2, 3, {1, 2, 3, 4, 5, 6});  // Creates a 2x3 matrix with values [1,
     * 2, 3, 4, 5, 6]
     * @endcode
     */
    explicit Matrix(size_t row_size, size_t col_size, const std::initializer_list<T> &values)
        : txeo::Tensor<T>({row_size, col_size}, std::vector<T>(values)) {};

    /**
     * @brief Constructs a matrix from a nested initializer list.
     *
     * @param values The values to initialize the matrix with.
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<int> matrix({{1, 2, 3}, {4, 5, 6}});  // Creates a 2x3 matrix with values [1, 2,
     * 3, 4, 5, 6]
     * @endcode
     */
    explicit Matrix(const std::initializer_list<std::initializer_list<T>> &values)
        : txeo::Tensor<T>(values) {};

    /**
     * @brief Constructs a matrix by moving data from a Tensor.
     *
     * @param tensor The Tensor to move data from.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});
     * txeo::Matrix<int> matrix(std::move(tensor));  // Moves data from tensor to matrix
     * @endcode
     */
    explicit Matrix(txeo::Tensor<T> &&tensor);

    /**
     * @brief Returns the size of the matrix.
     *
     * @return The total number of elements in the matrix.
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<int> matrix(2, 3);
     * size_t size = matrix.size();  // size = 6
     * @endcode
     */
    [[nodiscard]] size_t size() const { return txeo::Tensor<T>::dim(); };

    /**
     * @brief Normalizes matrix columns in-place using specified normalization method
     * @param type Normalization type to apply (MIN_MAX or Z_SCORE)
     *
     * **Example Usage:**
     * @code
     * // Example: Min-max normalize columns
     * Matrix<double> mat({{10.0, 20.0},  // Column 1: [10, 30]
     *                    {30.0, 40.0}}); // Column 2: [20, 40]
     * mat.normalize_columns(txeo::NormalizationType::MIN_MAX);
     * // Column 1 becomes [0.0, 1.0]
     * // Column 2 becomes [0.0, 1.0]
     *
     * // Example: Z-score normalize columns
     * Matrix<float> m({{1.0f, 4.0f},   // Column 1: μ=2.0, σ=1.414
     *                 {3.0f, 6.0f}});  // Column 2: μ=5.0, σ=1.414
     * m.normalize_columns(txeo::NormalizationType::Z_SCORE);
     * // Column 1 becomes [-0.707, 0.707]
     * // Column 2 becomes [-0.707, 0.707]
     * @endcode
     */
    void normalize_columns(txeo::NormalizationType type);

    /**
     * @brief Normalizes matrix rows in-place using specified normalization method
     * @param type Normalization type to apply (MIN_MAX or Z_SCORE)
     *
     * **Example Usage:**
     * @code
     * // Example: Min-max normalize rows
     * Matrix<double> mat({{10.0, 20.0, 30.0},  // Row 1: [10, 20, 30]
     *                    {5.0, 10.0, 15.0}});  // Row 2: [5, 10, 15]
     * mat.normalize_rows(txeo::NormalizationType::MIN_MAX);
     * // Row 1 becomes [0.0, 0.5, 1.0]
     * // Row 2 becomes [0.0, 0.5, 1.0]
     *
     * // Example: Z-score normalize rows
     * Matrix<float> m({{2.0f, 4.0f},   // Row 1: μ=3.0, σ=1.414
     *                {6.0f, 8.0f}});  // Row 2: μ=7.0, σ=1.414
     * m.normalize_rows(txeo::NormalizationType::Z_SCORE);
     * // Row 1 becomes [-0.707, 0.707]
     * // Row 2 becomes [-0.707, 0.707]
     * @endcode
     */
    void normalize_rows(txeo::NormalizationType type);

    void reshape(const txeo::TensorShape &shape);

    void reshape(const std::vector<size_t> &shape) { this->reshape(txeo::TensorShape(shape)); };

    void reshape(const std::initializer_list<size_t> &shape) {
      this->reshape(std::vector<size_t>(shape));
    };

    /**
     * @brief Converts a tensor to a matrix by moving data.
     *
     * @param tensor The input tensor to convert. Must be 2-dimensional.
     * @return A matrix created from the input tensor.
     *
     * @throws std::MatrixError
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});  // 2D tensor with shape (2, 3)
     * auto matrix = Matrix<int>::to_matrix(std::move(tensor));  // Convert to matrix
     * // matrix shape: (2, 3)
     * @endcode
     */
    static Matrix<T> to_matrix(txeo::Tensor<T> &&tensor);

    /**
     * @brief Creates a matrix from a tensor (preforms copy).
     *
     * @param tensor The input tensor to copied. Must be 2-dimensional.
     * @return A matrix created from the input tensor.
     *
     * @throws std::MatrixError
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});  // 2D tensor with shape (2, 3)
     * auto matrix = Matrix<int>::to_matrix(std::move(tensor));  //
     * // matrix shape: (2, 3)
     * @endcode
     */
    static Matrix<T> to_matrix(const txeo::Tensor<T> &tensor);

    /**
     * @brief Converts a matrix to a tensor by moving data.
     *
     * @param matrix The input matrix to convert. Must be second order.
     * @return A tensor created from the input matrix.
     *
     * @throws std::MatrixError
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<int> matrix(2, 3, {1, 2, 3, 4, 5, 6});  // 2x3 matrix
     * auto tensor = Matrix<int>::to_tensor(std::move(matrix));  // Convert to tensor
     * // tensor shape: (2, 3)
     * @endcode
     */
    static txeo::Tensor<T> to_tensor(Matrix<T> &&matrix);

    /**
     * @brief Creates a tensor from a matrix (performs copy).
     *
     * @param matrix The input matrix to copy. Must be second order.
     * @return A tensor created from the input matrix.
     *
     * @throws std::MatrixError
     *
     * **Example Usage:**
     * @code
     * txeo::Matrix<int> matrix(2, 3, {1, 2, 3, 4, 5, 6});  // 2x3 matrix
     * auto tensor = Matrix<int>::to_tensor(matrix);  // Convert to tensor
     * // tensor shape: (2, 3)
     * @endcode
     */
    static txeo::Tensor<T> to_tensor(const Matrix<T> &matrix);

  private:
    friend class txeo::Predictor<T>;
    friend class txeo::TensorAgg<T>;
    friend class txeo::TensorPart<T>;
    friend class txeo::TensorOp<T>;
    friend class txeo::TensorFunc<T>;
    friend class txeo::detail::TensorHelper;
};

/**
 * @brief Exceptions concerning @ref txeo::Matrix
 *
 */
class MatrixError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo
#endif