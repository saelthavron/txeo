#ifndef MATRIX_H
#define MATRIX_H
#pragma once

#include "txeo/Tensor.h"
#include "txeo/TensorShape.h"

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

template <typename T>
class Matrix : public txeo::Tensor<T> {
  public:
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

    explicit Matrix(size_t row_size, size_t col_size)
        : txeo::Tensor<T>{txeo::TensorShape({row_size, col_size})} {};

    explicit Matrix(size_t row_size, size_t col_size, const T &fill_value)
        : txeo::Tensor<T>({row_size, col_size}, fill_value) {};

    explicit Matrix(size_t row_size, size_t col_size, const std::vector<T> &values)
        : txeo::Tensor<T>({row_size, col_size}, values) {};

    explicit Matrix(size_t row_size, size_t col_size, const std::initializer_list<T> &values)
        : txeo::Tensor<T>({row_size, col_size}, std::vector<T>(values)) {};

    explicit Matrix(const std::initializer_list<std::initializer_list<T>> &values)
        : txeo::Tensor<T>(values) {};

    explicit Matrix(const txeo::Tensor<T> &tensor);

    explicit Matrix(txeo::Tensor<T> &&tensor);

    [[nodiscard]] size_t size() const { return txeo::Tensor<T>::dim(); };

  private:
    Matrix() = default;

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