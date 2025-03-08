#ifndef VECTOR_H
#define VECTOR_H
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

/**
 * @class Vector
 * @brief A class representing a vector, derived from Tensor.
 *
 * This class provides constructors for creating vectors with various initialization methods.
 * It inherits from `txeo::Tensor<T>` and specializes it for order one data.
 *
 * @tparam T The data type of the vector elements (e.g., int, double).
 */
template <typename T>
class Vector : public txeo::Tensor<T> {
  public:
    ~Vector() = default;

    Vector(const Vector &Vector) : txeo::Tensor<T>{Vector} {};
    Vector(Vector &&Vector) noexcept : txeo::Tensor<T>{std::move(Vector)} {};

    Vector &operator=(const Vector &Vector) {
      txeo::Tensor<T>::operator=(Vector);
      return *this;
    };

    Vector &operator=(Vector &&Vector) noexcept {
      txeo::Tensor<T>::operator=(std::move(Vector));
      return *this;
    };

    /**
     * @brief Constructs a vector with the specified dimension.
     *
     *
     * @param dim The dimension (size) of the vector.
     *
     ** **Example Usage:**
     * @code
     * txeo::Vector<int> vector(3);  // Creates a vector of size 3
     * @endcode
     */
    explicit Vector(size_t dim) : txeo::Tensor<T>{txeo::TensorShape({dim})} {};

    /**
     * @brief Constructs a vector with the specified dimension and fill value.
     *
     * @param dim The dimension (size) of the vector.
     * @param fill_value The value to fill the vector with.
     *
     ** **Example Usage:**
     * @code
     * txeo::Vector<int> vector(3, 5);  // Creates a vector of size 3 filled with 5
     * @endcode
     */
    explicit Vector(size_t dim, const T &fill_value) : txeo::Tensor<T>({dim}, fill_value) {};

    /**
     * @brief Constructs a vector with the specified dimension and values.
     *
     * @param dim The dimension (size) of the vector.
     * @param values The values to initialize the vector with.
     *
     ** **Example Usage:**
     * @code
     * txeo::Vector<int> vector(3, {1, 2, 3});  // Creates a vector of size 3 with values [1, 2, 3]
     * @endcode
     */
    explicit Vector(size_t dim, const std::vector<T> &values) : txeo::Tensor<T>({dim}, values) {};

    /**
     * @brief Constructs a vector with the specified dimension and initializer list.
     *
     * @param dim The dimension (size) of the vector.
     * @param values The values to initialize the vector with.
     *
     ** **Example Usage:**
     * @code
     * txeo::Vector<int> vector(3, {1, 2, 3});  // Creates a vector of size 3 with values [1, 2, 3]
     * @endcode
     */
    explicit Vector(size_t dim, const std::initializer_list<T> &values)
        : txeo::Tensor<T>({dim}, std::vector<T>(values)) {};

    /**
     * @brief Constructs a vector from an initializer list.
     *
     * @param values The values to initialize the vector with.
     *
     ** **Example Usage:**
     * @code
     * txeo::Vector<int> vector({1, 2, 3});  // Creates a vector of size 3 with values [1, 2, 3]
     * @endcode
     */
    explicit Vector(const std::initializer_list<T> &values)
        : txeo::Tensor<T>({values.size()}, std::vector<T>(values)) {};

    /**
     * @brief Constructs a vector by moving data from a Tensor.
     *
     * @param tensor The Tensor to move data from.
     *
     ** **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({3}, {1, 2, 3});
     * txeo::Vector<int> vector(std::move(tensor));  // Moves data from tensor to vector
     * @endcode
     */
    explicit Vector(txeo::Tensor<T> &&tensor);

    void reshape(const txeo::TensorShape &shape);

  private:
    Vector() = default;

    friend class txeo::Predictor<T>;
    friend class txeo::TensorAgg<T>;
    friend class txeo::TensorPart<T>;
    friend class txeo::TensorOp<T>;
    friend class txeo::TensorFunc<T>;
    friend class txeo::detail::TensorHelper;
};

/**
 * @brief Exceptions concerning @ref txeo::Vector
 *
 */
class VectorError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo

#endif