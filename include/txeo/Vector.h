#ifndef VECTOR_H
#define VECTOR_H
#include "txeo/types.h"
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
    explicit Vector();
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

    /**
     * @brief Normalizes the vector in-place using specified normalization method
     * @param type Normalization type to apply:
     *             - MIN_MAX: Scales values to [0, 1] range
     *             - Z_SCORE: Standardizes to mean=0, std=1
     *
     * @code
     * // Example 1: Min-max normalization
     * Vector<double> vec({2.0, 4.0, 6.0});
     * vec.normalize(txeo::NormalizationType::MIN_MAX);
     * // vec becomes [0.0, 0.5, 1.0] (original min=2, max=6)
     *
     * // Example 2: Z-score normalization
     * Vector<float> v({2.0f, 4.0f, 6.0f});
     * v.normalize(txeo::NormalizationType::Z_SCORE);
     * // v becomes approximately [-1.2247, 0.0, 1.2247]
     * // (μ=4.0, σ≈1.63299)
     * @endcode
     */
    void normalize(txeo::NormalizationType type);

    void reshape(const txeo::TensorShape &shape);

    void reshape(const std::vector<size_t> &shape) { this->reshape(txeo::TensorShape(shape)); };

    void reshape(const std::initializer_list<size_t> &shape) {
      this->reshape(std::vector<size_t>(shape));
    };

    /**
     * @brief Returns the size of the vector.
     *
     * @return The total number of elements in the vector.
     *
     * **Example Usage:**
     * @code
     * txeo::Vector<int> vec(3);
     * size_t size = vec.size();  // size = 3
     * @endcode
     */
    [[nodiscard]] size_t size() const { return this->dim(); }

    /**
     * @brief Converts a tensor to a vector by moving data.
     *
     * This function moves the data from the input tensor to a new vector.
     *
     * @param tensor The input tensor to convert. Must be 1-dimensional.
     * @return A vector created from the input tensor.
     *
     * @throws std::VectorError if the tensor is not 1-dimensional.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({4}, {1, 2, 3, 4});  // 1D tensor with shape (4)
     * auto vector = Vector<int>::to_vector(std::move(tensor));  // Convert to vector
     * // vector contains [1, 2, 3, 4]
     * @endcode
     */
    static Vector<T> to_vector(txeo::Tensor<T> &&tensor);

    /**
     * @brief Creates a vector from a tensor (performs copy).
     *
     * @param tensor The input tensor to copy. Must be first order.
     * @return A vector created from the input tensor.
     *
     * @throws std::VectorError.
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> tensor({4}, {1, 2, 3, 4});  // 1D tensor with shape (4)
     * auto vector = Vector<int>::to_vector(tensor);  // Convert to vector
     * // vector contains [1, 2, 3, 4]
     * @endcode
     */
    static Vector<T> to_vector(const txeo::Tensor<T> &tensor);

    /**
     * @brief Converts a vector to a tensor by moving data.
     *
     * @param vector The input vector to convert.
     * @return A tensor created from the input vector.
     *
     * **Example Usage:**
     * @code
     * txeo::Vector<int> vector({1, 2, 3, 4});  // Vector with 4 elements
     * auto tensor = Vector<int>::to_tensor(std::move(vector));  // Convert to tensor
     * // tensor shape: (4)
     * @endcode
     */
    static txeo::Tensor<T> to_tensor(Vector<T> &&vector);

    /**
     * @brief Creates a tensor from a vector (performs copy).
     *
     * @param vector The input vector to copy.
     * @return A tensor created from the input vector.
     *
     * **Example Usage:**
     * @code
     * txeo::Vector<int> vector({1, 2, 3, 4});  // Vector with 4 elements
     * auto tensor = Vector<int>::to_tensor(vector);  // Convert to tensor
     * // tensor shape: (4)
     * @endcode
     */
    static txeo::Tensor<T> to_tensor(const Vector<T> &vector);

    template <typename U>
    friend txeo::Vector<U> operator+(const txeo::Vector<U> &left, const txeo::Vector<U> &right);

    template <typename U>
    friend txeo::Vector<U> operator+(const txeo::Vector<U> &left, const U &right);

    template <typename U>
    friend txeo::Vector<U> operator-(const txeo::Vector<U> &left, const txeo::Vector<U> &right);

    template <typename U>
    friend txeo::Vector<U> operator-(const txeo::Vector<U> &left, const U &right);

    template <typename U>
    friend txeo::Vector<U> operator-(const U &left, const txeo::Vector<U> &right);

    template <typename U>
    friend txeo::Vector<U> operator*(const txeo::Vector<U> &vector, const U &scalar);

    template <typename U>
    friend txeo::Vector<U> operator*(const U &scalar, const txeo::Vector<U> &vector);

    template <typename U>
    friend txeo::Vector<U> operator/(const txeo::Vector<U> &left, const U &right);

    template <typename U>
    friend txeo::Vector<U> operator/(const U &left, const txeo::Vector<U> &right);

  private:
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