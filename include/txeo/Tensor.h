#ifndef TENSOR_H
#define TENSOR_H

#pragma once

#include "TensorShape.h"
#include "txeo/TensorIterator.h"

#include <cstddef>
#include <exception>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <vector>

namespace txeo {

template <typename T>
concept c_numeric = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

template <typename T>
class Predictor;

/**
 * @brief Implements the mathematical concept of tensor, which is a magnitude of multiple order. A
 * tensor of order zero is defined to be a scalar, of order one a vector, of order two a matrix and
 * so on. Each order of the tensor has a dimension. Elements are addressed via multidimensional
 * indexing.
 *
 */
template <typename T>
class Tensor {
  private:
    struct Impl;
    std::unique_ptr<Impl> _impl{nullptr};

    friend class Predictor<T>;

    template <typename P>
    void create_from_shape(P &&shape);

    void fill_data_shape(const std::initializer_list<std::initializer_list<T>> &list,
                         std::vector<T> &flat_data, std::vector<size_t> &shape);

    void fill_data_shape(
        const std::initializer_list<std::initializer_list<std::initializer_list<T>>> &list,
        std::vector<T> &flat_data, std::vector<size_t> &shape);

    void check_indexes(const std::vector<size_t> &indexes);

    explicit Tensor();

  public:
    /**
     * @note This copy constructor performs a deep copy, behaving differently from TensorFlow C++.
     */
    Tensor(const Tensor &tensor);
    Tensor(Tensor &&tensor) noexcept;
    ~Tensor();

    /**
     * @note This copy assignment performs a deep copy, behaving differently from TensorFlow C++.
     */
    Tensor &operator=(const Tensor &tensor);
    Tensor &operator=(Tensor &&tensor) noexcept;
    bool operator==(const Tensor &tensor);
    bool operator!=(const Tensor &tensor);

    template <typename U>
    friend std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor);

    /**
     * @brief Constructs a tensor from a specified @ref txeo::TensorShape
     *
     * @param shape Shape of the constructed tensor
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     txeo::TensorShape shape({3, 4}); // Define a 3x4 tensor shape
     *     txeo::Tensor<int> tensor(shape); // Create a tensor with the given shape
     *
     *     std::cout << "Tensor created with shape: " << tensor.shape() << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit Tensor(const txeo::TensorShape &shape);

    /**
     * @brief Constructs a Tensor from a specified @ref txeo::TensorShape
     *
     * @param shape Shape of the constructed tensor
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor(txeo::TensorShape({3, 4})); // Move-construct a tensor with
     * shape 3x4
     *
     *     std::cout << "Tensor created with shape: " << tensor.shape() << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit Tensor(txeo::TensorShape &&shape);

    /**
     * @brief Constructs a Tensor from a specified shape std::vector
     *
     * @param shape Vector of dimensions
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     std::vector<size_t> shape = {3, 4};
     *     txeo::Tensor<int> tensor(shape); // Create a tensor with shape 3x4
     *
     *     std::cout << "Tensor created with shape: " << tensor.shape() << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit Tensor(const std::vector<size_t> &shape);

    /**
     * @brief Constructs a Tensor from a specified shape std::vector
     *
     * @param shape Vector of dimensions
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     txeo::vector<int> aux({3,4})
     *     txeo::Tensor<int> tensor(aux);
     *
     *     std::cout << "Tensor created with shape: " << tensor.shape() << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit Tensor(std::vector<size_t> &&shape);

    /**
     * @brief Constructs a Tensor from a specified shape std::vector
     *
     * @param shape Vector of dimensions
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor({3,4});
     *
     *     std::cout << "Tensor created with shape: " << tensor.shape() << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit Tensor(const std::initializer_list<size_t> &shape)
        : Tensor(std::vector<size_t>(shape)) {}

    /**
     * @brief Constructs a Tensor from a specified @ref txeo::TensorShape and fills it with a value
     *
     * @param shape Shape of the constructed tensor
     * @param fill_value Value of the elements of the constructed tensor
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     txeo::TensorShape shape({3, 4});
     *     txeo::Tensor<int> tensor(shape, 5); // Create a 3x4 tensor filled with 5
     *
     *     std::cout << "Tensor initialized with: " << tensor << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit Tensor(const txeo::TensorShape &shape, const T &fill_value);

    /**
     * @brief Constructs a Tensor from a specified @ref txeo::TensorShape and fills it with a value
     *
     * @param shape Shape of the constructed tensor
     * @param fill_value Value of the elements of the constructed tensor
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor(txeo::TensorShape({3, 4}), 7);
     *
     *     std::cout << "Tensor initialized with: " << tensor << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit Tensor(txeo::TensorShape &&shape, const T &fill_value);

    /**
     * @brief Constructs a Tensor from a specified shape std::vector and fills it with a value
     *
     * @param shape Shape of the constructed tensor
     * @param fill_value Value of the elements of the constructed tensor
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     std::vector<size_t> shape = {3, 4};
     *     txeo::Tensor<int> tensor(shape, 5); // Create a 3x4 tensor filled with 5
     *
     *     std::cout << "Tensor initialized with: " << tensor << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit Tensor(const std::vector<size_t> &shape, const T &fill_value);

    /**
     * @brief Constructs a Tensor from a specified shape std::vector and fills it with a value
     *
     * @param shape Shape of the constructed tensor
     * @param fill_value Value of the elements of the constructed tensor
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor(std::vector<size_t>({3, 4}), 7);
     *
     *     std::cout << "Tensor initialized with: " << tensor << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit Tensor(std::vector<size_t> &&shape, const T &fill_value);

    /**
     * @brief Constructs a Tensor from a specified initializer list and fills it with a value
     *
     * @param shape Shape of the constructed tensor
     * @param fill_value Value of the elements of the constructed tensor
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor({3, 4}, 7);
     *
     *     std::cout << "Tensor initialized with: " << tensor << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit Tensor(const std::initializer_list<size_t> &shape, const T &fill_value)
        : Tensor(std::vector<size_t>(shape), fill_value) {}

    /**
     * @brief Constructs a Tensor object from a specified @ref txeo::TensorShape and fills it with a
     * std::vector of values in a row-major scheme
     *
     * @param shape Shape of the constructed tensor
     * @param values Elements of the constructed tensor
     *
     * @throw  txeo::TensorError
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor(txeo::TensorShape({2, 3}), {1, 2, 3, 4, 5, 6}); // Create a 2x3
     * tensor initialized with given values
     *
     *     std::cout << "Tensor initialized with: " << tensor << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit Tensor(const txeo::TensorShape &shape, const std::vector<T> &values);

    /**
     * @brief Constructs a Tensor object from a specified std::vector<size_t> and fills it with a
     * std::vector of values in a row-major scheme
     *
     * @param shape Shape of the constructed tensor
     * @param values Elements of the constructed tensor
     *
     * @throw  txeo::TensorError
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor(std::vector<size_t>({2, 3}), {1, 2, 3, 4, 5, 6});
     *
     *     std::cout << "Tensor initialized with: " << tensor << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit Tensor(const std::vector<size_t> &shape, const std::vector<T> &values);

    /**
     * @brief Constructs a Tensor object from a specified initializer list and fills it with a
     * std::vector of values in a row-major scheme
     *
     * @param shape Shape of the constructed tensor
     * @param values Elements of the constructed tensor
     *
     * @throw  txeo::TensorError
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor({2, 3}, {1, 2, 3, 4, 5, 6});
     *
     *     std::cout << "Tensor initialized with: " << tensor << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit Tensor(const std::initializer_list<size_t> &shape, const std::vector<T> &values)
        : Tensor(std::vector<size_t>(shape), values) {}

    /**
     * @brief Constructs a second order Tensor from a nested std::initializer_list.
     *
     * @param values Nested initializer list
     *
     * @throw txeo::TensorError
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor{{1, 2, 3}, {4, 5, 6}};
     *     std::cout << tensor << std::endl; // Print the tensor
     *     return 0;
     * }
     * @endcode
     */
    explicit Tensor(const std::initializer_list<std::initializer_list<T>> &values);

    /**
     * @brief Constructs a third order Tensor from a nested std::initializer_list.
     *
     * @param values Nested initializer list
     *
     * @throw txeo::TensorError
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     // Creating a 3D tensor with shape (4,2,3)
     *     txeo::Tensor<int> tensor{
     *         {{1, 2, 3}, {4, 5, 6}},
     *         {{7, 8, 9}, {10, 11, 12}},
     *         {{-1, -2, -3}, {-4, -5, -6}},
     *         {{-7, -8, -9}, {-10, -11, -12}}
     *     };
     *
     *     std::cout << tensor << std::endl; // Print the tensor
     *     return 0;
     * }
     * @endcode
     *
     */
    explicit Tensor(
        const std::initializer_list<std::initializer_list<std::initializer_list<T>>> &values);

    /**
     * @brief Returns the shape of this tensor
     *
     * @return const txeo::TensorShape&
     */
    [[nodiscard]] const txeo::TensorShape &shape() const;

    /**
     * @brief Returns the data type of this tensor
     *
     * @return constexpr std::type_identity_t<T>
     */
    constexpr std::type_identity_t<T> type() const;

    /**
     * @brief Returns the order of this tensor
     *
     * @return int
     */
    [[nodiscard]] int order() const;

    /**
     * @brief Returns the dimension of this tensor
     *
     * @return size_t
     */
    [[nodiscard]] size_t dim() const;

    /**
     * @brief Returns the number of elements of this tensor, which corresponds to the dimension of
     * this tensor
     *
     * @return size_t
     */
    [[nodiscard]] size_t number_of_elements() const { return this->dim(); };

    /**
     * @brief Returns the number of bytes occupied by this tensor
     *
     * @return size_t
     */
    [[nodiscard]] size_t memory_size() const;

    /**
     * @brief Returns a view ot this tensor from a specified range of dimensions of the first axis
     *
     * @details This function creates a new tensor that views the content of this tensor according
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
     *     txeo::Tensor<int> sliced_tensor = tensor.slice(0, 2); // Extract rows 0 and 1
     *
     *     std::cout << "Sliced Tensor: " << sliced_tensor << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    Tensor<T> slice(size_t first_axis_begin, size_t first_axis_end) const;

    /**
     * @brief Views the content of the specified tensor according to the specified shape. There is
     * no element copying.
     *
     * @param tensor Viewed tensor
     * @param shape  New shape of this tensor
     */
    void view_of(const Tensor<T> &tensor, const txeo::TensorShape &shape);

    /**
     * @brief Compares the shape of this tensor with the shape of the specified tensor
     *
     * @tparam U Data type of the specfied tensor
     * @param other The tensor to compare
     * @return true if both this tensor and the other have the same shape
     * @return false otherwise
     *
     ** **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor1{{1, 2, 3}, {4, 5, 6}};
     *     txeo::Tensor<double> tensor2{{7.0, 8.0, 9.0}, {10.0, 11.0, 12.0}};
     *
     *     if (tensor1.is_equal_shape(tensor2)) {
     *         std::cout << "Tensors have the same shape." << std::endl;
     *     } else {
     *         std::cout << "Tensors have different shapes." << std::endl;
     *     }
     *     return 0;
     * }
     * @endcode
     *
     */
    template <typename U>
    [[nodiscard]] bool is_equal_shape(const Tensor<U> &other) const {
      return this->shape() == other.shape();
    };

    /**
     * @brief Accesses the value of this tensor if it is a scalar (order zero).
     *
     * @details Since this function does not perform any checking, it accesses the first value of
     * this tensor if it has order greater than zero.
     *
     * @return T& Value of the zeroth order tensor
     */
    T &operator()();

    /**
     * @brief Accesses an element of this tensor according to the specified indexes.
     *
     * @tparam Args A variadic list of indices that must be convertible to `size_t`.
     * @param args The indices specifying the position of the element.
     * @return Element at the specified indices.
     *
     * @note No checking is performed.
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor{{1, 2, 3}, {4, 5, 6}};
     *     tensor(1, 2) = 42; // Modify an element at row 1, column 2
     *
     *     std::cout << "Updated value: " << tensor(1, 2) << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    template <typename... Args>
      requires(std::convertible_to<Args, size_t> && ...)
    T &operator()(Args... args);

    /**
     * @brief Accesses the value of this tensor if it is a scalar (order zero).
     *
     * @return T& Value of the zeroth order tensor
     *
     * @throw  txeo::TensorError
     *
     * @note Order checking is performed.
     *
     */
    T &at();

    /**
     * @brief Accesses an element of this tensor according to the specified indexes.
     *
     * @tparam Args A variadic list of indices that must be convertible to `size_t`.
     * @param args The indices specifying the position of the element.
     * @return Element at the specified indices.
     *
     * @throw  txeo::TensorError
     *
     * @note Bound and order checkings are performed.
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor{{1, 2, 3}, {4, 5, 6}};
     *     tensor(1, 2) = 42; // Modify an element at row 1, column 2
     *
     *     std::cout << "Updated value: " << tensor(1, 2) << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    template <typename... Args>
      requires(std::convertible_to<Args, size_t> && ...)
    T &at(Args... args);

    /**
     * @brief Reads the value of this tensor if it is a scalar (order zero).
     *
     * @return T& Value of the zeroth order tensor
     *
     * @note Since this function does not perform any checking, it accesses the first value of
     * this tensor if it has order greater than zero.
     */
    const T &operator()() const;

    /**
     * @brief Reads an element of this tensor according to the specified indexes.
     *
     * @tparam Args A variadic list of indices that must be convertible to `size_t`.
     * @param args The indices specifying the position of the element.
     * @return Element at the specified indices.
     *
     * @note No checking is performed.
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor{{1, 2, 3}, {4, 5, 6}};
     *     tensor(1, 2) = 42; // Modify an element at row 1, column 2
     *
     *     std::cout << "Updated value: " << tensor(1, 2) << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    template <typename... Args>
      requires(std::convertible_to<Args, size_t> && ...)
    const T &operator()(Args... args) const;

    /**
     * @brief Reads the value of this tensor if it is a scalar (order zero). Order checking is
     * performed.
     *
     * @throw  txeo::TensorError
     *
     * @return T& Value of the zeroth order tensor
     */
    const T &at() const;

    /**
     * @brief Reads an element of this tensor according to the specified indexes.
     *
     * @tparam Args A variadic list of indices that must be convertible to `size_t`.
     * @param args The indices specifying the position of the element.
     * @return Element at the specified indices.
     *
     * @throw  txeo::TensorError
     *
     * @note Bound and order checkings are performed.
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor{{1, 2, 3}, {4, 5, 6}};
     *     tensor(1, 2) = 42; // Modify an element at row 1, column 2
     *
     *     std::cout << "Updated value: " << tensor(1, 2) << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    template <typename... Args>
      requires(std::convertible_to<Args, size_t> && ...)
    const T &at(Args... args) const;

    /**
     * @brief Reshapes this tensor if the specified shape defines a number of elements equal to
     * this tensor order.
     *
     * @param shape New shape for this tensor
     *
     * @throw  txeo::TensorError
     *
     * @note No copy is performed.
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor{{1, 2, 3, 4}}; // Shape (1, 4)
     *     tensor.reshape(txeo::TensorShape({2, 2})); // Change shape to (2, 2)
     *
     *     std::cout << "Reshaped Tensor: " << tensor << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    void reshape(const txeo::TensorShape &shape);

    /**
     * @brief Reshapes this tensor if the specified shape vector defines a number of elements equal
     * to this tensor order.
     *
     * @param shape New shape vector for this tensor
     *
     * @throw  txeo::TensorError
     *
     * @note No copy is performed.
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor{{1, 2, 3, 4}}; // Shape (1, 4)
     *     tensor.reshape(std::vector<size_t>{2, 2}); // Change shape to (2, 2)
     *
     *     std::cout << "Reshaped Tensor: " << tensor << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    void reshape(const std::vector<size_t> &shape);

    /**
     * @brief Reshapes this tensor if the specified shape vector defines a number of elements equal
     * to this tensor order.
     *
     * @param shape New shape vector for this tensor
     *
     * @throw  txeo::TensorError
     *
     * @note No copy is performed.
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     * #include "txeo/TensorShape.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor{{1, 2, 3, 4}}; // Shape (1, 4)
     *     tensor.reshape({2, 2}); // Change shape to (2, 2)
     *
     *     std::cout << "Reshaped Tensor: " << tensor << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    void reshape(const std::initializer_list<size_t> &shape) {
      this->reshape(std::vector<size_t>(shape));
    };

    /**
     * @brief Returns a first order reshaped view of this tensor.
     *
     * @return Tensor<T> First order tensor that views this tensor.
     *
     * @throw  txeo::TensorError
     *
     * @note No copy is performed.
     */
    Tensor<T> flatten() const;

    /**
     * @brief Fills this tensor with the specified value
     *
     * @param value Value to fill this tensor
     */
    void fill(const T &value);

    /**
     * @brief Fills the tensor with uniformly distributed random values ranging according to the
     * specified interval.
     *
     *
     * @tparam T The data type of the tensor elements.
     * @param min The minimum possible random value.
     * @param max The maximum possible random value.
     *
     * @throw txeo::TensorError
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     txeo::Tensor<float> tensor{3, 3}; // Create a 3x3 tensor
     *     tensor.fill_with_uniform_random(0.0f, 1.0f);
     *
     *     std::cout << "Tensor filled with random values: " << tensor << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    void fill_with_uniform_random(const T &min, const T &max);

    /**
     * @brief Fills the tensor with uniformly distributed random values ranging according to the
     * specified interval.
     *
     *
     * @tparam T The data type of the tensor elements.
     * @param min The minimum possible random value.
     * @param max The maximum possible random value.
     * @param seed1 The first seed for random number generation (in order to enable
     * reproducibility).
     * @param seed2 The second seed for random number generation (in order to enable
     * reproducibility).
     *
     * @throw txeo::TensorError
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     txeo::Tensor<float> tensor{3, 3}; // Create a 3x3 tensor
     *     tensor.fill_with_uniform_random(0.0f, 1.0f, 42, 123);
     *
     *     std::cout << "Tensor filled with random values: " << tensor << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    void fill_with_uniform_random(const T &min, const T &max, const size_t &seed1,
                                  const size_t &seed2);

    /**
     * @brief Shuffles the elements of this tensor
     *
     */
    void shuffle();

    /**
     * @brief Reshapes this tensor by removing all the axes of dimension one
     *
     * @note If the tensor has no axes of order one, its shape remains unchanged.
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor{{1}, {2}, {3}}; // Shape (3,1)
     *     tensor.squeeze(); // Removes singleton dimension
     *
     *     std::cout << "Squeezed Tensor: " << tensor << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    void squeeze();

    /**
     * @brief Assigns a specified value to this tensor elements
     *
     * @param value Value to be assigned
     * @return Tensor<T>&
     */
    Tensor<T> &operator=(const T &value);

    /**
     * @brief Acesses the raw data of this tensor
     *
     * @return const T*
     */
    T *data();

    /**
     * @brief Reads the raw data of this tensor
     *
     * @return const T*
     */
    [[nodiscard]] const T *data() const;

    /**
     * @brief Returns a clone of this tensor
     *
     * @return Tensor<T> A clone of this tensor
     *
     * @note A copy is performed
     */
    Tensor<T> clone() const;

    /**
     * @brief Returns the sum of two tensors
     *
     * @tparam U type of the tensors involved
     * @param left Left operand
     * @param right Right operand
     * @return txeo::Tensor Result
     *
     * @exception txeo::TensorOpError
     *
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<float> a({2,2}, {1,2,3,4});
     * txeo::Tensor<float> b({2,2}, {5,6,7,8});
     * auto c = a + b;  // Result: [[6,8],[10,12]]
     * @endcode
     */
    template <typename U>
    friend txeo::Tensor<U> operator+(const txeo::Tensor<U> &left, const txeo::Tensor<U> &right);

    template <typename U>
    friend txeo::Tensor<U> operator+(const txeo::Tensor<U> &left, const U &right);

    /**
     * @brief Returns the subtraction of two tensors
     *
     * @tparam U type of the tensors involved
     * @param left Left operand
     * @param right Right operand
     * @return txeo::Tensor Result
     *
     * @exception txeo::TensorOpError
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<double> a({2}, {5,7});
     * txeo::Tensor<double> b({2}, {1,3});
     * auto c = a - b;  // Result: [4,4]
     * @endcode
     */
    template <typename U>
    friend txeo::Tensor<U> operator-(const txeo::Tensor<U> &left, const txeo::Tensor<U> &right);

    /**
     * @brief Element-wise tensor-scalar addition operator

     * @tparam U Numeric type of tensor elements
     * @param left Input tensor
     * @param right Scalar to add
     * @return New tensor with elements: left[i] + right
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> t({2, 2}, {1, 2, 3, 4});
     * auto result = t + 5;
     * // result contains [6, 7, 8, 9] with shape [2, 2]
     * @endcode
     */
    template <typename U>
    friend txeo::Tensor<U> operator-(const txeo::Tensor<U> &left, const U &right);

    /**
     * @brief Element-wise scalar-tensor subtraction operator

     * @tparam U Numeric type of tensor elements
     * @param left Scalar value
     * @param right Input tensor
     * @return New tensor with elements: left - right[i]
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> t({4}, {2, 3, 5, 7});
     * auto result = 10 - t;
     * // result contains [8, 7, 5, 3]
     * @endcode
     */
    template <typename U>
    friend txeo::Tensor<U> operator-(const U &left, const txeo::Tensor<U> &right);

    /**
     * @brief Returns the scalar multiplication of a tensor
     *
     * @tparam U type of the tensor involved
     * @param tensor Operand to be multiplied
     * @param scalar Operand that multiplies
     * @return txeo::Tensor Result
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> a({3}, {1,2,3});
     * auto b = a * 2;  // Result: [2,4,6]
     * @endcode
     */
    template <typename U>
    friend txeo::Tensor<U> operator*(const txeo::Tensor<U> &tensor, const U &scalar);

    /**
     * @brief Element-wise division operator (tensor / scalar)
     *
     * @param left Input tensor
     * @param right Divisor value
     * @return New tensor with element-wise division results
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<double> t({2, 2}, {10.0, 20.0, 30.0, 40.0});
     * auto result = t / 2.0;
     * // result contains [5.0, 10.0, 15.0, 20.0] with shape [2, 2]
     * @endcode
     */
    template <typename U>
    friend txeo::Tensor<U> operator/(const txeo::Tensor<U> &left, const U &right);

    /**
     * @brief Element-wise scalar-tensor division operator
     *
     * @tparam U Numeric type of tensor elements
     * @param left Scalar dividend
     * @param right Tensor divisor
     * @return New tensor with elements: left / right[i]
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<double> t({3}, {2.0, 4.0, 5.0});
     * auto result = 100.0 / t;
     * // result contains [50.0, 25.0, 20.0]
     * @endcode
     */
    template <typename U>
    friend txeo::Tensor<U> operator/(const U &left, const txeo::Tensor<U> &right);

    /**
     * @brief Performs the element-wise multiplication (Hadamard Product) of this parameter on this
     * tensor
     *
     * @param tensor Tensor that multiplies
     * @return Tensor<T>& This tensor modified
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<float> a({2,2}, {1,2,3,4});
     * txeo::Tensor<float> b({2,2}, {2,3,4,5});
     * a.hadamard_prod_by(b);  // a becomes [[2,6],[12,20]]
     * @endcode
     *
     */
    Tensor<T> &hadamard_prod_by(const Tensor<T> &tensor);

    /**
     * @brief In-place element-wise Hadamard division (chainable)

     * @param tensor Divisor tensor (must match dimensions)
     * @return Reference to modified tensor for chaining
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<int> a({2, 2}, {10, 20, 30, 40});
     * txeo::Tensor<int> b({2, 2}, {2, 5, 10, 8});
     * a.hadamard_div_by(b);
     * // a now contains [5, 4, 3, 5]
     * @endcode
     */
    Tensor<T> &hadamard_div_by(const Tensor<T> &tensor);

    /**
     * @brief Performs the element-wise potentiation of this tensor
     *
     * @param exponent Exponent of the potentiation
     * @return Tensor<T>& This tensor modified
     *
     * **Example Usage:**
     * @code
     * txeo::Tensor<double> a({3}, {2,3,4});
     * a.power_elem_by(2);  // a becomes [4,9,16]
     * @endcode
     *
     * @note Handles negative exponents through reciprocal calculation*
     *
     */
    Tensor<T> &power_elem_by(const T &exponent);

    Tensor<T> &operator+=(const Tensor<T> &tensor);
    Tensor<T> &operator+=(const T &tensor);
    Tensor<T> &operator-=(const Tensor<T> &tensor);
    Tensor<T> &operator-=(const T &tensor);
    Tensor<T> &operator*=(const T &scalar);
    Tensor<T> &operator/=(const T &scalar);

    txeo::TensorIterator<T> begin();
    txeo::TensorIterator<T> end();
    txeo::TensorIterator<const T> begin() const;
    txeo::TensorIterator<const T> end() const;
};

/**
 * @brief Exceptions concerning @ref txeo::Tensor
 *
 */
class TensorError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

template <typename T>
template <typename... Args>
  requires(std::convertible_to<Args, size_t> && ...)
inline T &Tensor<T>::operator()(Args... args) {
  size_t indexes[] = {static_cast<size_t>(args)...};
  size_t size = this->order();
  auto *stride = this->shape().stride().data();
  size_t flat_index{indexes[size - 1]};

  for (size_t i = 0; i < size - 1; ++i)
    flat_index += indexes[i] * stride[i];

  return this->data()[flat_index];
}

template <typename T>
template <typename... Args>
  requires(std::convertible_to<Args, size_t> && ...)
inline const T &Tensor<T>::operator()(Args... args) const {
  size_t indexes[] = {static_cast<size_t>(args)...};
  size_t size = this->order();
  auto *stride = this->shape().stride().data();
  size_t flat_index{indexes[size - 1]};

  for (size_t i = 0; i < size - 1; ++i)
    flat_index += indexes[i] * stride[i];

  return this->data()[flat_index];
}

template <typename T>
template <typename... Args>
  requires(std::convertible_to<Args, size_t> && ...)
inline T &Tensor<T>::at(Args... args) {
  if (this->order() != sizeof...(Args))
    throw TensorError("The number of axes specified and the order of this tensor do no match.");
  try {
    check_indexes({static_cast<size_t>(args)...});
  } catch (std::exception e) {
    throw txeo::TensorError(e.what());
  }

  return (*this)(args...);
}

template <typename T>
template <typename... Args>
  requires(std::convertible_to<Args, size_t> && ...)
inline const T &Tensor<T>::at(Args... args) const {
  if (this->order() != sizeof...(Args))
    throw TensorError("The number of axes specified and the order of this tensor do no match.");
  try {
    check_indexes({static_cast<size_t>(args)...});
  } catch (std::exception e) {
    throw txeo::TensorError(e.what());
  }

  return (*this)(args...);
}

template <typename T>
void Tensor<T>::fill_data_shape(const std::initializer_list<std::initializer_list<T>> &list,
                                std::vector<T> &flat_data, std::vector<size_t> &shape) {

  shape.emplace_back(list.size());
  std::vector<std::initializer_list<T>> v_list(list);
  for (size_t i{1}; i < v_list.size(); ++i)
    if (v_list[i].size() != v_list[i - 1].size())
      throw txeo::TensorError("Tensor initialization is inconsistent!");

  shape.emplace_back(v_list[0].size());
  for (auto &item : v_list)
    for (auto &subitem : item)
      flat_data.emplace_back(subitem);
}

template <typename T>
void Tensor<T>::fill_data_shape(
    const std::initializer_list<std::initializer_list<std::initializer_list<T>>> &list,
    std::vector<T> &flat_data, std::vector<size_t> &shape) {
  shape.emplace_back(list.size());
  std::vector<std::initializer_list<std::initializer_list<T>>> v_list(list);
  for (size_t i{1}; i < v_list.size(); ++i)
    if (v_list[i].size() != v_list[i - 1].size())
      throw txeo::TensorError("Tensor initialization is inconsistent!");

  shape.emplace_back(v_list[0].size());
  bool emplaced{false};
  for (size_t i{0}; i < v_list.size(); ++i) {
    std::vector<std::initializer_list<T>> v_sublist(v_list[i]);
    for (size_t i{1}; i < v_sublist.size(); ++i)
      if (v_sublist[i].size() != v_sublist[i - 1].size())
        throw txeo::TensorError("Tensor initialization is inconsistent!");

    if (!emplaced) {
      shape.emplace_back(v_sublist[0].size());
      emplaced = true;
    }
    for (auto &item : v_sublist)
      for (auto &subitem : item)
        flat_data.emplace_back(subitem);
  }
}

} // namespace txeo

#endif // TENSOR_H
