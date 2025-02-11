#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#pragma once

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <type_traits>

#include "TensorShape.h"

namespace txeo {

template <typename T>
concept c_numeric = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

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

    template <typename P>
    void create_from_shape(P &&shape);

    void fill_data_shape(const std::initializer_list<std::initializer_list<T>> &list,
                         std::vector<T> &flat_data, std::vector<size_t> &shape);

    void fill_data_shape(
        const std::initializer_list<std::initializer_list<std::initializer_list<T>>> &list,
        std::vector<T> &flat_data, std::vector<size_t> &shape);

    void check_indexes(const std::vector<size_t> &indexes);

  public:
    explicit Tensor() = delete;
    Tensor(const Tensor &tensor);
    Tensor(Tensor &&tensor) noexcept;
    ~Tensor();

    Tensor &operator=(const Tensor &tensor);
    Tensor &operator=(Tensor &&tensor) noexcept;
    bool operator==(const Tensor &tensor);
    bool operator!=(const Tensor &tensor);

    template <typename U>
    friend std::ostream &operator<<(std::ostream &os, const Tensor<U> &tensor);

    /**
     * @brief Constructs a Tensor from a specified @ref txeo::TensorShape
     *
     * @param shape
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
     * @param shape
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
     * @param shape vector of dimensions
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
     * @param shape vector of dimensions
     *
     * **Example Usage:**
     * @code
     * #include <iostream>
     * #include "txeo/Tensor.h"
     *
     * int main() {
     *     txeo::Tensor<int> tensor({3, 4});
     *
     *     std::cout << "Tensor created with shape: " << tensor.shape() << std::endl;
     *     return 0;
     * }
     * @endcode
     */
    explicit Tensor(std::vector<size_t> &&shape);

    /**
     * @brief Constructs a Tensor from a specified @ref txeo::TensorShape and fills it with a value
     *
     * @param shape
     * @param fill_value
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
     * @param shape
     * @param fill_value
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
     * @param shape
     * @param fill_value
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
     * @param shape
     * @param fill_value
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
    explicit Tensor(std::vector<size_t> &&shape, const T &fill_value);

    /**
     * @brief Constructs a Tensor object from a specified @ref txeo::TensorShape and fills it with a
     * std::vector of values in a row-major scheme
     *
     * @param shape
     * @param values
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
     * @brief Constructs a second order Tensor from a nested std::initializer_list.
     *
     *
     * @param values nested initializer list
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
     *
     * @param values nested initializer list
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
     * @brief Returns the raw data of this tensor for reading
     *
     * @return const T*
     */
    [[nodiscard]] const T *data() const;

    /**
     * @brief Returns a view ot this tensor from a specified range of dimensions of the first axis
     *
     * @details This function creates a new tensor that shares the content of this tensor according
     * to the specified parameters.
     *
     * @param first_axis_begin begin index along the first axis (inclusive).
     * @param first_axis_end end index along the first axis (exclusive).
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
    void share_from(const Tensor<T> &tensor, const txeo::TensorShape &shape);

    template <typename U>
    [[nodiscard]] bool is_equal_shape(const Tensor<U> &other) const;

    T &operator()();

    template <typename... Args>
      requires(std::convertible_to<Args, size_t> && ...)
    T &operator()(Args... args);

    T &at();

    template <typename... Args>
    T &at(Args... args);

    const T &operator()() const;

    template <typename... Args>
      requires(std::convertible_to<Args, size_t> && ...)
    const T &operator()(Args... args) const;

    const T &at() const;

    template <typename... Args>
    const T &at(Args... args) const;

    void reshape(const txeo::TensorShape &shape);
    void reshape(const std::vector<size_t> &shape);
    Tensor<T> flatten() const;
    void fill(const T &value);

    void fill_with_uniform_random(const T &min, const T &max, size_t seed1, size_t seed2);
    void fill_with_uniform_random(const T &min, const T &max);

    void shuffle();

    void squeeze();

    Tensor<T> &operator=(const T &value);
    T *data();

    Tensor<T> clone() const;
};

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
inline T &Tensor<T>::at(Args... args) {
  if (this->order() != sizeof...(Args))
    throw TensorError("The number of axes specified and the order of this tensor do no match.");
  check_indexes({static_cast<size_t>(args)...});

  return (*this)(args...);
}

template <typename T>
template <typename... Args>
inline const T &Tensor<T>::at(Args... args) const {
  if (this->order() != sizeof...(Args))
    throw TensorError("The number of axes specified and the order of this tensor do no match.");
  check_indexes({static_cast<size_t>(args)...});

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

// Colocar usage na doc do TensorShape
// Colocar throws na doc do TensorShape
// Colocar throws na doc do Tensor
// construir um identity factory
// gpt void map(std::function<T(T)> func);
// gpt Tensor<T> transpose(const std::vector<size_t> &perm) const;
// deep Iterators for STL Compatibility
