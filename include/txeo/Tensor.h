#ifndef TENSOR_H
#define TENSOR_H
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

#include "TensorShape.h"

namespace txeo {

template <typename T>
concept c_numeric = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

// std::is_arithmetic_v<bool>            == true  and
// std::is_arithmetic_v<char>            == true  and
// std::is_arithmetic_v<char const>      == true  and
// std::is_arithmetic_v<int>             == true  and
// std::is_arithmetic_v<int const>       == true  and
// std::is_arithmetic_v<float>           == true  and
// std::is_arithmetic_v<float const>     == true  and
// std::is_arithmetic_v<std::size_t>     == true  and

/**
 * @brief Implements the mathematical concept of tensor, which is a magnitude of multiple order. A
 * tensor of order zero is defined to be a scalar, of order one a vector, of order two a matrix.
 * Each order of the tensor has a dimension.
 *
 */
template <typename T>
class Tensor {
  private:
    struct Impl;
    std::unique_ptr<Impl> _impl{nullptr};

    template <typename P>
    void create_from_shape(P &&shape);

    template <typename P>
    void create_from_vector(P &&shape);

    void check_indexes(const std::vector<size_t> &indexes);

    explicit Tensor();

  public:
    Tensor(const Tensor &tensor);
    Tensor(Tensor &&tensor) noexcept;
    ~Tensor();

    Tensor &operator=(const Tensor &tensor);
    Tensor &operator=(Tensor &&tensor) noexcept;
    bool operator==(const Tensor &tensor);
    bool operator!=(const Tensor &tensor);

    explicit Tensor(const txeo::TensorShape &shape);
    explicit Tensor(txeo::TensorShape &&shape);
    explicit Tensor(const std::vector<int64_t> &shape);
    explicit Tensor(std::vector<int64_t> &&shape);

    [[nodiscard]] const txeo::TensorShape &shape() const;
    constexpr std::type_identity_t<T> type() const;
    [[nodiscard]] int order() const;
    [[nodiscard]] int64_t dim() const;
    [[nodiscard]] int64_t number_of_elements() const { return this->dim(); };
    [[nodiscard]] size_t memory_size() const;
    [[nodiscard]] const T *data() const;
    Tensor<T> slice(size_t first_axis_begin, size_t first_axis_end) const;
    void copy_from(const Tensor<T> &tensor, const txeo::TensorShape &shape);

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

    void reshape(const std::vector<int64_t> &shape);
    void reshape(const txeo::TensorShape &shape);
    Tensor<T> flatten() const;
    void fill(const T &value);

    template <c_numeric N>
    void fill_with_uniform_random(const N &min, const N &max, size_t seed1, size_t seed2);

    Tensor<T> &operator=(const T &value);
    T *data();
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
  int size = this->order();
  size_t accum_sizes{1};
  size_t flat_index{indexes[size - 1]};

  for (size_t i = size - 1; i > 0; --i) {
    accum_sizes *= this->shape().axis_dim(i);
    flat_index += indexes[i - 1] * accum_sizes;
  }
  return this->data()[flat_index];
}

template <typename T>
template <typename... Args>
  requires(std::convertible_to<Args, size_t> && ...)
inline const T &Tensor<T>::operator()(Args... args) const {
  size_t indexes[] = {static_cast<size_t>(args)...};
  int size = this->order();
  size_t accum_sizes{1};
  size_t flat_index{indexes[size - 1]};

  for (size_t i = size - 1; i > 0; --i) {
    accum_sizes *= this->shape().axis_dim(i);
    flat_index += indexes[i - 1] * accum_sizes;
  }
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

} // namespace txeo

#endif // TENSOR_H

// Implementar o esquema que ele sugeriu: int_64 por dentro e size_t por fora
// Tentar colocar os strides como variável da classe Tensor [criar stride em TensorShape]
// deep Static Factory Methods: tensors with zeros, ones, random values, etc.
// gpt void map(std::function<T(T)> func);
// gpt Tensor<T> transpose(const std::vector<size_t> &perm) const;
// gpt friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor);
// deep Iterators for STL Compatibility
