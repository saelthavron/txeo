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

    explicit Tensor(const txeo::TensorShape &shape, const T &fill_value);
    explicit Tensor(txeo::TensorShape &&shape, const T &fill_value);
    explicit Tensor(const std::vector<int64_t> &shape, const T &fill_value);
    explicit Tensor(std::vector<int64_t> &&shape, const T &fill_value);

    [[nodiscard]] const txeo::TensorShape &shape() const;
    constexpr std::type_identity_t<T> type() const;
    [[nodiscard]] int order() const;
    [[nodiscard]] size_t dim() const;
    [[nodiscard]] size_t number_of_elements() const { return this->dim(); };
    [[nodiscard]] size_t memory_size() const;
    [[nodiscard]] const T *data() const;
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

    void reshape(const std::vector<int64_t> &shape);
    void reshape(const txeo::TensorShape &shape);
    Tensor<T> flatten() const;
    void fill(const T &value);

    template <c_numeric N>
    void fill_with_uniform_random(const N &min, const N &max, size_t seed1, size_t seed2);

    template <c_numeric N>
    void fill_with_normal_random(const N &min, const N &max, size_t seed1, size_t seed2);

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

} // namespace txeo

#endif // TENSOR_H

// construir um identity factory
// gpt void map(std::function<T(T)> func);
// gpt Tensor<T> transpose(const std::vector<size_t> &perm) const;
// gpt friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor);
// deep Iterators for STL Compatibility
