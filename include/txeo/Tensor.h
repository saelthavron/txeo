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
concept numeric = std::is_arithmetic_v<T>;

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
    [[nodiscard]] const std::type_identity_t<T> *data() const;
    Tensor<T> slice(size_t first_axis_begin, size_t first_axis_end) const;
    void copy_from(const Tensor<T> &tensor, const txeo::TensorShape &shape);

    template <typename U>
    [[nodiscard]] bool is_equal_shape(const Tensor<U> &other) const;

    T &operator()();
    T &operator()(size_t x);
    T &operator()(size_t x, size_t y);
    T &operator()(size_t x, size_t y, size_t z);
    T &operator()(size_t x, size_t y, size_t z, size_t k);
    T &operator()(size_t x, size_t y, size_t z, size_t k, size_t w);

    T &at();
    T &at(size_t x);
    T &at(size_t x, size_t y);
    T &at(size_t x, size_t y, size_t z);
    T &at(size_t x, size_t y, size_t z, size_t k);
    T &at(size_t x, size_t y, size_t z, size_t k, size_t w);

    const T &operator()() const;
    const T &operator()(size_t x) const;
    const T &operator()(size_t x, size_t y) const;
    const T &operator()(size_t x, size_t y, size_t z) const;
    const T &operator()(size_t x, size_t y, size_t z, size_t k) const;
    const T &operator()(size_t x, size_t y, size_t z, size_t k, size_t w) const;

    const T &at() const;
    const T &at(size_t x) const;
    const T &at(size_t x, size_t y) const;
    const T &at(size_t x, size_t y, size_t z) const;
    const T &at(size_t x, size_t y, size_t z, size_t k) const;
    const T &at(size_t x, size_t y, size_t z, size_t k, size_t w) const;

    template <typename... Args>
    T &element_at(Args... args);

    template <typename... Args>
    const T &element_at(Args... args) const;

    void reshape(const std::vector<int64_t> &shape);
    void reshape(const txeo::TensorShape &shape);
    Tensor<T> flatten() const;
    void fill(const T &value);

    template <numeric N>
    void fill_with_uniform_random(const N &min, const N &max, size_t seed1, size_t seed2);

    Tensor<T> &operator=(const T &value);
    T *data();
};

class TensorError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

} // namespace txeo

#endif // TENSOR_H

// gpt void map(std::function<T(T)> func);
// gpt Tensor<T> transpose(const std::vector<size_t> &perm) const;
// gpt friend std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor);
// deep Iterators for STL Compatibility
// deep Static Factory Methods: tensors with zeros, ones, random values, etc.
