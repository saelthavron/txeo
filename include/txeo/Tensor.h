#ifndef TENSOR_H
#define TENSOR_H
#include <cstddef>
#include <cstdint>
#include <type_traits>
#pragma once

#include "TensorShape.h"
#include <memory>

namespace txeo {

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

  public:
    explicit Tensor() = delete;
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

    [[nodiscard]] const txeo::TensorShape &shape();
    constexpr std::type_identity_t<T> type() const;
    [[nodiscard]] int order() const;
    [[nodiscard]] int64_t dim() const;
    [[nodiscard]] size_t size_in_bytes() const;
    [[nodiscard]] const std::type_identity_t<T> *data() const;

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

    template <typename U>
    [[nodiscard]] bool is_equal_shape(const txeo::Tensor<U> &other) const;
};

#endif // TENSOR_H
}

class TensorError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// Slice(): Extracts a sub-tensor from the original tensor.

// Reshape(): Changes the shape of the tensor without altering its data.

// Tensor::FromString(): Converts a string representation of a tensor into an actual tensor.

// Tensor::FromStringWithDefault(): Similar to FromString, but with a default value if the string is
// empty.

// Tensor::FromStringWithDefaultAndType(): Converts a string representation of a tensor with a
// specified data type.

// flat<T>()	Access tensor as a 1D array
// matrix<T>()	Access tensor as a 2D matrix
// IsInitialized()	Checks if the tensor has memory allocated
// set_shape(new_shape)	Changes the tensor shape (if valid)