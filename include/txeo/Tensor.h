#ifndef TENSOR_H
#define TENSOR_H
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
};

#endif // TENSOR_H
}

// Assignment Operator: Assigns the value of one tensor to another.

// DataType(): Returns the data type of the tensor.

// Shape(): Returns the shape of the tensor as a tensorflow::TensorShape object.

// NumElements(): Returns the total number of elements in the tensor.

// Slice(): Extracts a sub-tensor from the original tensor.

// Reshape(): Changes the shape of the tensor without altering its data.

// Tensor::FromString(): Converts a string representation of a tensor into an actual tensor.

// Tensor::FromStringWithDefault(): Similar to FromString, but with a default value if the string is
// empty.

// Tensor::FromStringWithDefaultAndType(): Converts a string representation of a tensor with a
// specified data type.

// dtype()	Returns the data type (DT_FLOAT, etc.)
// shape()	Returns the tensor shape
// tensor_data()	Accesses raw memory
// flat<T>()	Access tensor as a 1D array
// matrix<T>()	Access tensor as a 2D matrix
// NumElements()	Returns the number of elements
// IsInitialized()	Checks if the tensor has memory allocated
// IsSameSize(other)	Checks if two tensors have the same shape
// set_shape(new_shape)	Changes the tensor shape (if valid)