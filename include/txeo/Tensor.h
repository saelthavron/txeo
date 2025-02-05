#ifndef TENSOR_H
#define TENSOR_H
#include <cstdint>
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
    explicit Tensor(const txeo::TensorShape &shape);
    explicit Tensor(txeo::TensorShape &&shape);
    explicit Tensor(const std::vector<int64_t> &shape);
    explicit Tensor(std::vector<int64_t> &&shape);

    [[nodiscard]] txeo::TensorShape shape() const;

    ~Tensor();
};

#endif // TENSOR_H
}